#include "estimators/absolute_pose/joint_pose_estimator.h"
#include "base/graph.h"
#include "base/infinite_line.h"
#include "ceresbase/line_projection.h"
#include "ceresbase/point_projection.h"
#include "estimators/absolute_pose/pl_absolute_pose_ransac.h"
#include "optimize/hybrid_localization/cost_functions.h"

#include <PoseLib/camera_pose.h>
#include <PoseLib/solvers/p1p2ll.h>
#include <PoseLib/solvers/p2p1ll.h>
#include <PoseLib/solvers/p3ll.h>
#include <PoseLib/solvers/p3p.h>

namespace limap {

namespace estimators {

namespace absolute_pose {

namespace hybridloc = optimize::hybrid_localization;

std::pair<CameraPose, ransac_lib::RansacStatistics>
EstimateAbsolutePose_PointLine(const std::vector<Line3d> &l3ds,
                               const std::vector<int> &l3d_ids,
                               const std::vector<Line2d> &l2ds,
                               const std::vector<V3D> &p3ds,
                               const std::vector<V2D> &p2ds, const Camera &cam,
                               const JointPoseEstimatorOptions &options) {
  ransac_lib::LORansacOptions ransac_options = options.ransac_options;
  std::random_device rand_dev;
  if (options.random)
    ransac_options.random_seed_ = rand_dev();

  JointPoseEstimator solver(
      l3ds, l3d_ids, l2ds, p3ds, p2ds, cam, options.lineloc_config,
      options.cheirality_min_depth, options.cheirality_overlap_pixels);

  CameraPose best_model;
  ransac_lib::RansacStatistics ransac_stats;

  if (options.sample_solver_first) {
    PointLineAbsolutePoseRansac<CameraPose, std::vector<CameraPose>,
                                JointPoseEstimator>
        lomsac_solver_first;
    lomsac_solver_first.EstimateModel(ransac_options, solver, &best_model,
                                      &ransac_stats);
  } else {
    PointLineAbsolutePoseRansac<CameraPose, std::vector<CameraPose>,
                                JointPoseEstimator,
                                ransac_lib::UniformSampling<JointPoseEstimator>>
        lomsac_uniform;
    lomsac_uniform.EstimateModel(ransac_options, solver, &best_model,
                                 &ransac_stats);
  }
  return std::make_pair(best_model, ransac_stats);
}

JointPoseEstimator::JointPoseEstimator(
    const std::vector<Line3d> &l3ds, const std::vector<int> &l3d_ids,
    const std::vector<Line2d> &l2ds, const std::vector<V3D> &p3ds,
    const std::vector<V2D> &p2ds, const Camera &cam,
    const hybridloc::LineLocConfig &cfg, const double cheirality_min_depth,
    const double cheirality_overlap_pixels)
    : loc_config_(cfg), cheirality_min_depth_(cheirality_min_depth),
      cheirality_overlap_pixels_(cheirality_overlap_pixels) {
  l3ds_ = &l3ds;
  l3d_ids_ = &l3d_ids;
  l2ds_ = &l2ds;
  p3ds_ = &p3ds;
  p2ds_ = &p2ds;
  cam_ = Camera(cam.K());
  num_data_ = p3ds.size() + l3d_ids.size();

  if (loc_config_.cost_function == E3DLineLineDist2 ||
      loc_config_.cost_function == E3DPlaneLineDist2)
    loc_config_.points_3d_dist = true;
}

int JointPoseEstimator::MinimalSolver(const std::vector<int> &sample,
                                      std::vector<CameraPose> *poses) const {
  size_t sample_sz = min_sample_size();
  std::vector<V3D> xs(sample_sz), Xs(sample_sz), ls(sample_sz), Cs(sample_sz),
      Vs(sample_sz);

  size_t pt_idx = 0;
  size_t line_idx = 0;
  for (size_t k = 0; k < sample.size(); k++) {
    size_t idx = sample[k];
    if (idx < num_data_points()) {
      // we sampled a point correspondence
      xs[pt_idx] = cam_.K_inv() * p2ds_->at(idx).homogeneous();
      xs[pt_idx].normalize();
      Xs[pt_idx] = p3ds_->at(idx);
      pt_idx++;
    } else {
      // we sampled a line correspondence
      idx -= num_data_points();
      V3D normalized_start =
          (cam_.K_inv() * l2ds_->at(idx).start.homogeneous()).normalized();
      V3D normalized_end =
          (cam_.K_inv() * l2ds_->at(idx).end.homogeneous()).normalized();
      ls[line_idx] = normalized_start.cross(normalized_end);
      ls[line_idx].normalize();
      const Line3d &l3d = l3ds_->at(l3d_ids_->at(idx));
      Cs[line_idx] = l3d.start;
      Vs[line_idx] = l3d.direction().normalized();
      line_idx++;
    }
  }

  poses->clear();
  std::vector<poselib::CameraPose> poselib_poses;
  int ret = 0;
  if (pt_idx == 3 && line_idx == 0) {
    ret = poselib::p3p(xs, Xs, &poselib_poses);
  } else if (pt_idx == 2 && line_idx == 1) {
    ret = poselib::p2p1ll(xs, Xs, ls, Cs, Vs, &poselib_poses);
  } else if (pt_idx == 1 && line_idx == 2) {
    ret = poselib::p1p2ll(xs, Xs, ls, Cs, Vs, &poselib_poses);
  } else if (pt_idx == 0 && line_idx == 3) {
    ret = poselib::p3ll(ls, Cs, Vs, &poselib_poses);
  }

  for (auto &p : poselib_poses) {
    poses->emplace_back(CameraPose(p.q, p.t));
  }
  return ret;
}

int JointPoseEstimator::NonMinimalSolver(const std::vector<int> &sample,
                                         CameraPose *pose) const {
  if (sample.size() < non_minimal_sample_size())
    return 0;
  const int kNumSamples = static_cast<int>(sample.size());

  std::vector<V3D> opt_p3ds;
  std::vector<V2D> opt_p2ds;
  std::vector<Line3d> opt_l3ds;
  std::vector<std::vector<Line2d>> opt_l2ds;
  std::map<int, int> l3d_id_to_idx;

  for (int i = 0; i < sample.size(); i++) {
    int idx = sample[i];
    if (idx < num_data_points()) {
      opt_p3ds.emplace_back(p3ds_->at(idx));
      opt_p2ds.emplace_back(p2ds_->at(idx));
    } else {
      idx -= num_data_points();
      int l3d_id = l3d_ids_->at(idx);
      if (l3d_id_to_idx.find(l3d_id) == l3d_id_to_idx.end()) {
        l3d_id_to_idx[l3d_id] = opt_l3ds.size();
        opt_l3ds.emplace_back(l3ds_->at(l3d_id));
        opt_l2ds.resize(opt_l3ds.size());
      }
      opt_l2ds[l3d_id_to_idx[l3d_id]].emplace_back(l2ds_->at(idx));
    }
  }

  LineLocConfig config = loc_config_;
  config.weight_line = config.weight_point = 1.0;
  config.loss_function.reset(new ceres::TrivialLoss());
  JointLocEngine loc_engine(config);
  V4D kvec = V4D(cam_.params.data());
  loc_engine.Initialize(opt_l3ds, opt_l2ds, opt_p3ds, opt_p2ds, kvec,
                        pose->qvec, pose->tvec);
  loc_engine.SetUp();
  loc_engine.Solve();
  if (loc_engine.IsSolutionUsable()) {
    *pose = CameraPose(loc_engine.GetFinalR(), loc_engine.GetFinalT());
    return 1;
  }
  return 0;
}

// Evaluates the pose on the i-th data point.
double JointPoseEstimator::EvaluateModelOnPoint(const CameraPose &pose,
                                                int i) const {
  double res[4];
  if (i < num_data_points()) {
    // we sampled a point correspondence
    bool cheirality = cheirality_test_point(p3ds_->at(i), pose);
    if (!cheirality)
      return std::numeric_limits<double>::max();
    hybridloc::ReprojectionPointFunctor(p3ds_->at(i), p2ds_->at(i),
                                        loc_config_.points_3d_dist)(
        cam_.params.data(), pose.qvec.data(), pose.tvec.data(), res);
    return V2D(res[0], res[1]).squaredNorm();
  } else {
    // we sampled a line correspondence
    i -= num_data_points();
    const Line3d &l3d = l3ds_->at(l3d_ids_->at(i));
    bool cheirality = cheirality_test_line(l2ds_->at(i), l3d, pose);
    if (!cheirality)
      return std::numeric_limits<double>::max();
    hybridloc::ReprojectionLineFunctor(loc_config_.cost_function, ENoneWeight,
                                       l3d, l2ds_->at(i))(
        cam_.params.data(), pose.qvec.data(), pose.tvec.data(), res);
    if (getResidualNum(loc_config_.cost_function) == 2) {
      return V2D(res[0], res[1]).squaredNorm();
    } else if (getResidualNum(loc_config_.cost_function) == 4)
      return V4D(res[0], res[1], res[2], res[3]).squaredNorm();
    else
      throw std::runtime_error("Error! Not supported!");
  }
}

bool JointPoseEstimator::cheirality_test_point(const V3D &p3d,
                                               const CameraPose &pose) const {
  if (std::isnan(pose.qvec.norm()) || std::isnan(pose.tvec.norm()))
    return false;
  return pose.projdepth(p3d) >= cheirality_min_depth_;
}

// Cheirality check and also filtering on overlap and projected 2D length
bool JointPoseEstimator::cheirality_test_line(const Line2d &l2d,
                                              const Line3d &l3d,
                                              const CameraPose &pose) const {
  if (std::isnan(pose.qvec.norm()) || std::isnan(pose.tvec.norm()))
    return false;
  InfiniteLine3d line = InfiniteLine3d(l3d);
  InfiniteLine3d ray_start = InfiniteLine3d(
      pose.center(), CameraView(cam_, pose).ray_direction(l2d.start));
  InfiniteLine3d ray_end = InfiniteLine3d(
      pose.center(), CameraView(cam_, pose).ray_direction(l2d.end));
  // check ill-posed condition
  double angle_start = acos(std::abs(ray_start.d.dot(line.d))) * 180.0 / M_PI;
  if (angle_start < 1.0)
    return false;
  double angle_end = acos(std::abs(ray_end.d.dot(line.d))) * 180.0 / M_PI;
  if (angle_end < 1.0)
    return false;

  // unprojection
  V3D p_start = line.project_from_infinite_line(ray_start);
  if (!cheirality_test_point(p_start, pose))
    return false;
  V3D p_end = line.project_from_infinite_line(ray_end);
  if (!cheirality_test_point(p_end, pose))
    return false;

  // check 2D overlap
  CameraView camview(cam_, pose);
  Line2d proj_l2d = l3d.projection(camview);
  V2D p1 = proj_l2d.point_projection(l2d.start);
  V2D p2 = proj_l2d.point_projection(l2d.end);
  if (Line2d(p1, p2).length() < cheirality_overlap_pixels_)
    return false;

  return true;
}

void JointPoseEstimator::LeastSquares(const std::vector<int> &sample,
                                      CameraPose *pose) const {
  const int kNumSamples = static_cast<int>(sample.size());

  std::vector<V3D> opt_p3ds;
  std::vector<V2D> opt_p2ds;
  std::vector<Line3d> opt_l3ds;
  std::vector<std::vector<Line2d>> opt_l2ds;
  std::map<int, int> l3d_id_to_idx;

  for (int i = 0; i < sample.size(); i++) {
    int idx = sample[i];
    if (idx < num_data_points()) {
      opt_p3ds.emplace_back(p3ds_->at(idx));
      opt_p2ds.emplace_back(p2ds_->at(idx));
    } else {
      idx -= num_data_points();
      int l3d_id = l3d_ids_->at(idx);
      if (l3d_id_to_idx.find(l3d_id) == l3d_id_to_idx.end()) {
        l3d_id_to_idx[l3d_id] = opt_l3ds.size();
        opt_l3ds.emplace_back(l3ds_->at(l3d_id));
        opt_l2ds.resize(opt_l3ds.size());
      }
      opt_l2ds[l3d_id_to_idx[l3d_id]].emplace_back(l2ds_->at(idx));
    }
  }

  // Here the passed in config is used for LSQ
  JointLocEngine loc_engine(loc_config_);
  V4D kvec = V4D(cam_.params.data());
  loc_engine.Initialize(opt_l3ds, opt_l2ds, opt_p3ds, opt_p2ds, kvec,
                        pose->qvec, pose->tvec);
  loc_engine.SetUp();
  loc_engine.Solve();
  if (loc_engine.IsSolutionUsable())
    *pose = CameraPose(loc_engine.GetFinalR(), loc_engine.GetFinalT());
}

} // namespace absolute_pose

} // namespace estimators

} // namespace limap
