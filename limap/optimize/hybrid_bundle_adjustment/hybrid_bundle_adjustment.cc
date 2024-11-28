#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment.h"
#include "base/camera_models.h"
#include "ceresbase/parameterization.h"
#include "optimize/hybrid_bundle_adjustment/cost_functions.h"
#include "optimize/line_refinement/cost_functions.h"

#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>

namespace limap {

namespace optimize {

namespace hybrid_bundle_adjustment {

void HybridBAEngine::InitPointTracks(
    const std::vector<PointTrack> &point_tracks) {
  point_tracks_.clear();
  points_.clear();
  size_t num_tracks = point_tracks.size();
  for (size_t track_id = 0; track_id < num_tracks; ++track_id) {
    point_tracks_.insert(std::make_pair(track_id, point_tracks[track_id]));
    points_.insert(std::make_pair(track_id, point_tracks[track_id].p));
  }
}

void HybridBAEngine::InitPointTracks(
    const std::map<int, PointTrack> &point_tracks) {
  point_tracks_.clear();
  points_.clear();
  point_tracks_ = point_tracks;
  for (auto it = point_tracks.begin(); it != point_tracks.end(); ++it) {
    points_.insert(std::make_pair(it->first, it->second.p));
  }
}

void HybridBAEngine::InitLineTracks(const std::vector<LineTrack> &line_tracks) {
  line_tracks_.clear();
  lines_.clear();
  size_t num_tracks = line_tracks.size();
  for (size_t track_id = 0; track_id < num_tracks; ++track_id) {
    line_tracks_.insert(std::make_pair(track_id, line_tracks[track_id]));
    lines_.insert(std::make_pair(
        track_id, MinimalInfiniteLine3d(line_tracks[track_id].line)));
  }
}

void HybridBAEngine::InitLineTracks(
    const std::map<int, LineTrack> &line_tracks) {
  line_tracks_.clear();
  lines_.clear();
  line_tracks_ = line_tracks;
  for (auto it = line_tracks.begin(); it != line_tracks.end(); ++it) {
    lines_.insert(
        std::make_pair(it->first, MinimalInfiniteLine3d(it->second.line)));
  }
}

void HybridBAEngine::ParameterizeCameras() {
  for (const int &img_id : imagecols_.get_img_ids()) {
    double *params_data = imagecols_.params_data(img_id);
    double *qvec_data = imagecols_.qvec_data(img_id);
    double *tvec_data = imagecols_.tvec_data(img_id);

    if (!problem_->HasParameterBlock(params_data))
      continue;
    if (config_.constant_intrinsics) {
      problem_->SetParameterBlockConstant(params_data);
    } else if (config_.constant_principal_point) {
      int cam_id = imagecols_.camimage(img_id).cam_id;
      std::vector<int> const_idxs;
      auto principal_point_idxs = imagecols_.cam(cam_id).PrincipalPointIdxs();
      const_idxs.insert(const_idxs.end(), principal_point_idxs.begin(),
                        principal_point_idxs.end());
      SetSubsetManifold(imagecols_.cam(cam_id).params.size(), const_idxs,
                        problem_.get(), params_data);
    }

    if (config_.constant_pose) {
      if (!problem_->HasParameterBlock(qvec_data))
        continue;
      problem_->SetParameterBlockConstant(qvec_data);
      if (!problem_->HasParameterBlock(tvec_data))
        continue;
      problem_->SetParameterBlockConstant(tvec_data);
    } else {
      if (!problem_->HasParameterBlock(qvec_data))
        continue;
      SetQuaternionManifold(problem_.get(), qvec_data);
    }
  }
}

void HybridBAEngine::ParameterizePoints() {
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    double *point_data = it->second.data();
    if (!problem_->HasParameterBlock(point_data))
      continue;
    if (config_.constant_point)
      problem_->SetParameterBlockConstant(point_data);
  }
}

void HybridBAEngine::ParameterizeLines() {
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    int track_id = it->first;
    size_t n_images = it->second.count_images();
    double *uvec_data = lines_[track_id].uvec.data();
    double *wvec_data = lines_[track_id].wvec.data();
    if (!problem_->HasParameterBlock(uvec_data) ||
        !problem_->HasParameterBlock(wvec_data))
      continue;
    if (config_.constant_line || n_images < config_.min_num_images) {
      problem_->SetParameterBlockConstant(uvec_data);
      problem_->SetParameterBlockConstant(wvec_data);
    } else {
      SetQuaternionManifold(problem_.get(), uvec_data);
      SetSphereManifold<2>(problem_.get(), wvec_data);
    }
  }
}

void HybridBAEngine::AddPointGeometricResiduals(const int track_id) {
  if (config_.lw_point <= 0)
    return;
  const PointTrack &track = point_tracks_.at(track_id);
  ceres::LossFunction *loss_function =
      config_.point_geometric_loss_function.get();
  for (size_t i = 0; i < track.count_images(); ++i) {
    int img_id = track.image_id_list[i];
    auto model_id = imagecols_.camview(img_id).cam.model_id;
    V2D p2d = track.p2d_list[i];

    ceres::CostFunction *cost_function = nullptr;
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function = PointGeometricRefinementFunctor<CameraModel>::Create(      \
        p2d, NULL, NULL, NULL);                                                \
    break;
      LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }

    ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
        loss_function, config_.lw_point, ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
        cost_function, scaled_loss_function, points_.at(track_id).data(),
        imagecols_.params_data(img_id), imagecols_.qvec_data(img_id),
        imagecols_.tvec_data(img_id));
  }
}

void HybridBAEngine::AddLineGeometricResiduals(const int track_id) {
  const LineTrack &track = line_tracks_.at(track_id);
  ceres::LossFunction *loss_function =
      config_.line_geometric_loss_function.get();

  // compute line weights
  auto idmap = track.GetIdMap();
  std::vector<double> weights;
  ComputeLineWeights(track, weights);

  // add to problem for each supporting image (for each supporting line)
  auto &minimal_line = lines_.at(track_id);
  std::vector<int> image_ids = track.GetSortedImageIds();
  for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
    int img_id = *it1;
    auto model_id = imagecols_.camview(img_id).cam.model_id;
    const auto &ids = idmap.at(img_id);
    for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
      const Line2d &line = track.line2d_list[*it2];
      double weight = weights[*it2];
      ceres::CostFunction *cost_function = nullptr;

      switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function =                                                            \
        line_refinement::GeometricRefinementFunctor<CameraModel>::Create(      \
            line, NULL, NULL, NULL, config_.geometric_alpha);                  \
    break;
        LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
      }

      ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, minimal_line.uvec.data(),
          minimal_line.wvec.data(), imagecols_.params_data(img_id),
          imagecols_.qvec_data(img_id), imagecols_.tvec_data(img_id));
    }
  }
}

void HybridBAEngine::SetUp() {
  // setup problem
  problem_.reset(new ceres::Problem(config_.problem_options));

  // add residuals
  // R1.1: point geometric residual
  if (!config_.constant_point || !config_.constant_intrinsics ||
      !config_.constant_pose) {
    for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
      int point3d_id = it->first;
      AddPointGeometricResiduals(point3d_id);
    }
  }
  // R1.2: line geometric residual
  if (!config_.constant_line || !config_.constant_intrinsics ||
      !config_.constant_pose) {
    for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
      int line3d_id = it->first;
      AddLineGeometricResiduals(line3d_id);
    }
  }

  // parameterization
  ParameterizeCameras();
  ParameterizePoints();
  ParameterizeLines();
}

bool HybridBAEngine::Solve() {
  if (problem_->NumResiduals() == 0)
    return false;
  ceres::Solver::Options solver_options = config_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 900;
  const size_t num_images = imagecols_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else { // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (config_.print_summary) {
    colmap::PrintSolverSummary(summary_, "Optimization report");
  }
  return true;
}

std::map<int, V3D> HybridBAEngine::GetOutputPoints() const {
  std::map<int, V3D> outputs;
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
    int track_id = it->first;
    outputs.insert(std::make_pair(track_id, points_.at(track_id)));
  }
  return outputs;
}

std::map<int, PointTrack> HybridBAEngine::GetOutputPointTracks() const {
  std::map<int, PointTrack> outputs;
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
    int track_id = it->first;
    PointTrack track = it->second;
    track.p = points_.at(track_id);
    outputs.insert(std::make_pair(track_id, track));
  }
  return outputs;
}

std::map<int, Line3d>
HybridBAEngine::GetOutputLines(const int num_outliers) const {
  std::map<int, Line3d> outputs;
  std::map<int, LineTrack> output_line_tracks =
      GetOutputLineTracks(num_outliers);
  for (auto it = output_line_tracks.begin(); it != output_line_tracks.end();
       ++it) {
    outputs.insert(std::make_pair(it->first, it->second.line));
  }
  return outputs;
}

std::map<int, LineTrack>
HybridBAEngine::GetOutputLineTracks(const int num_outliers) const {
  std::map<int, LineTrack> outputs;
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    int track_id = it->first;
    LineTrack track = it->second;
    Line3d line = GetLineSegmentFromInfiniteLine3d(
        lines_.at(track_id).GetInfiniteLine(), track.line3d_list, num_outliers);
    track.line = line;
    outputs.insert(std::make_pair(track_id, track));
  }
  return outputs;
}

} // namespace hybrid_bundle_adjustment

} // namespace optimize

} // namespace limap
