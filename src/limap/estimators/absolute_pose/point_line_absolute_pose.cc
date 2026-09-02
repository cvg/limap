#include "limap/estimators/absolute_pose/point_line_absolute_pose.h"

#include <colmap/estimators/solvers/poselib_utils.h>

#include <PoseLib/robust.h>
#include <PoseLib/robust/hybrid_ransac.h>
#include <PoseLib/types.h>

#include <algorithm>
#include <random>

namespace limap {
namespace estimators {
namespace absolute_pose {

PointLineAbsolutePoseResult EstimatePointLineAbsolutePose(
    const std::vector<Line3d> &l3ds, const std::vector<Line2d> &l2ds,
    const std::vector<V3D> &p3ds, const std::vector<V2D> &p2ds,
    const colmap::Camera &cam, const PointLineAbsolutePoseOptions &options) {
  PointLineAbsolutePoseResult result;

  // Compute K and K_inv for coordinate conversion
  Eigen::Matrix3d K = cam.CalibrationMatrix();
  Eigen::Matrix3d K_inv = K.inverse();

  // Convert 2D points from pixels to normalized coordinates
  std::vector<poselib::Point2D> pts2d;
  pts2d.reserve(p2ds.size());
  for (const auto &p : p2ds) {
    V3D normalized = K_inv * p.homogeneous();
    pts2d.push_back(normalized.hnormalized());
  }
  std::vector<poselib::Point3D> pts3d(p3ds.begin(), p3ds.end());

  // Convert 2D lines from pixels to normalized coordinates
  std::vector<poselib::Line2D> lns2d;
  std::vector<poselib::Line3D> lns3d;
  lns2d.reserve(l2ds.size());
  lns3d.reserve(l3ds.size());
  for (size_t i = 0; i < l2ds.size(); ++i) {
    V3D start_norm = K_inv * l2ds[i].start.homogeneous();
    V3D end_norm = K_inv * l2ds[i].end.homogeneous();
    lns2d.push_back(
        poselib::Line2D(start_norm.hnormalized(), end_norm.hnormalized()));
    lns3d.push_back(poselib::Line3D(l3ds[i].start, l3ds[i].end));
  }

  if (options.estimate_focal_length) {
    // PnPLf takes pixel coordinates and pixel thresholds: it unprojects
    // through the camera itself and rescales the thresholds internally, so
    // nothing may be normalized here.
    std::vector<poselib::Point2D> pts2d_px(p2ds.begin(), p2ds.end());
    std::vector<poselib::Line2D> lns2d_px;
    lns2d_px.reserve(l2ds.size());
    for (const auto &l2d : l2ds)
      lns2d_px.push_back(poselib::Line2D(l2d.start, l2d.end));

    poselib::AbsolutePoseOptions opt_f;
    opt_f.max_errors = {options.max_error_point, options.max_error_line};
    opt_f.ransac.max_iterations = options.max_iterations;
    opt_f.ransac.min_iterations = options.min_iterations;
    opt_f.ransac.success_prob = options.success_prob;
    if (options.random_seed) {
      std::random_device rd;
      opt_f.ransac.seed = rd();
    } else {
      opt_f.ransac.seed = options.seed;
    }

    poselib::Image image;
    poselib::RansacStats stats = poselib::estimate_absolute_pose_pnplf(
        pts2d_px, pts3d, lns2d_px, lns3d,
        colmap::ConvertCameraToPoseLibCamera(cam), opt_f, &image,
        &result.inliers_points, &result.inliers_lines);

    result.pose = colmap::ConvertPoseLibPoseToRigid3d(image.pose);
    result.camera = colmap::ConvertPoseLibCameraToCamera(image.camera);
    result.iterations = stats.iterations;
    result.num_inliers = stats.num_inliers;
    result.model_score = stats.model_score;
    // RansacStats has no per-type breakdown, so count from the masks.
    result.num_inliers_points =
        std::count(result.inliers_points.begin(), result.inliers_points.end(),
                   static_cast<char>(1));
    result.num_inliers_lines =
        std::count(result.inliers_lines.begin(), result.inliers_lines.end(),
                   static_cast<char>(1));
    result.success = (stats.num_inliers > 0);
    return result;
  }

  // Convert thresholds to normalized units
  double focal = (K(0, 0) + K(1, 1)) / 2.0;
  poselib::HybridRansacOptions opt;
  opt.max_errors = {options.max_error_point / focal,
                    options.max_error_line / focal};
  opt.data_type_weights = {options.weight_point, options.weight_line};
  opt.max_iterations = options.max_iterations;
  opt.min_iterations = options.min_iterations;
  opt.success_prob = options.success_prob;
  if (options.random_seed) {
    std::random_device rd;
    opt.seed = rd();
  } else {
    opt.seed = options.seed;
  }

  poselib::CameraPose poselib_pose;
  poselib_pose.q << 1.0, 0.0, 0.0, 0.0;
  poselib_pose.t.setZero();

  poselib::HybridRansacStats stats = poselib::hybrid_ransac_pnpl(
      pts2d, pts3d, lns2d, lns3d, opt, &poselib_pose, &result.inliers_points,
      &result.inliers_lines);

  result.pose = colmap::ConvertPoseLibPoseToRigid3d(poselib_pose);

  // Copy stats to result
  result.iterations = stats.iterations;
  result.num_inliers = stats.num_inliers;
  result.model_score = stats.model_score;
  if (stats.num_inliers_per_type.size() >= 2) {
    result.num_inliers_points = stats.num_inliers_per_type[0];
    result.num_inliers_lines = stats.num_inliers_per_type[1];
  }
  result.success = (stats.num_inliers > 0);

  return result;
}

} // namespace absolute_pose
} // namespace estimators
} // namespace limap
