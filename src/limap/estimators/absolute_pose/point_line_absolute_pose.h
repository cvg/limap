#pragma once

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/camera.h>

#include <limits>
#include <optional>
#include <vector>

namespace limap {
namespace estimators {
namespace absolute_pose {

// Options for point-line absolute pose estimation
struct PointLineAbsolutePoseOptions {
  // RANSAC thresholds (pixels)
  double max_error_point = 12.0;
  double max_error_line = 12.0;

  // Weights for hybrid scoring
  double weight_point = 1.0;
  double weight_line = 1.0;

  // RANSAC iterations
  int max_iterations = 10000;
  int min_iterations = 100;
  double success_prob = 0.9999;

  // Random seed: if random_seed is true, use a random seed from random_device
  // Otherwise use the fixed seed value
  bool random_seed = true;
  unsigned int seed = 0;

  // Estimate a shared focal length along with the pose (PnPLf). The camera
  // then supplies the principal point and an initial guess for the focal
  // instead of a known calibration. The per-type weights above are unused
  // in this mode; PoseLib's PnPLf estimator does not take them.
  bool estimate_focal_length = false;
};

// Result structure with colmap types
struct PointLineAbsolutePoseResult {
  // Estimated pose as colmap::Rigid3d (cam_from_world)
  colmap::Rigid3d pose;

  // Camera carrying the estimated focal length. Set only when
  // options.estimate_focal_length was.
  std::optional<colmap::Camera> camera;

  // RANSAC statistics
  size_t num_inliers = 0;
  size_t num_inliers_points = 0;
  size_t num_inliers_lines = 0;
  size_t iterations = 0;
  double model_score = std::numeric_limits<double>::max();

  // Inlier masks
  std::vector<char> inliers_points;
  std::vector<char> inliers_lines;

  // Whether estimation succeeded
  bool success = false;
};

// Main estimation function using PoseLib's hybrid RANSAC.
// l3ds and l2ds must have the same size (direct correspondence).
// With options.estimate_focal_length the focal length is estimated too, and
// cam supplies the principal point and an initial guess for it.
PointLineAbsolutePoseResult EstimatePointLineAbsolutePose(
    const std::vector<Line3d> &l3ds, const std::vector<Line2d> &l2ds,
    const std::vector<V3D> &p3ds, const std::vector<V2D> &p2ds,
    const colmap::Camera &cam, const PointLineAbsolutePoseOptions &options);

} // namespace absolute_pose
} // namespace estimators
} // namespace limap
