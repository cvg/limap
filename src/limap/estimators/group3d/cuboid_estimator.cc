#include "limap/estimators/group3d/cuboid_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

#include <cmath>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Internal cuboid model for RANSAC
struct CuboidModel {
  M3D rotation; // Columns are cuboid axes
  double d[6];  // Face offsets: +x, -x, +y, -y, +z, -z

  // Distance from point to nearest cuboid face
  double DistanceToSurface(const V3D &point) const {
    double dot0 = rotation.col(0).dot(point);
    double dot1 = rotation.col(1).dot(point);
    double dot2 = rotation.col(2).dot(point);

    double dist[6];
    dist[0] = std::abs(dot0 - d[0]);
    dist[1] = std::abs(-dot0 - d[1]);
    dist[2] = std::abs(dot1 - d[2]);
    dist[3] = std::abs(-dot1 - d[3]);
    dist[4] = std::abs(dot2 - d[4]);
    dist[5] = std::abs(-dot2 - d[5]);

    double min_dist = dist[0];
    for (int i = 1; i < 6; ++i) {
      min_dist = std::min(min_dist, dist[i]);
    }
    return min_dist;
  }
};

// Fit cuboid to points using PCA for R, then min/max projections for d.
bool FitCuboidFromPoints(const std::vector<V3D> &points, CuboidModel *cuboid) {
  const size_t n = points.size();
  if (n < 6) {
    return false;
  }

  // Compute centroid
  V3D centroid = V3D::Zero();
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(n);

  // Compute covariance matrix
  M3D cov = M3D::Zero();
  for (const auto &p : points) {
    V3D diff = p - centroid;
    cov += diff * diff.transpose();
  }
  cov /= static_cast<double>(n);

  // PCA to get principal directions
  Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  // Eigenvectors sorted by eigenvalue (ascending); we want descending
  M3D R;
  R.col(0) = solver.eigenvectors().col(2);
  R.col(1) = solver.eigenvectors().col(1);
  R.col(2) = solver.eigenvectors().col(0);

  // Ensure right-handed frame
  if (R.determinant() < 0) {
    R.col(2) = -R.col(2);
  }

  // Compute face offsets from projections
  double dot_max[3] = {-std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()};
  double dot_min[3] = {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity()};

  for (const auto &p : points) {
    for (int k = 0; k < 3; ++k) {
      double dk = R.col(k).dot(p);
      dot_max[k] = std::max(dot_max[k], dk);
      dot_min[k] = std::min(dot_min[k], dk);
    }
  }

  cuboid->rotation = R;
  cuboid->d[0] = dot_max[0];  // +x
  cuboid->d[1] = -dot_min[0]; // -x
  cuboid->d[2] = dot_max[1];  // +y
  cuboid->d[3] = -dot_min[1]; // -y
  cuboid->d[4] = dot_max[2];  // +z
  cuboid->d[5] = -dot_min[2]; // -z

  return true;
}

// Convert CuboidModel to 10D parameter vector [quat(4), d(6)]
std::vector<double> CuboidToParams(const CuboidModel &model) {
  Eigen::Quaterniond quat(model.rotation);
  quat.normalize();

  std::vector<double> params(10);
  params[0] = quat.x();
  params[1] = quat.y();
  params[2] = quat.z();
  params[3] = quat.w();
  for (int i = 0; i < 6; ++i) {
    params[4 + i] = model.d[i];
  }
  return params;
}

// Cuboid Estimator for RANSAC
class CuboidEstimator {
public:
  CuboidEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                  const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<CuboidModel> *models);
  double score_model(const CuboidModel &cuboid, size_t *inlier_count) const;
  void refine_model(CuboidModel *cuboid) const;

  const size_t sample_sz = 6; // Minimum for PCA + 6 face offsets
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;
};

void CuboidEstimator::generate_models(std::vector<CuboidModel> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  std::vector<V3D> sample_points;
  sample_points.reserve(sample_sz);
  for (size_t idx : sample) {
    sample_points.push_back(points_.col(idx));
  }

  CuboidModel cuboid;
  if (FitCuboidFromPoints(sample_points, &cuboid)) {
    models->push_back(cuboid);
  }
}

double CuboidEstimator::score_model(const CuboidModel &cuboid,
                                    size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = cuboid.DistanceToSurface(point);
    double error = dist * dist;

    if (error < sq_threshold) {
      (*inlier_count)++;
      score += error;
    } else {
      score += sq_threshold;
    }
  }

  return score;
}

void CuboidEstimator::refine_model(CuboidModel *cuboid) const {
  const double sq_threshold = max_error_ * max_error_;

  // Collect inliers
  std::vector<V3D> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = cuboid->DistanceToSurface(point);
    if (dist * dist < sq_threshold) {
      inliers.push_back(point);
    }
  }

  if (inliers.size() < 6)
    return;

  // Refit with all inliers
  CuboidModel refined;
  if (FitCuboidFromPoints(inliers, &refined)) {
    *cuboid = refined;
  }
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<std::vector<double>>
EstimateCuboidRobust(const std::vector<V3D> &points, double max_error,
                     const poselib::RansacOptions &options,
                     poselib::RansacStats *stats) {
  if (points.size() < 6) {
    LOG(WARNING) << "EstimateCuboidRobust: Need at least 6 points, got "
                 << points.size();
    return std::nullopt;
  }

  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  CuboidEstimator solver(max_error, options, points_mat);

  CuboidModel best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<CuboidEstimator, CuboidModel>(solver, options,
                                                    &best_model);

  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid
  if (!std::isfinite(best_model.rotation(0, 0))) {
    return std::nullopt;
  }
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(best_model.d[i])) {
      return std::nullopt;
    }
  }

  return CuboidToParams(best_model);
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<std::vector<double>>
EstimateCuboid(const std::vector<V3D> &points) {
  if (points.size() < 6) {
    LOG(WARNING) << "EstimateCuboid: Need at least 6 points, got "
                 << points.size();
    return std::nullopt;
  }

  CuboidModel model;
  if (!FitCuboidFromPoints(points, &model)) {
    LOG(WARNING) << "EstimateCuboid: PCA-based cuboid fitting failed";
    return std::nullopt;
  }

  return CuboidToParams(model);
}

} // namespace group3d

} // namespace estimators

} // namespace limap
