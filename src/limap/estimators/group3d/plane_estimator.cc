#include "limap/estimators/group3d/plane_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Minimum eigenvalue ratio (λ_mid / λ_max) of the inlier scatter matrix.
// Below this threshold, inliers are near-collinear and the plane is degenerate.
// Corresponds roughly to the squared aspect ratio: a 100:1 elongated plane
// gives ~1e-4, while truly collinear points give ~0.
constexpr double kPlaneDegeneracyEigenvalueRatio = 1e-4;

// Plane Estimator for RANSAC (internal implementation)
class PlaneEstimator {
public:
  PlaneEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                 const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<V4D> *models);
  double score_model(const V4D &plane, size_t *inlier_count) const;
  void refine_model(V4D *plane) const;

  const size_t sample_sz = 3;
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;
};

// ============================================================================
// PlaneEstimator implementation
// ============================================================================

void PlaneEstimator::generate_models(std::vector<V4D> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  // Minimal solver: fit plane from 3 points
  V3D p1 = points_.col(sample[0]);
  V3D p2 = points_.col(sample[1]);
  V3D p3 = points_.col(sample[2]);

  // Compute plane normal via cross product
  V3D v1 = p2 - p1;
  V3D v2 = p3 - p1;
  V3D normal = v1.cross(v2);

  if (normal.norm() >= 1e-12) {
    normal.normalize();
    double d = -normal.dot(p1);
    V4D plane;
    plane << normal, d;
    models->push_back(plane);
  }
}

double PlaneEstimator::score_model(const V4D &plane,
                                   size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  V3D normal = plane.head<3>();
  double d = plane(3);

  for (size_t i = 0; i < num_data; ++i) {
    // Point-to-plane squared distance
    V3D point = points_.col(i);
    double dist = std::abs(normal.dot(point) + d);
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

void PlaneEstimator::refine_model(V4D *plane) const {
  const double sq_threshold = max_error_ * max_error_;

  V3D normal = plane->head<3>();
  double d = (*plane)(3);

  // Collect inliers
  std::vector<size_t> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = std::abs(normal.dot(point) + d);
    if (dist * dist < sq_threshold) {
      inliers.push_back(i);
    }
  }

  // Refine with SVD on inliers (need at least 3 points)
  if (inliers.size() < 3)
    return;

  // Compute centroid
  V3D centroid = V3D::Zero();
  for (size_t idx : inliers) {
    centroid += points_.col(idx);
  }
  centroid /= static_cast<double>(inliers.size());

  // Build covariance matrix
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (size_t idx : inliers) {
    V3D d_vec = points_.col(idx) - centroid;
    C += d_vec * d_vec.transpose();
  }
  C /= static_cast<double>(inliers.size() - 1);

  // Find normal (smallest eigenvector)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() == Eigen::Success) {
    V3D refined_normal = eig_solver.eigenvectors().col(0).normalized();
    double refined_d = -refined_normal.dot(centroid);
    (*plane) << refined_normal, refined_d;
  }
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<V4D> EstimatePlaneRobust(const std::vector<V3D> &points,
                                       const std::vector<Line3d> &lines,
                                       double max_error,
                                       const poselib::RansacOptions &options,
                                       poselib::RansacStats *stats) {
  // Combine points and line endpoints
  size_t total_points = points.size() + lines.size() * 2;
  if (total_points < 3) {
    return std::nullopt;
  }

  Eigen::Matrix3Xd points_mat(3, total_points);
  size_t idx = 0;

  // Add points
  for (const auto &p : points) {
    points_mat.col(idx++) = p;
  }

  // Add line endpoints
  for (const auto &line : lines) {
    points_mat.col(idx++) = line.start;
    points_mat.col(idx++) = line.end;
  }

  PlaneEstimator solver(max_error, options, points_mat);

  // Check if we have enough points for minimal solver
  if (solver.num_data < solver.sample_sz) {
    return std::nullopt;
  }
  V4D best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<PlaneEstimator, V4D>(solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid and ensure normalization
  V3D normal = best_model.head<3>();
  double normal_norm = normal.norm();
  if (std::isnan(normal_norm) || normal_norm < 1e-12) {
    return std::nullopt;
  }

  // Degeneracy check: verify inliers are not near-collinear.
  // Compute scatter matrix of final inliers and check eigenvalue ratio
  // λ_mid / λ_max. For collinear inliers this ratio is ~0 (noise/signal),
  // while even a 100:1 elongated plane gives ~1e-4 (squared aspect ratio).
  {
    const double sq_threshold = max_error * max_error;
    V3D n = normal / normal_norm;
    double d = best_model(3) / normal_norm;

    V3D centroid = V3D::Zero();
    std::vector<size_t> inliers;
    for (size_t i = 0; i < points_mat.cols(); ++i) {
      double dist = std::abs(n.dot(points_mat.col(i)) + d);
      if (dist * dist < sq_threshold) {
        inliers.push_back(i);
        centroid += points_mat.col(i);
      }
    }
    if (inliers.size() >= 3) {
      centroid /= static_cast<double>(inliers.size());
      Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
      for (size_t idx : inliers) {
        V3D diff = points_mat.col(idx) - centroid;
        C += diff * diff.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
      if (eig.info() == Eigen::Success) {
        double lambda_mid = eig.eigenvalues()(1);
        double lambda_max = eig.eigenvalues()(2);
        if (lambda_max > 0 &&
            lambda_mid / lambda_max < kPlaneDegeneracyEigenvalueRatio) {
          LOG(WARNING) << "EstimatePlaneRobust: degenerate (collinear) inliers "
                       << "(eigenvalue ratio=" << lambda_mid / lambda_max
                       << "), skipping plane";
          return std::nullopt;
        }
      }
    }
  }

  // Normalize: ||(a,b,c)|| = 1, scale d accordingly
  V4D normalized_plane;
  normalized_plane << normal / normal_norm, best_model(3) / normal_norm;
  return normalized_plane;
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<V4D> EstimatePlaneFromPoints(const std::vector<V3D> &points) {
  if (points.size() < 3) {
    return std::nullopt;
  }

  const int n_points = static_cast<int>(points.size());

  // Compute centroid
  V3D centroid = V3D::Zero();
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(n_points);

  // Compute covariance matrix
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto &p : points) {
    V3D d = p - centroid;
    C += d * d.transpose();
  }
  C /= static_cast<double>(n_points - 1);

  // Find the eigenvector corresponding to the smallest eigenvalue
  // This is the normal to the best-fit plane
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() != Eigen::Success) {
    LOG(WARNING) << "EstimatePlaneFromPoints: Eigen decomposition failed";
    return std::nullopt;
  }

  // Degeneracy check: eigenvalues are sorted in increasing order.
  // If λ_mid / λ_max is too small, points are near-collinear.
  double lambda_mid = eig_solver.eigenvalues()(1);
  double lambda_max = eig_solver.eigenvalues()(2);
  if (lambda_max > 0 &&
      lambda_mid / lambda_max < kPlaneDegeneracyEigenvalueRatio) {
    LOG(WARNING) << "EstimatePlaneFromPoints: degenerate (collinear) points "
                 << "(eigenvalue ratio=" << lambda_mid / lambda_max
                 << "), skipping plane";
    return std::nullopt;
  }

  V3D normal = eig_solver.eigenvectors().col(0);
  normal.normalize();

  // Compute d: plane equation is n'*x + d = 0
  // We have n'*centroid + d = 0, so d = -n'*centroid
  double d = -normal.dot(centroid);

  V4D plane_params;
  plane_params << normal, d;

  return plane_params;
}

std::optional<V4D> EstimatePlaneFromLines(const std::vector<Line3d> &lines) {
  // Need at least 2 lines (4 endpoints) to fit a plane
  if (lines.size() < 2) {
    return std::nullopt;
  }

  // Collect all endpoints
  std::vector<V3D> points;
  points.reserve(lines.size() * 2);
  for (const auto &line : lines) {
    points.push_back(line.start);
    points.push_back(line.end);
  }

  return EstimatePlaneFromPoints(points);
}

std::optional<V4D> EstimatePlane(const std::vector<V3D> &points,
                                 const std::vector<Line3d> &lines) {
  // Need at least 3 points total (each line contributes 2 endpoints)
  if (points.size() + lines.size() * 2 < 3) {
    return std::nullopt;
  }

  // Concatenate points and line endpoints
  std::vector<V3D> all_points;
  all_points.reserve(points.size() + lines.size() * 2);

  // Copy input points
  all_points.insert(all_points.end(), points.begin(), points.end());

  // Add line endpoints
  for (const auto &line : lines) {
    all_points.push_back(line.start);
    all_points.push_back(line.end);
  }

  return EstimatePlaneFromPoints(all_points);
}

} // namespace group3d

} // namespace estimators

} // namespace limap
