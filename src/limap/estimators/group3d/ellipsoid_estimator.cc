#include "limap/estimators/group3d/ellipsoid_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Ellipsoid Estimator for RANSAC (internal implementation)
// Uses 9 points for minimal solver (9 DOF ellipsoid)
class EllipsoidEstimator {
public:
  // Model type: (center, rotation matrix, scales)
  struct EllipsoidModel {
    V3D center;
    M3D rotation; // Columns are principal axes
    V3D scales;   // Semi-axis lengths

    // Distance from point to ellipsoid surface using scaled-radial projection.
    // Projects point onto surface and returns Euclidean distance.
    double DistanceToSurface(const V3D &point) const {
      V3D local = rotation.transpose() * (point - center);
      double a = scales.x(), b = scales.y(), c = scales.z();

      // Scale to unit sphere
      V3D scaled(local.x() / a, local.y() / b, local.z() / c);
      double r = scaled.norm();

      if (r < 1e-12) {
        return scales.minCoeff(); // Point at center
      }

      // Normalize and scale back to get point on ellipsoid surface
      V3D surface_local =
          V3D(scaled.x() / r * a, scaled.y() / r * b, scaled.z() / r * c);

      return (local - surface_local).norm();
    }
  };

  EllipsoidEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                     const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<EllipsoidModel> *models);
  double score_model(const EllipsoidModel &ellipsoid,
                     size_t *inlier_count) const;
  void refine_model(EllipsoidModel *ellipsoid) const;

  const size_t sample_sz = 9; // 9 points for ellipsoid (9 DOF)
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;

  // Helper: fit ellipsoid to points using algebraic method
  static bool FitEllipsoidFromPoints(const std::vector<V3D> &points,
                                     EllipsoidModel *ellipsoid);
};

// ============================================================================
// EllipsoidEstimator implementation
// ============================================================================

bool EllipsoidEstimator::FitEllipsoidFromPoints(const std::vector<V3D> &points,
                                                EllipsoidModel *ellipsoid) {
  const size_t n = points.size();
  if (n < 9) {
    return false;
  }

  // Algebraic ellipsoid fitting
  // Ellipsoid equation: ax^2 + by^2 + cz^2 + 2dxy + 2exz + 2fyz + 2gx + 2hy +
  // 2iz + j = 0 We use the constraint that a + b + c = 3 (ensures ellipsoid,
  // not other quadrics) This gives us 9 unknowns with the constraint.

  // Build design matrix D where each row is:
  // [x^2, y^2, z^2, 2xy, 2xz, 2yz, 2x, 2y, 2z, 1]
  Eigen::MatrixXd D(n, 10);
  for (size_t i = 0; i < n; ++i) {
    double x = points[i].x();
    double y = points[i].y();
    double z = points[i].z();
    D(i, 0) = x * x;
    D(i, 1) = y * y;
    D(i, 2) = z * z;
    D(i, 3) = 2.0 * x * y;
    D(i, 4) = 2.0 * x * z;
    D(i, 5) = 2.0 * y * z;
    D(i, 6) = 2.0 * x;
    D(i, 7) = 2.0 * y;
    D(i, 8) = 2.0 * z;
    D(i, 9) = 1.0;
  }

  // SVD to find null space
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullV);
  Eigen::VectorXd v = svd.matrixV().col(9); // Last column (smallest singular
                                            // value)

  // Check if solution is valid (a + b + c should have same sign as -j for
  // ellipsoid)
  double a = v(0), b = v(1), c = v(2);
  double d = v(3), e = v(4), f = v(5);
  double g = v(6), h = v(7), i = v(8);
  double j = v(9);

  // Build 3x3 matrix A (quadratic part) and 3x1 vector B (linear part)
  M3D A;
  A << a, d, e, d, b, f, e, f, c;

  V3D B(g, h, i);

  // Check if A is positive definite (ellipsoid condition)
  Eigen::SelfAdjointEigenSolver<M3D> solver(A);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  V3D eigenvalues = solver.eigenvalues();
  // For an ellipsoid, all eigenvalues should have the same sign
  if (eigenvalues.minCoeff() * eigenvalues.maxCoeff() <= 0) {
    // Not an ellipsoid (could be hyperboloid)
    return false;
  }

  // Compute center: c = -A^{-1} * B
  V3D center = -A.inverse() * B;

  // Compute the constant term after shifting to center
  // j' = j - B^T * A^{-1} * B = j + B^T * center
  double j_prime = j + B.dot(center);

  if (std::abs(j_prime) < 1e-12) {
    return false; // Degenerate
  }

  // Normalize so that the quadric equals 1 at surface
  // A_normalized = A / (-j')
  M3D A_norm = A / (-j_prime);

  // Eigendecomposition of A_norm gives rotation and scales
  Eigen::SelfAdjointEigenSolver<M3D> solver2(A_norm);
  if (solver2.info() != Eigen::Success) {
    return false;
  }

  V3D eigs = solver2.eigenvalues();
  M3D vecs = solver2.eigenvectors();

  // Eigenvalues should all be positive for ellipsoid
  if (eigs.minCoeff() <= 0) {
    return false;
  }

  // Scales are 1/sqrt(eigenvalue)
  V3D scales(1.0 / std::sqrt(eigs(0)), 1.0 / std::sqrt(eigs(1)),
             1.0 / std::sqrt(eigs(2)));

  ellipsoid->center = center;
  ellipsoid->rotation = vecs;
  ellipsoid->scales = scales;

  return true;
}

void EllipsoidEstimator::generate_models(std::vector<EllipsoidModel> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  std::vector<V3D> sample_points;
  sample_points.reserve(sample_sz);
  for (size_t idx : sample) {
    sample_points.push_back(points_.col(idx));
  }

  EllipsoidModel ellipsoid;
  if (FitEllipsoidFromPoints(sample_points, &ellipsoid)) {
    models->push_back(ellipsoid);
  }
}

double EllipsoidEstimator::score_model(const EllipsoidModel &ellipsoid,
                                       size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = ellipsoid.DistanceToSurface(point);
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

void EllipsoidEstimator::refine_model(EllipsoidModel *ellipsoid) const {
  const double sq_threshold = max_error_ * max_error_;

  // Collect inliers
  std::vector<V3D> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = ellipsoid->DistanceToSurface(point);
    if (dist * dist < sq_threshold) {
      inliers.push_back(point);
    }
  }

  if (inliers.size() < 9)
    return;

  // Refit with all inliers using algebraic method
  EllipsoidModel refined;
  if (FitEllipsoidFromPoints(inliers, &refined)) {
    *ellipsoid = refined;
  }
}

// Convert EllipsoidModel to minimal 10D representation [quat(4), center(3),
// log_scales(3)]
std::vector<double>
EllipsoidToMinimalParams(const EllipsoidEstimator::EllipsoidModel &model) {
  Eigen::Quaterniond quat(model.rotation);
  quat.normalize();

  std::vector<double> params(10);
  params[0] = quat.x();
  params[1] = quat.y();
  params[2] = quat.z();
  params[3] = quat.w();
  params[4] = model.center.x();
  params[5] = model.center.y();
  params[6] = model.center.z();
  params[7] = std::log(model.scales.x());
  params[8] = std::log(model.scales.y());
  params[9] = std::log(model.scales.z());
  return params;
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<std::vector<double>>
EstimateEllipsoidRobust(const std::vector<V3D> &points, double max_error,
                        const poselib::RansacOptions &options,
                        poselib::RansacStats *stats) {
  if (points.size() < 9) {
    LOG(WARNING) << "EstimateEllipsoidRobust: Need at least 9 points, got "
                 << points.size();
    return std::nullopt;
  }

  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  EllipsoidEstimator solver(max_error, options, points_mat);

  EllipsoidEstimator::EllipsoidModel best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<EllipsoidEstimator, EllipsoidEstimator::EllipsoidModel>(
          solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid
  if (!std::isfinite(best_model.center.x()) ||
      !std::isfinite(best_model.scales.x()) ||
      best_model.scales.minCoeff() <= 0) {
    return std::nullopt;
  }

  return EllipsoidToMinimalParams(best_model);
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<std::vector<double>>
EstimateEllipsoid(const std::vector<V3D> &points) {
  if (points.size() < 9) {
    LOG(WARNING) << "EstimateEllipsoid: Need at least 9 points, got "
                 << points.size();
    return std::nullopt;
  }

  // Use the same algebraic fitting as in the estimator
  const size_t n = points.size();

  // Build design matrix D where each row is:
  // [x^2, y^2, z^2, 2xy, 2xz, 2yz, 2x, 2y, 2z, 1]
  Eigen::MatrixXd D(n, 10);
  for (size_t i = 0; i < n; ++i) {
    double x = points[i].x();
    double y = points[i].y();
    double z = points[i].z();
    D(i, 0) = x * x;
    D(i, 1) = y * y;
    D(i, 2) = z * z;
    D(i, 3) = 2.0 * x * y;
    D(i, 4) = 2.0 * x * z;
    D(i, 5) = 2.0 * y * z;
    D(i, 6) = 2.0 * x;
    D(i, 7) = 2.0 * y;
    D(i, 8) = 2.0 * z;
    D(i, 9) = 1.0;
  }

  // SVD to find null space
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullV);
  Eigen::VectorXd v = svd.matrixV().col(9);

  double a = v(0), b = v(1), c = v(2);
  double d = v(3), e = v(4), f = v(5);
  double g = v(6), h = v(7), i = v(8);
  double j = v(9);

  M3D A;
  A << a, d, e, d, b, f, e, f, c;

  V3D B(g, h, i);

  Eigen::SelfAdjointEigenSolver<M3D> solver(A);
  if (solver.info() != Eigen::Success) {
    LOG(WARNING) << "EstimateEllipsoid: Eigendecomposition failed";
    return std::nullopt;
  }

  V3D eigenvalues = solver.eigenvalues();
  if (eigenvalues.minCoeff() * eigenvalues.maxCoeff() <= 0) {
    LOG(WARNING) << "EstimateEllipsoid: Not an ellipsoid";
    return std::nullopt;
  }

  V3D center = -A.inverse() * B;
  double j_prime = j + B.dot(center);

  if (std::abs(j_prime) < 1e-12) {
    LOG(WARNING) << "EstimateEllipsoid: Degenerate ellipsoid";
    return std::nullopt;
  }

  M3D A_norm = A / (-j_prime);

  Eigen::SelfAdjointEigenSolver<M3D> solver2(A_norm);
  if (solver2.info() != Eigen::Success) {
    LOG(WARNING) << "EstimateEllipsoid: Second eigendecomposition failed";
    return std::nullopt;
  }

  V3D eigs = solver2.eigenvalues();
  M3D vecs = solver2.eigenvectors();

  if (eigs.minCoeff() <= 0) {
    LOG(WARNING) << "EstimateEllipsoid: Negative eigenvalue";
    return std::nullopt;
  }

  EllipsoidEstimator::EllipsoidModel model;
  model.center = center;
  model.rotation = vecs;
  model.scales = V3D(1.0 / std::sqrt(eigs(0)), 1.0 / std::sqrt(eigs(1)),
                     1.0 / std::sqrt(eigs(2)));
  return EllipsoidToMinimalParams(model);
}

} // namespace group3d

} // namespace estimators

} // namespace limap
