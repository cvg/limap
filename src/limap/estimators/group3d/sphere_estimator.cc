#include "limap/estimators/group3d/sphere_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Sphere Estimator for RANSAC (internal implementation)
// Sphere params: [cx, cy, cz, log_r] where r = exp(log_r)
class SphereEstimator {
public:
  SphereEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                  const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<V4D> *models);
  double score_model(const V4D &sphere, size_t *inlier_count) const;
  void refine_model(V4D *sphere) const;

  const size_t sample_sz = 4; // 4 points define a sphere
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;

  // Helper: fit sphere to 4 points (exact fit)
  static bool FitSphereFrom4Points(const V3D &p1, const V3D &p2, const V3D &p3,
                                   const V3D &p4, V4D *sphere);

  // Helper: fit sphere to points using least squares
  static bool FitSphereLeastSquares(const std::vector<V3D> &points,
                                    V4D *sphere);
};

// ============================================================================
// SphereEstimator implementation
// ============================================================================

bool SphereEstimator::FitSphereFrom4Points(const V3D &p1, const V3D &p2,
                                           const V3D &p3, const V3D &p4,
                                           V4D *sphere) {
  // Fit sphere through 4 points using the determinant method
  // Sphere equation: (x-cx)² + (y-cy)² + (z-cz)² = r²
  // Expanding and rearranging: A*x + B*y + C*z + D = -(x² + y² + z²)
  // where A = -2*cx, B = -2*cy, C = -2*cz, D = cx² + cy² + cz² - r²

  Eigen::Matrix4d M;
  Eigen::Vector4d b;

  auto row = [](const V3D &p) -> Eigen::Vector4d {
    return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
  };

  M.row(0) = row(p1);
  M.row(1) = row(p2);
  M.row(2) = row(p3);
  M.row(3) = row(p4);

  b(0) = -(p1.squaredNorm());
  b(1) = -(p2.squaredNorm());
  b(2) = -(p3.squaredNorm());
  b(3) = -(p4.squaredNorm());

  // Check if matrix is singular (points are coplanar)
  double det = M.determinant();
  if (std::abs(det) < 1e-12) {
    return false;
  }

  Eigen::Vector4d x = M.colPivHouseholderQr().solve(b);

  // Extract center and radius
  double cx = -x(0) / 2.0;
  double cy = -x(1) / 2.0;
  double cz = -x(2) / 2.0;
  double r_sq = cx * cx + cy * cy + cz * cz - x(3);

  if (r_sq <= 1e-12) {
    return false;
  }

  double r = std::sqrt(r_sq);
  (*sphere) << cx, cy, cz, std::log(r);
  return true;
}

bool SphereEstimator::FitSphereLeastSquares(const std::vector<V3D> &points,
                                            V4D *sphere) {
  const size_t n = points.size();
  if (n < 4) {
    return false;
  }

  Eigen::MatrixXd M(n, 4);
  Eigen::VectorXd b(n);

  for (size_t i = 0; i < n; ++i) {
    const V3D &p = points[i];
    M(i, 0) = p.x();
    M(i, 1) = p.y();
    M(i, 2) = p.z();
    M(i, 3) = 1.0;
    b(i) = -(p.squaredNorm());
  }

  Eigen::Vector4d x = M.colPivHouseholderQr().solve(b);

  double cx = -x(0) / 2.0;
  double cy = -x(1) / 2.0;
  double cz = -x(2) / 2.0;
  double r_sq = cx * cx + cy * cy + cz * cz - x(3);

  if (r_sq <= 1e-12) {
    return false;
  }

  double r = std::sqrt(r_sq);
  (*sphere) << cx, cy, cz, std::log(r);
  return true;
}

void SphereEstimator::generate_models(std::vector<V4D> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  V3D p1 = points_.col(sample[0]);
  V3D p2 = points_.col(sample[1]);
  V3D p3 = points_.col(sample[2]);
  V3D p4 = points_.col(sample[3]);

  V4D sphere;
  if (FitSphereFrom4Points(p1, p2, p3, p4, &sphere)) {
    models->push_back(sphere);
  }
}

double SphereEstimator::score_model(const V4D &sphere,
                                    size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  V3D center = sphere.head<3>();
  double r = std::exp(sphere(3)); // r = exp(log_r)

  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    // Distance from point to sphere surface
    double dist_to_center = (point - center).norm();
    double dist_to_surface = std::abs(dist_to_center - r);
    double error = dist_to_surface * dist_to_surface;

    if (error < sq_threshold) {
      (*inlier_count)++;
      score += error;
    } else {
      score += sq_threshold;
    }
  }

  return score;
}

void SphereEstimator::refine_model(V4D *sphere) const {
  const double sq_threshold = max_error_ * max_error_;

  V3D center = sphere->head<3>();
  double r = std::exp((*sphere)(3));

  // Collect inliers
  std::vector<V3D> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist_to_center = (point - center).norm();
    double dist_to_surface = std::abs(dist_to_center - r);
    if (dist_to_surface * dist_to_surface < sq_threshold) {
      inliers.push_back(point);
    }
  }

  // Refine with least squares on inliers (need at least 4 points)
  if (inliers.size() < 4)
    return;

  V4D refined_sphere;
  if (FitSphereLeastSquares(inliers, &refined_sphere)) {
    *sphere = refined_sphere;
  }
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<V4D> EstimateSphereRobust(const std::vector<V3D> &points,
                                        double max_error,
                                        const poselib::RansacOptions &options,
                                        poselib::RansacStats *stats) {
  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  SphereEstimator solver(max_error, options, points_mat);

  // Check if we have enough points for minimal solver
  if (solver.num_data < solver.sample_sz) {
    return std::nullopt;
  }

  V4D best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<SphereEstimator, V4D>(solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid
  if (!std::isfinite(best_model(0)) || !std::isfinite(best_model(1)) ||
      !std::isfinite(best_model(2)) || !std::isfinite(best_model(3))) {
    return std::nullopt;
  }

  return best_model;
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<V4D> EstimateSphere(const std::vector<V3D> &points) {
  if (points.size() < 4) {
    return std::nullopt;
  }

  const size_t n = points.size();
  Eigen::MatrixXd M(n, 4);
  Eigen::VectorXd b(n);

  for (size_t i = 0; i < n; ++i) {
    const V3D &p = points[i];
    M(i, 0) = p.x();
    M(i, 1) = p.y();
    M(i, 2) = p.z();
    M(i, 3) = 1.0;
    b(i) = -(p.squaredNorm());
  }

  Eigen::Vector4d x = M.colPivHouseholderQr().solve(b);

  double cx = -x(0) / 2.0;
  double cy = -x(1) / 2.0;
  double cz = -x(2) / 2.0;
  double r_sq = cx * cx + cy * cy + cz * cz - x(3);

  if (r_sq <= 1e-12) {
    LOG(WARNING) << "EstimateSphere: Sphere fitting resulted in invalid radius";
    return std::nullopt;
  }

  double r = std::sqrt(r_sq);
  V4D sphere;
  sphere << cx, cy, cz, std::log(r);

  return sphere;
}

} // namespace group3d

} // namespace estimators

} // namespace limap
