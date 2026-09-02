#include "limap/estimators/group3d/cylinder_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Cylinder Estimator for RANSAC (internal implementation)
// Uses 5 points for minimal solver (5 DOF cylinder)
class CylinderEstimator {
public:
  // Model type: (axis_point, axis_dir, radius)
  struct CylinderModel {
    V3D axis_point; // A point on the axis (closest to origin)
    V3D axis_dir;   // Unit direction of the axis
    double radius;  // Cylinder radius

    // Distance from a point to the cylinder surface
    double DistanceToSurface(const V3D &point) const {
      V3D v = point - axis_point;
      V3D v_perp = v - v.dot(axis_dir) * axis_dir;
      return std::abs(v_perp.norm() - radius);
    }
  };

  CylinderEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                    const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<CylinderModel> *models);
  double score_model(const CylinderModel &cylinder, size_t *inlier_count) const;
  void refine_model(CylinderModel *cylinder) const;

  const size_t sample_sz = 5; // 5 points for cylinder (5 DOF)
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;

  // Helper: fit cylinder to points using PCA + circle fitting
  static bool FitCylinderFromPoints(const std::vector<V3D> &points,
                                    CylinderModel *cylinder);

  // Helper: fit circle to 2D points, returns (center, radius)
  static bool FitCircle2D(const std::vector<V2D> &points, V2D *center,
                          double *radius);
};

// ============================================================================
// CylinderEstimator implementation
// ============================================================================

bool CylinderEstimator::FitCircle2D(const std::vector<V2D> &points, V2D *center,
                                    double *radius) {
  const size_t n = points.size();
  if (n < 3) {
    return false;
  }

  // Algebraic circle fitting: (x-cx)^2 + (y-cy)^2 = r^2
  // Expanding: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
  // Let A = -2*cx, B = -2*cy, C = cx^2 + cy^2 - r^2
  // Then: x^2 + y^2 + A*x + B*y + C = 0
  // Rearrange: A*x + B*y + C = -(x^2 + y^2)

  Eigen::MatrixXd M(n, 3);
  Eigen::VectorXd b(n);

  for (size_t i = 0; i < n; ++i) {
    const V2D &p = points[i];
    M(i, 0) = p.x();
    M(i, 1) = p.y();
    M(i, 2) = 1.0;
    b(i) = -(p.squaredNorm());
  }

  Eigen::Vector3d x = M.colPivHouseholderQr().solve(b);

  double cx = -x(0) / 2.0;
  double cy = -x(1) / 2.0;
  double r_sq = cx * cx + cy * cy - x(2);

  if (r_sq <= 1e-12) {
    return false;
  }

  *center = V2D(cx, cy);
  *radius = std::sqrt(r_sq);
  return true;
}

bool CylinderEstimator::FitCylinderFromPoints(const std::vector<V3D> &points,
                                              CylinderModel *cylinder) {
  const size_t n = points.size();
  if (n < 5) {
    return false;
  }

  // Step 1: Compute centroid
  V3D centroid = V3D::Zero();
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(n);

  // Step 2: Compute covariance matrix and PCA
  M3D cov = M3D::Zero();
  for (const auto &p : points) {
    V3D d = p - centroid;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(n);

  Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  // Eigenvectors sorted in ascending order of eigenvalues
  // The cylinder axis should be along the direction of maximum variance
  V3D axis_dir = solver.eigenvectors().col(2).normalized();

  // Step 3: Project points onto plane perpendicular to axis
  // and fit a circle to find axis position and radius
  std::vector<V2D> points_2d;
  points_2d.reserve(n);

  // Create orthonormal basis for the perpendicular plane
  V3D u, v;
  if (std::abs(axis_dir.x()) < 0.9) {
    u = axis_dir.cross(V3D::UnitX()).normalized();
  } else {
    u = axis_dir.cross(V3D::UnitY()).normalized();
  }
  v = axis_dir.cross(u).normalized();

  for (const auto &p : points) {
    V3D d = p - centroid;
    points_2d.emplace_back(d.dot(u), d.dot(v));
  }

  V2D center_2d;
  double radius;
  if (!FitCircle2D(points_2d, &center_2d, &radius)) {
    return false;
  }

  // Step 4: Compute axis point (closest point to origin on axis)
  V3D axis_point_on_centroid = centroid + center_2d.x() * u + center_2d.y() * v;

  // Project to get closest point to origin
  double t = axis_point_on_centroid.dot(axis_dir);
  V3D axis_point = axis_point_on_centroid - t * axis_dir;

  cylinder->axis_point = axis_point;
  cylinder->axis_dir = axis_dir;
  cylinder->radius = radius;

  return true;
}

void CylinderEstimator::generate_models(std::vector<CylinderModel> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  std::vector<V3D> sample_points;
  sample_points.reserve(sample_sz);
  for (size_t idx : sample) {
    sample_points.push_back(points_.col(idx));
  }

  CylinderModel cylinder;
  if (FitCylinderFromPoints(sample_points, &cylinder)) {
    models->push_back(cylinder);
  }
}

double CylinderEstimator::score_model(const CylinderModel &cylinder,
                                      size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = cylinder.DistanceToSurface(point);
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

void CylinderEstimator::refine_model(CylinderModel *cylinder) const {
  const double sq_threshold = max_error_ * max_error_;

  // Collect inliers
  std::vector<V3D> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D point = points_.col(i);
    double dist = cylinder->DistanceToSurface(point);
    if (dist * dist < sq_threshold) {
      inliers.push_back(point);
    }
  }

  // Refine with all inliers (need at least 5 points)
  if (inliers.size() < 5)
    return;

  CylinderModel refined;
  if (FitCylinderFromPoints(inliers, &refined)) {
    *cylinder = refined;
  }
}

// Convert CylinderModel to minimal 7D representation [quat(4), wvec(2), log_r]
std::vector<double>
CylinderToMinimalParams(const CylinderEstimator::CylinderModel &model) {
  InfiniteLine3d inf_line(model.axis_point, model.axis_dir, true);
  MinimalInfiniteLine3d minimal_line(inf_line);

  std::vector<double> params(7);
  for (int i = 0; i < 6; ++i) {
    params[i] = minimal_line.data[i];
  }
  params[6] = std::log(model.radius);
  return params;
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<std::vector<double>>
EstimateCylinderRobust(const std::vector<V3D> &points, double max_error,
                       const poselib::RansacOptions &options,
                       poselib::RansacStats *stats) {
  if (points.size() < 5) {
    LOG(WARNING) << "EstimateCylinderRobust: Need at least 5 points, got "
                 << points.size();
    return std::nullopt;
  }

  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  CylinderEstimator solver(max_error, options, points_mat);

  CylinderEstimator::CylinderModel best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<CylinderEstimator, CylinderEstimator::CylinderModel>(
          solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid
  if (!std::isfinite(best_model.axis_point.x()) ||
      !std::isfinite(best_model.axis_dir.x()) ||
      !std::isfinite(best_model.radius) || best_model.radius <= 0) {
    return std::nullopt;
  }

  return CylinderToMinimalParams(best_model);
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<std::vector<double>>
EstimateCylinder(const std::vector<V3D> &points) {
  if (points.size() < 5) {
    LOG(WARNING) << "EstimateCylinder: Need at least 5 points, got "
                 << points.size();
    return std::nullopt;
  }

  CylinderEstimator::CylinderModel cylinder;

  // Use the static helper directly
  // Step 1: Compute centroid
  V3D centroid = V3D::Zero();
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(points.size());

  // Step 2: Compute covariance matrix and PCA
  M3D cov = M3D::Zero();
  for (const auto &p : points) {
    V3D d = p - centroid;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(points.size());

  Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
  if (solver.info() != Eigen::Success) {
    LOG(WARNING) << "EstimateCylinder: PCA failed";
    return std::nullopt;
  }

  V3D axis_dir = solver.eigenvectors().col(2).normalized();

  // Step 3: Project points onto plane perpendicular to axis
  std::vector<V2D> points_2d;
  points_2d.reserve(points.size());

  V3D u, v;
  if (std::abs(axis_dir.x()) < 0.9) {
    u = axis_dir.cross(V3D::UnitX()).normalized();
  } else {
    u = axis_dir.cross(V3D::UnitY()).normalized();
  }
  v = axis_dir.cross(u).normalized();

  for (const auto &p : points) {
    V3D d = p - centroid;
    points_2d.emplace_back(d.dot(u), d.dot(v));
  }

  // Fit circle
  const size_t n = points_2d.size();
  Eigen::MatrixXd M(n, 3);
  Eigen::VectorXd b(n);

  for (size_t i = 0; i < n; ++i) {
    const V2D &p = points_2d[i];
    M(i, 0) = p.x();
    M(i, 1) = p.y();
    M(i, 2) = 1.0;
    b(i) = -(p.squaredNorm());
  }

  Eigen::Vector3d x = M.colPivHouseholderQr().solve(b);

  double cx = -x(0) / 2.0;
  double cy = -x(1) / 2.0;
  double r_sq = cx * cx + cy * cy - x(2);

  if (r_sq <= 1e-12) {
    LOG(WARNING) << "EstimateCylinder: Circle fitting failed";
    return std::nullopt;
  }

  double radius = std::sqrt(r_sq);
  V3D axis_point_on_centroid = centroid + cx * u + cy * v;

  // Project to get closest point to origin
  double t = axis_point_on_centroid.dot(axis_dir);

  CylinderEstimator::CylinderModel model;
  model.axis_point = axis_point_on_centroid - t * axis_dir;
  model.axis_dir = axis_dir;
  model.radius = radius;
  return CylinderToMinimalParams(model);
}

} // namespace group3d

} // namespace estimators

} // namespace limap
