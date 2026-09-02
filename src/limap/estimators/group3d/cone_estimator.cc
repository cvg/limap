#include "limap/estimators/group3d/cone_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// Cone model for RANSAC
struct ConeModel {
  V3D apex;
  V3D direction; // Unit vector
  double tan_alpha;

  // Distance from point to cone surface
  double DistanceToSurface(const V3D &point) const {
    V3D v = point - apex;
    double h = v.dot(direction);
    V3D v_perp = v - h * direction;
    double r = v_perp.norm();

    // In the (|h|, r) plane, the cone is r = |h| * tan_alpha
    // Distance to the nearest arm of the V-shaped cone
    double cos_a = 1.0 / std::sqrt(1.0 + tan_alpha * tan_alpha);
    double sin_a = tan_alpha * cos_a;

    // Project onto the nearer arm
    double t_right = h * cos_a + r * sin_a;
    double t_left = -h * cos_a + r * sin_a;

    double h_proj, r_proj;
    if (t_right >= t_left) { // equivalently, h >= 0
      h_proj = t_right * cos_a;
      r_proj = t_right * sin_a;
    } else {
      h_proj = -t_left * cos_a;
      r_proj = t_left * sin_a;
    }

    // Distance in the (h, r) plane
    double dh = h - h_proj;
    double dr = r - r_proj;
    return std::sqrt(dh * dh + dr * dr);
  }
};

// Fit cone from points using PCA + linear fit
// Returns true on success
bool FitConeFromPoints(const std::vector<V3D> &points, ConeModel *cone) {
  const size_t n = points.size();
  if (n < 6) {
    return false;
  }

  // Step 1: Compute centroid
  V3D centroid = V3D::Zero();
  for (const auto &p : points) {
    centroid += p;
  }
  centroid /= static_cast<double>(n);

  // Step 2: PCA to find axis direction (principal eigenvector)
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

  // Principal eigenvector (largest eigenvalue is last for
  // SelfAdjointEigenSolver)
  V3D axis = solver.eigenvectors().col(2).normalized();

  // Step 3: Project all points to get (h, r) pairs
  std::vector<double> abs_h_vals(n), r_vals(n);
  for (size_t i = 0; i < n; ++i) {
    V3D d = points[i] - centroid;
    double h = d.dot(axis);
    V3D perp = d - h * axis;
    abs_h_vals[i] = std::abs(h);
    r_vals[i] = perp.norm();
  }

  // Step 4: Linear fit r = slope * |h| + intercept
  // Using least squares: [|h|, 1] * [slope; intercept] = r
  double sum_hh = 0, sum_h = 0, sum_r = 0, sum_hr = 0;
  for (size_t i = 0; i < n; ++i) {
    sum_hh += abs_h_vals[i] * abs_h_vals[i];
    sum_h += abs_h_vals[i];
    sum_r += r_vals[i];
    sum_hr += abs_h_vals[i] * r_vals[i];
  }

  double det = static_cast<double>(n) * sum_hh - sum_h * sum_h;
  if (std::abs(det) < 1e-12) {
    return false;
  }

  double slope = (static_cast<double>(n) * sum_hr - sum_h * sum_r) / det;
  double intercept = (sum_hh * sum_r - sum_h * sum_hr) / det;

  // slope should be positive for a valid cone
  if (slope <= 1e-8) {
    return false;
  }

  // Step 5: Compute apex position
  // Where r extrapolates to 0: |h_apex| = -intercept / slope
  // apex = centroid - (intercept / slope) * axis
  // (negative because when r=0, we're at the apex, offset from centroid)
  double h_apex = -intercept / slope;
  V3D apex = centroid + h_apex * axis;

  cone->apex = apex;
  cone->direction = axis;
  cone->tan_alpha = slope;

  return true;
}

// Convert ConeModel to 7D parameter vector [apex(3), direction(3),
// log_tan_alpha(1)]
std::vector<double> ConeToParams(const ConeModel &model) {
  std::vector<double> params(7);
  params[0] = model.apex.x();
  params[1] = model.apex.y();
  params[2] = model.apex.z();
  params[3] = model.direction.x();
  params[4] = model.direction.y();
  params[5] = model.direction.z();
  params[6] = std::log(model.tan_alpha);
  return params;
}

// RANSAC estimator class
class ConeEstimator {
public:
  using ConeModelType = ConeModel;

  ConeEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                const Eigen::Matrix3Xd &points)
      : num_data(points.cols()), max_error_(max_error), opt(ransac_opt),
        points_(points), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<ConeModel> *models) {
    models->clear();
    sampler.generate_sample(&sample);

    std::vector<V3D> sample_points;
    sample_points.reserve(sample_sz);
    for (size_t idx : sample) {
      sample_points.push_back(points_.col(idx));
    }

    ConeModel cone;
    if (FitConeFromPoints(sample_points, &cone)) {
      models->push_back(cone);
    }
  }

  double score_model(const ConeModel &cone, size_t *inlier_count) const {
    const double sq_threshold = max_error_ * max_error_;
    double score = 0.0;
    *inlier_count = 0;

    for (size_t i = 0; i < num_data; ++i) {
      V3D point = points_.col(i);
      double dist = cone.DistanceToSurface(point);
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

  void refine_model(ConeModel *cone) const {
    const double sq_threshold = max_error_ * max_error_;

    // Collect inliers
    std::vector<V3D> inliers;
    for (size_t i = 0; i < num_data; ++i) {
      V3D point = points_.col(i);
      double dist = cone->DistanceToSurface(point);
      if (dist * dist < sq_threshold) {
        inliers.push_back(point);
      }
    }

    if (inliers.size() < 6)
      return;

    // Refit with all inliers
    ConeModel refined;
    if (FitConeFromPoints(inliers, &refined)) {
      *cone = refined;
    }
  }

  const size_t sample_sz = 6; // 6 points for cone (6 DOF)
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &points_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;
};

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<std::vector<double>>
EstimateConeRobust(const std::vector<V3D> &points, double max_error,
                   const poselib::RansacOptions &options,
                   poselib::RansacStats *stats) {
  if (points.size() < 6) {
    LOG(WARNING) << "EstimateConeRobust: Need at least 6 points, got "
                 << points.size();
    return std::nullopt;
  }

  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  ConeEstimator solver(max_error, options, points_mat);

  ConeModel best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<ConeEstimator, ConeModel>(solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid
  if (!std::isfinite(best_model.apex.x()) ||
      !std::isfinite(best_model.tan_alpha) || best_model.tan_alpha <= 0) {
    return std::nullopt;
  }

  return ConeToParams(best_model);
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<std::vector<double>>
EstimateCone(const std::vector<V3D> &points) {
  if (points.size() < 6) {
    LOG(WARNING) << "EstimateCone: Need at least 6 points, got "
                 << points.size();
    return std::nullopt;
  }

  ConeModel model;
  if (!FitConeFromPoints(points, &model)) {
    LOG(WARNING) << "EstimateCone: Cone fitting failed";
    return std::nullopt;
  }

  return ConeToParams(model);
}

} // namespace group3d

} // namespace estimators

} // namespace limap
