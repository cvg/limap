#include "limap/estimators/group3d/vp_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>
#include <colmap/util/logging.h>

namespace limap {

namespace estimators {

namespace group3d {

namespace {

// VP Estimator for RANSAC (internal implementation)
class VPEstimator {
public:
  VPEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
              const std::vector<Line3d> &lines)
      : num_data(lines.size()), max_error_(max_error), opt(ransac_opt),
        lines_(lines), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<V3D> *models);
  double score_model(const V3D &direction, size_t *inlier_count) const;
  void refine_model(V3D *direction) const;

  const size_t sample_sz = 1;
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const std::vector<Line3d> &lines_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;
};

// ============================================================================
// VPEstimator implementation
// ============================================================================

void VPEstimator::generate_models(std::vector<V3D> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  // Minimal solver: VP from 1 line direction
  V3D dir = lines_[sample[0]].Direction().normalized();

  if (!std::isnan(dir.norm()) && dir.norm() >= 1e-12) {
    models->push_back(dir);
  }
}

double VPEstimator::score_model(const V3D &direction,
                                size_t *inlier_count) const {
  const double threshold = max_error_;
  double score = 0.0;
  *inlier_count = 0;

  for (size_t i = 0; i < num_data; ++i) {
    // Angular error: 1 - |cos(angle)| = 1 - |dot(dir1, dir2)|
    V3D line_dir = lines_[i].Direction().normalized();
    double cos_angle = std::abs(line_dir.dot(direction));
    double error = 1.0 - cos_angle;

    if (error < threshold) {
      (*inlier_count)++;
      score += error;
    } else {
      score += threshold;
    }
  }

  return score;
}

void VPEstimator::refine_model(V3D *direction) const {
  const double threshold = max_error_;

  // Collect inliers
  std::vector<size_t> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    V3D line_dir = lines_[i].Direction().normalized();
    double cos_angle = std::abs(line_dir.dot(*direction));
    double error = 1.0 - cos_angle;
    if (error < threshold) {
      inliers.push_back(i);
    }
  }

  // Refine with PCA on inlier line directions (need at least 2)
  if (inliers.size() < 2)
    return;

  // Build covariance matrix
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (size_t idx : inliers) {
    V3D d = lines_[idx].Direction().normalized();
    C += d * d.transpose();
  }
  C /= static_cast<double>(inliers.size());

  // Find principal direction (largest eigenvalue)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() == Eigen::Success) {
    *direction = eig_solver.eigenvectors().col(2).normalized();
  }
}

} // namespace

// ============================================================================
// Robust fitting functions
// ============================================================================

std::optional<V3D> EstimateVPRobust(const std::vector<Line3d> &lines,
                                    double max_error,
                                    const poselib::RansacOptions &options,
                                    poselib::RansacStats *stats) {
  VPEstimator solver(max_error, options, lines);

  // Check if we have enough lines for minimal solver
  if (solver.num_data < solver.sample_sz) {
    return std::nullopt;
  }

  V3D best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<VPEstimator, V3D>(solver, options, &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Check if model is valid and ensure normalization
  if (std::isnan(best_model.norm()) || best_model.norm() < 1e-12) {
    return std::nullopt;
  }

  return best_model.normalized();
}

// ============================================================================
// Non-robust estimation functions
// ============================================================================

std::optional<V3D> EstimateVP(const std::vector<Line3d> &lines) {
  if (lines.empty()) {
    return std::nullopt;
  }

  const size_t n_lines = lines.size();

  // For a single line, just return its direction
  if (n_lines == 1) {
    V3D dir = lines[0].Direction();
    return dir.normalized();
  }

  // For multiple lines, use PCA on directions
  // Build covariance matrix of directions
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

  for (const auto &line : lines) {
    V3D d = line.Direction().normalized();
    // Note: for VP, directions d and -d are equivalent
    // We use the outer product to make it symmetric
    C += d * d.transpose();
  }
  C /= static_cast<double>(n_lines);

  // Find the eigenvector corresponding to the largest eigenvalue
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() != Eigen::Success) {
    LOG(WARNING) << "EstimateVP: Eigen decomposition failed";
    return std::nullopt;
  }

  // Eigenvalues are sorted in increasing order, so take the last one
  V3D direction = eig_solver.eigenvectors().col(2);

  return direction.normalized();
}

} // namespace group3d

} // namespace estimators

} // namespace limap
