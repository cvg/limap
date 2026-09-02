#include "limap/estimators/line3d/line3d_estimator.h"

#include <Eigen/Eigenvalues>
#include <PoseLib/camera_pose.h>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/sampling.h>

namespace limap {

namespace estimators {

namespace line3d {

namespace {

// ============================================================================
// Line3dEstimator for RANSAC (internal implementation)
// ============================================================================

class Line3dEstimator {
public:
  Line3dEstimator(double max_error, const poselib::RansacOptions &ransac_opt,
                  const Eigen::Matrix3Xd &data)
      : num_data(data.cols()), max_error_(max_error), opt(ransac_opt),
        data_(data), sampler(num_data, sample_sz, opt.seed) {
    sample.resize(sample_sz);
  }

  void generate_models(std::vector<InfiniteLine3d> *models);
  double score_model(const InfiniteLine3d &line, size_t *inlier_count) const;
  void refine_model(InfiniteLine3d *line) const;

  const size_t sample_sz = 2;
  const size_t num_data;

private:
  double max_error_;
  const poselib::RansacOptions &opt;
  const Eigen::Matrix3Xd &data_;

  poselib::RandomSampler sampler;
  std::vector<size_t> sample;
};

void Line3dEstimator::generate_models(std::vector<InfiniteLine3d> *models) {
  models->clear();
  sampler.generate_sample(&sample);

  // Minimal solver: fit line from 2 points
  Eigen::Vector3d p1 = data_.col(sample[0]);
  Eigen::Vector3d p2 = data_.col(sample[1]);
  Line3d seg = Line3d(p1, p2);

  if (!std::isnan(seg.Length()) && seg.Length() >= 1e-12) {
    models->push_back(InfiniteLine3d(seg));
  }
}

double Line3dEstimator::score_model(const InfiniteLine3d &line,
                                    size_t *inlier_count) const {
  const double sq_threshold = max_error_ * max_error_;
  double score = 0.0;
  *inlier_count = 0;

  for (size_t i = 0; i < num_data; ++i) {
    // Compute squared distance from point to line
    Eigen::Vector3d p = data_.col(i);
    Eigen::Vector3d proj = line.PointProjection(p);
    double error = (p - proj).squaredNorm();

    if (error < sq_threshold) {
      (*inlier_count)++;
      score += error;
    } else {
      score += sq_threshold;
    }
  }

  return score;
}

void Line3dEstimator::refine_model(InfiniteLine3d *line) const {
  const double sq_threshold = max_error_ * max_error_;

  // Collect inliers
  std::vector<size_t> inliers;
  for (size_t i = 0; i < num_data; ++i) {
    Eigen::Vector3d p = data_.col(i);
    Eigen::Vector3d proj = line->PointProjection(p);
    if ((p - proj).squaredNorm() < sq_threshold) {
      inliers.push_back(i);
    }
  }

  // Refine with PCA on inliers (need at least 6 points)
  if (inliers.size() < 6)
    return;

  // Compute centroid
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (size_t idx : inliers) {
    center += data_.col(idx);
  }
  center /= static_cast<double>(inliers.size());

  // Build covariance matrix
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (size_t idx : inliers) {
    Eigen::Vector3d d = data_.col(idx) - center;
    C += d * d.transpose();
  }
  C /= static_cast<double>(inliers.size() - 1);

  // Find principal direction (largest eigenvalue)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() == Eigen::Success) {
    Eigen::Vector3d direc = eig_solver.eigenvectors().col(2);
    *line = InfiniteLine3d(center, direc);
  }
}

} // namespace

// ============================================================================
// Robust fitting function
// ============================================================================

std::optional<Line3d>
EstimateLine3DRobust(const std::vector<V3D> &points, double max_error,
                     const poselib::RansacOptions &options,
                     poselib::RansacStats *stats) {
  // Convert vector to Eigen matrix
  Eigen::Matrix3Xd points_mat(3, points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    points_mat.col(i) = points[i];
  }

  // Initialize estimator
  Line3dEstimator solver(max_error, options, points_mat);

  // Check if we have enough points for minimal solver
  if (solver.num_data < solver.sample_sz) {
    return std::nullopt;
  }

  // Run poselib RANSAC
  InfiniteLine3d best_model;
  poselib::RansacStats ransac_stats =
      poselib::ransac<Line3dEstimator, InfiniteLine3d>(solver, options,
                                                       &best_model);

  // Write stats if requested
  if (stats != nullptr) {
    *stats = ransac_stats;
  }

  // Extract inliers
  std::vector<int> inlier_indices;
  double squared_threshold = max_error * max_error;
  for (size_t i = 0; i < points.size(); ++i) {
    Eigen::Vector3d p = points_mat.col(i);
    Eigen::Vector3d proj = best_model.PointProjection(p);
    double error = (p - proj).squaredNorm();
    if (error < squared_threshold) {
      inlier_indices.push_back(i);
    }
  }

  // get line segment
  std::vector<double> projections;
  Eigen::Vector3d direc = best_model.Direction();
  // use the first one as reference and do the projections
  V3D p_ref = points_mat.col(inlier_indices[0]);
  for (auto it = inlier_indices.begin(); it != inlier_indices.end(); ++it) {
    Eigen::Vector3d p = points_mat.col(*it);
    double projection = (p - p_ref).dot(direc);
    projections.push_back(projection);
  }
  std::sort(projections.begin(), projections.end());
  size_t n_projs = projections.size();
  Eigen::Vector3d start = p_ref + direc * projections[0];
  Eigen::Vector3d end = p_ref + direc * projections[n_projs - 1];
  Line3d line(start, end);
  return line;
}

} // namespace line3d

} // namespace estimators

} // namespace limap
