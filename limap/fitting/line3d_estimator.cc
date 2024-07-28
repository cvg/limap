#include "fitting/line3d_estimator.h"

namespace limap {

namespace fitting {

std::pair<Line3d, ransac_lib::RansacStatistics>
Fit3DPoints(const Eigen::Matrix3Xd points,
            const ransac_lib::LORansacOptions &options_) {
  ransac_lib::LORansacOptions options = options_;
  std::random_device rand_dev;
  options.random_seed_ = rand_dev();

  // ransac
  Line3dEstimator solver(points);
  ransac_lib::LocallyOptimizedMSAC<InfiniteLine3d, std::vector<InfiniteLine3d>,
                                   Line3dEstimator>
      lomsac;
  InfiniteLine3d best_model;
  ransac_lib::RansacStatistics ransac_stats;
  int num_ransac_inliers =
      lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

  // get line segment
  std::vector<double> projections;
  Eigen::Vector3d direc = best_model.direction();
  if (ransac_stats.inlier_indices.empty()) {
    return std::make_pair(Line3d(), ransac_stats);
  }
  // use the first one as reference and do the projections
  V3D p_ref = points.col(ransac_stats.inlier_indices[0]);
  for (auto it = ransac_stats.inlier_indices.begin();
       it != ransac_stats.inlier_indices.end(); ++it) {
    Eigen::Vector3d p = points.col(*it);
    double projection = (p - p_ref).dot(direc);
    projections.push_back(projection);
  }
  std::sort(projections.begin(), projections.end());
  size_t n_projs = projections.size();
  Eigen::Vector3d start = p_ref + direc * projections[0];
  Eigen::Vector3d end = p_ref + direc * projections[n_projs - 1];
  Line3d line(start, end);
  return std::make_pair(line, ransac_stats);
}

Line3dEstimator::Line3dEstimator(const Eigen::Matrix3Xd &data) {
  data_ = data;
  num_data_ = data_.cols();
}

int Line3dEstimator::MinimalSolver(const std::vector<int> &sample,
                                   std::vector<InfiniteLine3d> *lines) const {
  lines->clear();
  if (sample.size() < 2u)
    return 0;

  lines->resize(1);
  Eigen::Vector3d p1 = data_.col(sample[0]);
  Eigen::Vector3d p2 = data_.col(sample[1]);
  Line3d seg = Line3d(p1, p2);
  if (seg.length() < EPS) {
    lines->clear();
    return 0;
  }
  if (std::isnan(seg.length()) || seg.length() < EPS)
    return 0;
  (*lines)[0] = InfiniteLine3d(seg);
  return 1;
}

int Line3dEstimator::NonMinimalSolver(const std::vector<int> &sample,
                                      InfiniteLine3d *line) const {
  if (sample.size() < 6u)
    return 0;

  const int kNumSamples = static_cast<int>(sample.size());

  // We fit the line by estimating the eigenvectors of the covariance matrix
  // of the data.
  Eigen::Vector3d center(0.0, 0.0, 0.0);
  for (int i = 0; i < kNumSamples; ++i) {
    center += data_.col(sample[i]);
  }
  center /= static_cast<double>(kNumSamples);

  // Builds the covariance matrix C.
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

  for (int i = 0; i < kNumSamples; ++i) {
    Eigen::Vector3d d = data_.col(sample[i]) - center;
    C += d * d.transpose();
  }
  C /= static_cast<double>(kNumSamples - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_solver(C);
  if (eig_solver.info() != Eigen::Success)
    return 0;
  Eigen::Vector3d direc = eig_solver.eigenvectors().col(0);
  *line = InfiniteLine3d(center, direc);
  return 1;
}

// Evaluates the line on the i-th data point.
double Line3dEstimator::EvaluateModelOnPoint(const InfiniteLine3d &line,
                                             int i) const {
  Eigen::Vector3d p = data_.col(i);
  Eigen::Vector3d proj = line.point_projection(p);
  return (p - proj).squaredNorm();
}

} // namespace fitting

} // namespace limap
