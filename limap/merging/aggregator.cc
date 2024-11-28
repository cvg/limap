#include "merging/aggregator.h"
#include <colmap/util/logging.h>

namespace limap {

namespace merging {

Line3d
Aggregator::aggregate_line3d_list_takebest(const std::vector<Line3d> &lines,
                                           const std::vector<double> &scores) {
  // For now the line with the best score is chosen
  THROW_CHECK_EQ(lines.size(), scores.size());
  int n_lines = lines.size();

  double best_score = 0.0;
  int best_idx = -1;
  double min_uncertainty = std::numeric_limits<double>::max();
  for (int i = 0; i < n_lines; ++i) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
    if (lines[i].uncertainty < min_uncertainty)
      min_uncertainty = lines[i].uncertainty;
  }
  Line3d best_line = lines[best_idx];
  best_line.uncertainty = min_uncertainty;
  return best_line;
}

Line3d Aggregator::aggregate_line3d_list_takelongest(
    const std::vector<Line3d> &lines, const std::vector<double> &scores) {
  // For now the line with the best score is chosen
  THROW_CHECK_EQ(lines.size(), scores.size());
  int n_lines = lines.size();

  double best_length = 0.0;
  int best_idx = -1;
  double min_uncertainty = std::numeric_limits<double>::max();
  for (int i = 0; i < n_lines; ++i) {
    if (lines[i].length() > best_length) {
      best_length = lines[i].length();
      best_idx = i;
    }
    if (lines[i].uncertainty < min_uncertainty)
      min_uncertainty = lines[i].uncertainty;
  }
  Line3d best_line = lines[best_idx];
  best_line.uncertainty = min_uncertainty;
  return best_line;
}

Line3d Aggregator::aggregate_line3d_list(const std::vector<Line3d> &lines,
                                         const std::vector<double> &scores,
                                         const int num_outliers) {
  THROW_CHECK_EQ(lines.size(), scores.size());
  int n_lines = lines.size();
  if (n_lines < 4) {
    return aggregate_line3d_list_takebest(lines, scores);
    // return aggregate_line3d_list_takelongest(lines, scores);
  }

  // total least square on endpoints
  V3D center(0.0, 0.0, 0.0);
  for (size_t i = 0; i < n_lines; ++i) {
    center += lines[i].start;
    center += lines[i].end;
  }
  center = center / (2 * n_lines);
  Eigen::MatrixXd endpoints;
  endpoints.resize(n_lines * 2, 3);
  for (size_t i = 0; i < n_lines; ++i) {
    endpoints.row(2 * i) = lines[i].start - center;
    endpoints.row(2 * i + 1) = lines[i].end - center;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(endpoints, Eigen::ComputeThinV);
  V3D direc = svd.matrixV().col(0);
  direc = direc / direc.norm();

  // projection
  std::vector<double> projections;
  for (size_t i = 0; i < n_lines; ++i) {
    projections.push_back((lines[i].start - center).dot(direc));
    projections.push_back((lines[i].end - center).dot(direc));
  }
  std::sort(projections.begin(), projections.end());

  // uncertainty
  double min_uncertainty = std::numeric_limits<double>::max();
  for (size_t i = 0; i < n_lines; ++i) {
    if (lines[i].uncertainty < min_uncertainty)
      min_uncertainty = lines[i].uncertainty;
  }

  // construct final line
  Line3d final_line;
  final_line.start = center + direc * projections[num_outliers];
  final_line.end = center + direc * projections[n_lines * 2 - 1 - num_outliers];
  final_line.uncertainty = min_uncertainty;
  return final_line;
}

} // namespace merging

} // namespace limap
