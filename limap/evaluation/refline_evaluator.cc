#include "evaluation/refline_evaluator.h"

#include <iostream>
#include <numeric>
#include <queue>

namespace limap {

namespace evaluation {

double RefLineEvaluator::SumLength() const {
  double sum_length = 0;
  for (auto it = ref_lines_.begin(); it != ref_lines_.end(); ++it) {
    sum_length += it->length();
  }
  return sum_length;
}

double RefLineEvaluator::ComputeRecallLength(
    const std::vector<Line3d> &ref_lines, const std::vector<Line3d> &lines,
    const double threshold, const int num_samples) const {
  size_t n_ref_lines = ref_lines.size();
  double recall = 0;
  for (size_t ref_line_id = 0; ref_line_id < n_ref_lines; ++ref_line_id) {
    const Line3d &ref_line = ref_lines[ref_line_id];
    double interval = ref_line.length() / (num_samples - 1);
    std::vector<V3D> points;
    for (size_t i = 0; i < num_samples; ++i) {
      points.push_back(ref_line.start + interval * i * ref_line.direction());
    }
    std::vector<int> flags(num_samples, 0);
#pragma omp parallel for
    for (size_t i = 0; i < num_samples; ++i) {
      double dist = DistPointLines(points[i], lines);
      if (dist < threshold)
        flags[i] = 1;
    }
    int counter = 0;
    for (size_t i = 0; i < num_samples; ++i)
      counter += flags[i];
    recall += ref_line.length() * double(counter) / num_samples;
  }
  return recall;
}

double RefLineEvaluator::ComputeRecallRef(const std::vector<Line3d> &lines,
                                          const double threshold,
                                          const int num_samples) const {
  return ComputeRecallLength(ref_lines_, lines, threshold, num_samples);
}

double RefLineEvaluator::ComputeRecallTested(const std::vector<Line3d> &lines,
                                             const double threshold,
                                             const int num_samples) const {
  return ComputeRecallLength(lines, ref_lines_, threshold, num_samples);
}

double RefLineEvaluator::DistPointLine(const V3D &p, const Line3d &line) const {
  double dist_start = (p - line.start).norm();
  double dist_end = (p - line.end).norm();

  double dist_perp_squared = (p - line.start).squaredNorm() -
                             pow((p - line.start).dot(line.direction()), 2);
  double dist_perp = sqrt(std::max(dist_perp_squared, 0.0));
  return std::min(dist_perp, std::min(dist_start, dist_end));
}

double
RefLineEvaluator::DistPointLines(const V3D &p,
                                 const std::vector<Line3d> &lines) const {
  double min_dist = std::numeric_limits<double>::max();
  for (auto it = lines.begin(); it != lines.end(); ++it) {
    double dist = DistPointLine(p, *it);
    if (dist < min_dist)
      min_dist = dist;
    if (min_dist < EPS)
      return 0.0;
  }
  return min_dist;
}

} // namespace evaluation

} // namespace limap
