#include "evaluation/base_evaluator.h"

#include <iostream>
#include <numeric>
#include <queue>

namespace limap {

namespace evaluation {

double BaseEvaluator::ComputeDistLine(const Line3d &line, int n_samples) {
  if (n_samples <= 2) {
    throw std::runtime_error("n_samples should be >= 3");
  }
  double interval = 1.0 / (n_samples - 1);
  std::vector<double> dists;
  for (int i = 0; i < n_samples; ++i) {
    V3D p = line.start + i * interval * (line.end - line.start);
    double dist = ComputeDistPoint(p);
    dists.push_back(dist);
  }
  double sum = std::accumulate(dists.begin(), dists.end(), 0.0);
  double avg_dist = sum / double(n_samples);
  return avg_dist;
}

double BaseEvaluator::ComputeInlierRatio(const Line3d &line, double threshold,
                                         int n_samples) {
  int counter = 0;
  double interval = 1.0 / n_samples;
  std::vector<double> dists(n_samples);
#pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    V3D p = line.start + (i + 0.5) * interval * (line.end - line.start);
    double dist = ComputeDistPoint(p);
    dists[i] = dist;
  }
  for (int i = 0; i < n_samples; ++i) {
    if (dists[i] <= threshold)
      counter++;
  }
  double ratio = double(counter) / n_samples;
  return ratio;
}

std::vector<Line3d> BaseEvaluator::ComputeInlierSegsOneLine(const Line3d &line,
                                                            double threshold,
                                                            int n_samples) {
  std::vector<Line3d> res;
  double interval = 1.0 / n_samples;
  std::vector<double> dists(n_samples);
  std::vector<bool> flags(n_samples, false);
#pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    V3D p = line.start + (i + 0.5) * interval * (line.end - line.start);
    double dist = ComputeDistPoint(p);
    dists[i] = dist;
    if (dist <= threshold)
      flags[i] = true;
  }
  int start = -1;
  int end = -1;
  for (int i = 0; i < n_samples; ++i) {
    if (flags[i]) {
      if (start == -1) {
        start = i;
        end = i + 1;
      } else {
        end = i + 1;
      }
    } else {
      if (start != -1) {
        V3D pstart = line.start + start * interval * (line.end - line.start);
        V3D pend = line.start + end * interval * (line.end - line.start);
        res.push_back(Line3d(pstart, pend));
        start = -1;
        end = -1;
      }
    }
  }
  if (start != -1) {
    V3D pstart = line.start + start * interval * (line.end - line.start);
    V3D pend = line.start + end * interval * (line.end - line.start);
    res.push_back(Line3d(pstart, pend));
  }
  return res;
}

std::vector<Line3d>
BaseEvaluator::ComputeInlierSegs(const std::vector<Line3d> &lines,
                                 double threshold, int n_samples) {
  std::vector<Line3d> res;
  for (auto it = lines.begin(); it != lines.end(); ++it) {
    auto res_it = ComputeInlierSegsOneLine(*it, threshold, n_samples);
    res.insert(res.end(), res_it.begin(), res_it.end());
  }
  return res;
}

std::vector<Line3d> BaseEvaluator::ComputeOutlierSegsOneLine(const Line3d &line,
                                                             double threshold,
                                                             int n_samples) {
  std::vector<Line3d> res;
  double interval = 1.0 / n_samples;
  std::vector<double> dists(n_samples);
  std::vector<bool> flags(n_samples, true);
#pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    V3D p = line.start + (i + 0.5) * interval * (line.end - line.start);
    double dist = ComputeDistPoint(p);
    dists[i] = dist;
    if (dist <= threshold)
      flags[i] = false;
  }
  int start = -1;
  int end = -1;
  for (int i = 0; i < n_samples; ++i) {
    if (flags[i]) {
      if (start == -1) {
        start = i;
        end = i + 1;
      } else {
        end = i + 1;
      }
    } else {
      if (start != -1) {
        V3D pstart = line.start + start * interval * (line.end - line.start);
        V3D pend = line.start + end * interval * (line.end - line.start);
        res.push_back(Line3d(pstart, pend));
        start = -1;
        end = -1;
      }
    }
  }
  if (start != -1) {
    V3D pstart = line.start + start * interval * (line.end - line.start);
    V3D pend = line.start + end * interval * (line.end - line.start);
    res.push_back(Line3d(pstart, pend));
  }
  return res;
}

std::vector<Line3d>
BaseEvaluator::ComputeOutlierSegs(const std::vector<Line3d> &lines,
                                  double threshold, int n_samples) {
  std::vector<Line3d> res;
  for (auto it = lines.begin(); it != lines.end(); ++it) {
    auto res_it = ComputeOutlierSegsOneLine(*it, threshold, n_samples);
    res.insert(res.end(), res_it.begin(), res_it.end());
  }
  return res;
}

} // namespace evaluation

} // namespace limap
