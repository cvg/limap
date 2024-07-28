#ifndef LIMAP_EVALUATION_BASE_EVALUATOR_H_
#define LIMAP_EVALUATION_BASE_EVALUATOR_H_

#include "base/linebase.h"
#include "util/types.h"

#include <string>
#include <tuple>

namespace limap {

namespace evaluation {

class BaseEvaluator {
public:
  virtual double ComputeDistPoint(const V3D &point) = 0;
  double ComputeDistLine(const Line3d &line, int n_samples = 10);

  // compute inlier ratio
  double ComputeInlierRatio(const Line3d &line, double threshold,
                            int n_samples = 1000);

  // visualization
  std::vector<Line3d> ComputeInlierSegsOneLine(const Line3d &line,
                                               double threshold,
                                               int n_samples = 1000);
  std::vector<Line3d> ComputeInlierSegs(const std::vector<Line3d> &lines,
                                        double threshold, int n_samples = 1000);
  std::vector<Line3d> ComputeOutlierSegsOneLine(const Line3d &line,
                                                double threshold,
                                                int n_samples = 1000);
  std::vector<Line3d> ComputeOutlierSegs(const std::vector<Line3d> &lines,
                                         double threshold,
                                         int n_samples = 1000);
};

} // namespace evaluation

} // namespace limap

#endif
