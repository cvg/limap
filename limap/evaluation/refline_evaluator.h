#ifndef LIMAP_EVALUATION_REFLINE_EVALUATOR_H_
#define LIMAP_EVALUATION_REFLINE_EVALUATOR_H_

#include "base/linebase.h"
#include "util/types.h"

#include <tuple>

namespace limap {

namespace evaluation {

class RefLineEvaluator {
public:
  RefLineEvaluator() {}
  RefLineEvaluator(const std::vector<Line3d> &ref_lines)
      : ref_lines_(ref_lines) {};

  double SumLength() const;
  double ComputeRecallRef(const std::vector<Line3d> &lines,
                          const double threshold,
                          const int num_samples = 1000) const;
  double ComputeRecallTested(const std::vector<Line3d> &lines,
                             const double threshold,
                             const int num_samples = 1000) const;

private:
  std::vector<Line3d> ref_lines_;
  double DistPointLine(const V3D &point, const Line3d &line) const;
  double DistPointLines(const V3D &point,
                        const std::vector<Line3d> &line) const;

  double ComputeRecallLength(const std::vector<Line3d> &ref_lines,
                             const std::vector<Line3d> &lines,
                             const double threshold,
                             const int num_samples = 1000) const;
};

} // namespace evaluation

} // namespace limap

#endif
