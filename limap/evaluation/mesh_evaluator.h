#pragma once

#include "base/linebase.h"
#include "evaluation/base_evaluator.h"
#include "util/types.h"

#include <igl/AABB.h>
#include <string>
#include <tuple>

namespace limap {

namespace evaluation {

class MeshEvaluator : public BaseEvaluator {
public:
  MeshEvaluator() : BaseEvaluator() {}
  MeshEvaluator(const std::string &filename, const double &mpau);

  // compute dist point
  double ComputeDistPoint(const V3D &point) override;

private:
  Eigen::MatrixXd V_;
  Eigen::MatrixXi F_;
  igl::AABB<Eigen::MatrixXd, 3> tree_;
};

} // namespace evaluation

} // namespace limap
