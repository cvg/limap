#pragma once

#include "limap/base/linebase.h"
#include "limap/evaluation/base_evaluator.h"
#include "limap/util/kd_tree.h"
#include "limap/util/types.h"

#include <string>
#include <tuple>

namespace limap {

namespace evaluation {

class PointCloudEvaluator : public BaseEvaluator {
public:
  PointCloudEvaluator() : BaseEvaluator() {}
  PointCloudEvaluator(const std::vector<V3D> &points) : BaseEvaluator() {
    tree_.initialize(points, false);
  }
  PointCloudEvaluator(const Eigen::MatrixXd &points) : BaseEvaluator() {
    tree_.initialize(points, false);
  }

  // build indexes
  void Build() { tree_.buildIndex(); }

  // IO
  void Save(const std::string &filename) { tree_.save(filename); }
  void Load(const std::string &filename) { tree_.load(filename); }

  // compute dist point
  double ComputeDistPoint(const V3D &point) override;

  // inverse recall computed on the reference point cloud
  std::vector<double>
  ComputeDistsforEachPoint(const std::vector<Line3d> &lines) const;
  // approximation
  std::vector<double>
  ComputeDistsforEachPoint_KDTree(const std::vector<Line3d> &lines) const;

private:
  KDTree tree_;
};

} // namespace evaluation

} // namespace limap
