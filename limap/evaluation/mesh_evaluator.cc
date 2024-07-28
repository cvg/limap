#include "evaluation/mesh_evaluator.h"

#include <cmath>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <iostream>
#include <numeric>
#include <queue>

namespace limap {

namespace evaluation {

MeshEvaluator::MeshEvaluator(const std::string &filename, const double &mpau)
    : BaseEvaluator() {
  std::string extension = filename.substr(filename.rfind('.') + 1);
  if (extension == std::string("off"))
    igl::readOFF(filename, V_, F_);
  else if (extension == std::string("obj"))
    igl::readOBJ(filename, V_, F_);
  else
    throw std::runtime_error("Not Implemented!!");
  std::cout << "read a new mesh: V.rows() = " << V_.rows()
            << ", F.rows() = " << F_.rows() << std::endl;
  V_ *= mpau;
  tree_.init(V_, F_);
}

double MeshEvaluator::ComputeDistPoint(const V3D &point) {
  Eigen::VectorXd sqrD;
  Eigen::VectorXi I;
  Eigen::MatrixXd C;
  Eigen::MatrixXd input(1, 3);
  input.row(0) = point;
  tree_.squared_distance(V_, F_, input, sqrD, I, C);
  double dist = sqrt(sqrD(0));
  return dist;
}

} // namespace evaluation

} // namespace limap
