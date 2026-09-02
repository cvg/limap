#include "limap/geometry/groups/vp.h"

#include <ceres/manifold.h>
#include <colmap/util/logging.h>

#include <cmath>
#include <stdexcept>

#include "limap/estimators/bundle_adjustment/analytical_group_cost_functions.h"
#include "limap/estimators/bundle_adjustment/group_cost_functions.h"

namespace limap {

std::vector<double> VPGroup::GetDefaultParams2D() const {
  // Default 2D: homogeneous vanishing point at infinity (vertical direction)
  return {0.0, 1.0, 0.0};
}

std::vector<double> VPGroup::GetDefaultParams3D() const {
  // Default 3D: unit Z direction
  return {0.0, 0.0, 1.0};
}

void VPGroup::NormalizeParams3D(double *params) const {
  double norm = std::sqrt(params[0] * params[0] + params[1] * params[1] +
                          params[2] * params[2]);
  if (norm < 1e-12) {
    throw std::runtime_error("VP params have near-zero norm");
  }
  params[0] /= norm;
  params[1] /= norm;
  params[2] /= norm;
}

bool VPGroup::CheckParams3D(const double *params, double tol) const {
  double norm = std::sqrt(params[0] * params[0] + params[1] * params[1] +
                          params[2] * params[2]);
  return std::abs(norm - 1.0) < tol;
}

ceres::CostFunction *VPGroup::CreateLineCost3D() const {
  if (residual_type_ == ResidualType::SINE) {
    return new estimators::AnalyticalLineToVPSineCostFunction();
  } else {
    return new estimators::AnalyticalLineToVPCosineCostFunction();
  }
}

ceres::Manifold *VPGroup::CreateManifold3D() const {
  return new ceres::SphereManifold<3>();
}

} // namespace limap
