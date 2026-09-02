#pragma once

#include "limap/geometry/groups.h"

namespace limap {

class VPGroup : public BaseGroup {
public:
  // Configuration for VP residual formulation
  enum class ResidualType { SINE, COSINE };

  // Angular error threshold (radians)
  double GetDefaultThreshold() const override { return 0.1; }

  size_t GetNumParamsIn2D() const override { return 3; }
  size_t GetNumParamsIn3D() const override { return 3; }

  // Default 2D: homogeneous vanishing point at infinity (0, 1, 0)
  std::vector<double> GetDefaultParams2D() const override;
  // Default 3D: unit Z direction (0, 0, 1)
  std::vector<double> GetDefaultParams3D() const override;

  // Normalize to unit 3-vector direction
  void NormalizeParams3D(double *params) const override;

  // Check: ||params|| == 1 (unit vector)
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  // 3D cost: line direction should be parallel to VP direction
  ceres::CostFunction *CreateLineCost3D() const override;
  int NumLineResiduals3D() const override { return 1; }

  // Manifold: SphereManifold<3> for unit direction on S²
  ceres::Manifold *CreateManifold3D() const override;

  // Configuration
  void SetResidualType(ResidualType type) { residual_type_ = type; }
  ResidualType GetResidualType() const { return residual_type_; }

private:
  ResidualType residual_type_ = ResidualType::SINE;
};

} // namespace limap
