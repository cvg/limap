#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Cone (double cone / infinite conical surface): 7 params [apex(3), dir(3),
// log_tan_alpha(1)]
//
// Double cone: both sides of apex are valid surface.
// Surface equation: for point P, the angle between (P - apex) and dir
//   equals alpha, where tan(alpha) = exp(log_tan_alpha).
//
// Params layout:
//   - apex[0:3]: Cone tip position (3 DOF)
//   - direction[3:6]: Unit vector for axis direction (2 DOF, SphereManifold<3>)
//   - log_tan_alpha[6]: Log of tangent of half-angle (1 DOF, unconstrained)
//
// Manifold: ProductManifold<EuclideanManifold<3>, SphereManifold<3>,
//                           EuclideanManifold<1>>
// DOF: 3 (apex) + 2 (direction) + 1 (half-angle) = 6
class ConeGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 0; }      // not supported
  size_t GetNumParamsIn3D() const override { return 7; }

  std::vector<double> GetDefaultParams2D() const override; // not supported
  std::vector<double> GetDefaultParams3D() const override; // 45° cone at origin

  void NormalizeParams3D(double *params) const override;
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D costs: project point -> cone -> camera -> 2D (line costs N/A)
  ceres::CostFunction *
  CreatePointCost2D(colmap::CameraModelId camera_model_id) const override;
  ceres::CostFunction *CreatePointCost2DConstantPose(
      colmap::CameraModelId camera_model_id,
      const colmap::Rigid3d &cam_from_world) const override;
  ceres::CostFunction *
  CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const override;
  ceres::CostFunction *CreatePointCost2DConstantRig(
      colmap::CameraModelId camera_model_id,
      const colmap::Rigid3d &cam_from_rig) const override;
};

} // namespace limap
