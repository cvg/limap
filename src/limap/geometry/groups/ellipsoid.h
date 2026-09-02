#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Ellipsoid: 10 continuous params [quat(4), center(3), log_scales(3)]
// where the ellipsoid is defined as:
//   (x - c)^T R^T D R (x - c) = 1
// with D = diag(1/sx^2, 1/sy^2, 1/sz^2), si = exp(log_si)
//
// Params layout:
//   - quat[0:4]: Quaternion (x, y, z, w) encoding rotation R
//   - center[4:7]: Center point (cx, cy, cz)
//   - log_scales[7:10]: Log semi-axes (log_sx, log_sy, log_sz)
//
// Manifold: ProductManifold<EigenQuaternionManifold, EuclideanManifold<3>,
//                           EuclideanManifold<3>>
// DOF: 3 (rotation) + 3 (center) + 3 (scales) = 9
//
// Special cases:
//   - sx = sy = sz: sphere
//   - sx = sy != sz: spheroid (oblate or prolate)
class EllipsoidGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 0; }      // not supported
  size_t GetNumParamsIn3D() const override { return 10; }

  std::vector<double> GetDefaultParams2D() const override; // not supported
  std::vector<double> GetDefaultParams3D() const override; // unit sphere

  void NormalizeParams3D(double *params) const override;
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D costs: project point -> ellipsoid -> camera -> 2D (line costs N/A)
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
