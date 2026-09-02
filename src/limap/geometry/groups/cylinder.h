#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Infinite Cylinder: 7 continuous params [quat(4), wvec(2), log_r(1)]
// where quat and wvec encode the axis using MinimalInfiniteLine3d
// representation (SO(3) x SO(2) for the axis line, plus log-radius for the
// cylinder radius).
//
// Manifold: ProductManifold<EigenQuaternionManifold, SphereManifold<2>,
//                           EuclideanManifold<1>>
// DOF: 3 + 1 + 1 = 5
//
// The axis is represented using Plücker coordinates via MinimalInfiniteLine3d:
// - Quaternion encodes an orthonormal frame where col(0) = direction d
// - wvec encodes the moment magnitude ||m|| via wvec[1]/wvec[0]
// - log_r encodes the cylinder radius r = exp(log_r)
class CylinderGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 0; }      // not supported
  size_t GetNumParamsIn3D() const override { return 7; }

  std::vector<double> GetDefaultParams2D() const override; // not supported
  std::vector<double> GetDefaultParams3D() const override; // unit cylinder

  void NormalizeParams3D(double *params) const override;
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D costs: project point -> cylinder -> camera -> 2D (line costs N/A)
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
