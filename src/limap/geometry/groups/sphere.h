#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Sphere: [cx,cy,cz,log_r] center + log-radius (2D: [cx,cy,log_r]).
// Manifold: EuclideanManifold<4> (3D), EuclideanManifold<3> (2D)
class SphereGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 3; }
  size_t GetNumParamsIn3D() const override { return 4; }

  std::vector<double> GetDefaultParams2D() const override; // (0,0,0) unit circ
  std::vector<double> GetDefaultParams3D() const override; // (0,0,0,0) unit sph

  void NormalizeParams3D(double *params) const override; // no-op
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D costs: project point → sphere → camera → 2D (line costs N/A)
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
