#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Plane: [a,b,c,d] with ax+by+cz+d=0, ||(a,b,c)||=1, d=signed distance to
// origin. Manifold: SphereManifold<3> x EuclideanManifold<1>
class PlaneGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 3; }
  size_t GetNumParamsIn3D() const override { return 4; }

  std::vector<double> GetDefaultParams2D() const override; // (0,1,0) horiz line
  std::vector<double> GetDefaultParams3D() const override; // (0,0,1,0) XY plane

  void NormalizeParams3D(double *params) const override; // ||(a,b,c)||=1
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;
  Line3d ProjectLine(const Line3d &line, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D costs: project point/line → plane → camera → 2D
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

  ceres::CostFunction *
  CreateLineCost2D(colmap::CameraModelId camera_model_id,
                   const Line2d &observed_2d) const override;
  ceres::CostFunction *CreateLineCost2DConstantPose(
      colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
      const colmap::Rigid3d &cam_from_world) const override;
  ceres::CostFunction *
  CreateLineCost2DRig(colmap::CameraModelId camera_model_id,
                      const Line2d &observed_2d) const override;
  ceres::CostFunction *CreateLineCost2DConstantRig(
      colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
      const colmap::Rigid3d &cam_from_rig) const override;
};

} // namespace limap
