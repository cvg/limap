#pragma once

#include "limap/geometry/groups.h"

namespace limap {

// Cuboid: 10 continuous params [quat(4), d(6)]
// Inspired by PixCuboid (ICCVW 2025).
//
// A cuboid has 6 planar faces sharing an orientation R (from quaternion).
// Lines and points on cuboid faces are constrained to lie on one of 6 planes.
// This is much stronger than 6 independent planes (9 DOF vs 18 DOF).
//
// Params layout:
//   - quat[0:4]: Quaternion (x, y, z, w) encoding rotation R
//   - d[4:10]: Signed offsets for 6 faces along ±x, ±y, ±z in cuboid frame
//
// Face planes in world frame (with R as rotation matrix):
//   Face 0 (+x): normal = R[:,0],  plane: R[:,0]^T * X = d[0]
//   Face 1 (-x): normal = -R[:,0], plane: -R[:,0]^T * X = d[1]
//   Face 2 (+y): normal = R[:,1],  plane: R[:,1]^T * X = d[2]
//   Face 3 (-y): normal = -R[:,1], plane: -R[:,1]^T * X = d[3]
//   Face 4 (+z): normal = R[:,2],  plane: R[:,2]^T * X = d[4]
//   Face 5 (-z): normal = -R[:,2], plane: -R[:,2]^T * X = d[5]
//
// Manifold: ProductManifold<EigenQuaternionManifold, EuclideanManifold<6>>
// DOF: 3 (rotation) + 6 (offsets) = 9
class CuboidGroup : public BaseGroup {
public:
  double GetDefaultThreshold() const override { return 0.1; } // world units
  size_t GetNumParamsIn2D() const override { return 0; }      // not supported
  size_t GetNumParamsIn3D() const override { return 10; }

  std::vector<double> GetDefaultParams2D() const override; // not supported
  std::vector<double> GetDefaultParams3D() const override; // unit cube

  void NormalizeParams3D(double *params) const override;
  bool CheckParams3D(const double *params, double tol = 1e-6) const override;

  V3D ProjectPoint(const V3D &point, const double *params) const override;
  Line3d ProjectLine(const Line3d &line, const double *params) const override;

  ceres::CostFunction *CreatePointCost3D() const override;
  ceres::Manifold *CreateManifold3D() const override;

  // 2D point costs: project point -> cuboid face -> camera -> 2D
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

  // 2D line costs: project line -> cuboid face -> camera -> 2D
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
