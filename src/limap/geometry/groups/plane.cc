#include "limap/geometry/groups/plane.h"

#include <ceres/manifold.h>
#include <colmap/util/logging.h>

#include <cmath>
#include <stdexcept>

#include "limap/estimators/bundle_adjustment/analytical_group_cost_functions.h"
#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"

namespace limap {

std::vector<double> PlaneGroup::GetDefaultParams2D() const {
  // Default 2D: homogeneous line (horizontal)
  return {0.0, 1.0, 0.0};
}

std::vector<double> PlaneGroup::GetDefaultParams3D() const {
  // Default 3D: XY plane at origin (normal pointing up)
  return {0.0, 0.0, 1.0, 0.0};
}

void PlaneGroup::NormalizeParams3D(double *params) const {
  double normal_norm = std::sqrt(params[0] * params[0] + params[1] * params[1] +
                                 params[2] * params[2]);
  if (normal_norm < 1e-12) {
    throw std::runtime_error("Plane normal has near-zero norm");
  }
  params[0] /= normal_norm;
  params[1] /= normal_norm;
  params[2] /= normal_norm;
  params[3] /= normal_norm;
}

bool PlaneGroup::CheckParams3D(const double *params, double tol) const {
  double normal_norm = std::sqrt(params[0] * params[0] + params[1] * params[1] +
                                 params[2] * params[2]);
  return std::abs(normal_norm - 1.0) < tol;
}

V3D PlaneGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  PlaneProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

Line3d PlaneGroup::ProjectLine(const Line3d &line, const double *params) const {
  V3D proj_start, proj_end;
  PlaneProjection::ProjectPoint(line.start.data(), params, proj_start.data());
  PlaneProjection::ProjectPoint(line.end.data(), params, proj_end.data());
  return Line3d(proj_start, proj_end);
}

//============================================================================
// PlaneGroup 3D Cost Factory Methods
//============================================================================

// TODO: Line-to-Plane 3D cost - needs further design consideration
// ceres::CostFunction *PlaneGroup::CreateLineCost3D() const {
//   return estimators::LineToPlaneCostFunctor3D::Create();
// }

ceres::CostFunction *PlaneGroup::CreatePointCost3D() const {
  return estimators::PointToPlane3DCostFunctor::Create();
}

ceres::Manifold *PlaneGroup::CreateManifold3D() const {
  // ProductManifold: SphereManifold<3> for unit normal (a,b,c),
  // EuclideanManifold<1> for d (signed distance to origin)
  return new ceres::ProductManifold<ceres::SphereManifold<3>,
                                    ceres::EuclideanManifold<1>>();
}

//============================================================================
// PlaneGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
PlaneGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToPlane2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *PlaneGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return new estimators::AnalyticalPointToPlane2DConstantPoseCostFunction<   \
        CameraModel>(cam_from_world);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    return estimators::CreateSurfaceCameraCostFunction2D<
        estimators::PointToPlane2DConstantPoseCostFunctor>(camera_model_id,
                                                           cam_from_world);
  }
}

ceres::CostFunction *
PlaneGroup::CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToPlane2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *PlaneGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToPlane2DConstantRigCostFunctor>(camera_model_id,
                                                        cam_from_rig);
}

//============================================================================
// PlaneGroup 2D Line Cost Factory Methods
//============================================================================

ceres::CostFunction *
PlaneGroup::CreateLineCost2D(colmap::CameraModelId camera_model_id,
                             const Line2d &observed_2d) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToPlane2DCostFunctor>(camera_model_id, observed_2d);
}

ceres::CostFunction *PlaneGroup::CreateLineCost2DConstantPose(
    colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
    const colmap::Rigid3d &cam_from_world) const {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return new estimators::AnalyticalLineToPlane2DConstantPoseCostFunction<    \
        CameraModel>(observed_2d, cam_from_world);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    return estimators::CreateSurfaceCameraCostFunction2D<
        estimators::LineToPlane2DConstantPoseCostFunctor>(
        camera_model_id, observed_2d, cam_from_world);
  }
}

ceres::CostFunction *
PlaneGroup::CreateLineCost2DRig(colmap::CameraModelId camera_model_id,
                                const Line2d &observed_2d) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToPlane2DRigCostFunctor>(camera_model_id, observed_2d);
}

ceres::CostFunction *PlaneGroup::CreateLineCost2DConstantRig(
    colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToPlane2DConstantRigCostFunctor>(
      camera_model_id, observed_2d, cam_from_rig);
}

} // namespace limap
