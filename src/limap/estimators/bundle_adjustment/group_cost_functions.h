#pragma once

#include "limap/estimators/bundle_adjustment/group_cost_function_utils.h"
#include "limap/geometry/ceres_angle_utils.h"
#include "limap/geometry/groups/projection.h"

namespace limap {
namespace estimators {

//============================================================================
// 3D COST FUNCTORS - VP (Vanishing Point)
//============================================================================

// Line-to-VP cost using sine(angle) between line direction and VP direction
// Residual: 1 (sine of angle, 0 when parallel)
// Note: Weight should be applied via ceres::ScaledLoss, not baked into residual
struct LineToVPSineCostFunctor {
  LineToVPSineCostFunctor() = default;

  template <typename T>
  bool operator()(const T *const line_params, // [uvec(4), wvec(2)]
                  const T *const vp_params,   // [vx, vy, vz] (unit vector)
                  T *residuals) const {
    // Convert MinimalPlucker to direction vector
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    residuals[0] = AngleSine3D<T>(dvec, vp_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<LineToVPSineCostFunctor,
                                           1,  // num residuals
                                           6,  // line_params
                                           3>( // vp_params
        new LineToVPSineCostFunctor());
  }
};

// Line-to-VP cost using 1 - |cos(angle)| between line direction and VP
// direction Residual: 1 (deviation from parallel, 0 when parallel)
// Note: Weight should be applied via ceres::ScaledLoss, not baked into residual
struct LineToVPCosineCostFunctor {
  LineToVPCosineCostFunctor() = default;

  template <typename T>
  bool operator()(const T *const line_params, // [uvec(4), wvec(2)]
                  const T *const vp_params,   // [vx, vy, vz] (unit vector)
                  T *residuals) const {
    // Convert MinimalPlucker to direction vector
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    residuals[0] = T(1) - AngleCosine3D<T>(dvec, vp_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<LineToVPCosineCostFunctor,
                                           1,  // num residuals
                                           6,  // line_params
                                           3>( // vp_params
        new LineToVPCosineCostFunctor());
  }
};

// VP-VP orthogonality cost: penalizes deviation from 90° between two VPs
// Residual: 1D |cos(angle)| — equals 0 when orthogonal, 1 when parallel
// Both VP params are unit vectors with SphereManifold<3>
struct VPOrthogonalityCostFunctor {
  template <typename T>
  bool operator()(const T *const vp1_params, const T *const vp2_params,
                  T *residuals) const {
    residuals[0] = AngleCosine3D<T>(vp1_params, vp2_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<VPOrthogonalityCostFunctor, 1, 3, 3>(
        new VPOrthogonalityCostFunctor());
  }
};

// Plane-Plane normal orthogonality cost: penalizes deviation from 90° between
// two plane normals. Residual: 1D |cos(angle)| — equals 0 when orthogonal.
// Plane params are 4D (a, b, c, d); AngleCosine3D reads only the first 3
// elements (the normal), so gradient for d is automatically zero.
struct PlaneNormalOrthogonalityCostFunctor {
  template <typename T>
  bool operator()(const T *const p1_params, const T *const p2_params,
                  T *residuals) const {
    residuals[0] = AngleCosine3D<T>(p1_params, p2_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<PlaneNormalOrthogonalityCostFunctor,
                                           1, 4, 4>(
        new PlaneNormalOrthogonalityCostFunctor());
  }
};

// VP-VP parallelism cost: penalizes deviation from 0° between two VPs
// Residual: 1D |sin(angle)| — equals 0 when parallel, 1 when orthogonal
// Both VP params are unit vectors with SphereManifold<3>
struct VPParallelismCostFunctor {
  template <typename T>
  bool operator()(const T *const vp1_params, const T *const vp2_params,
                  T *residuals) const {
    residuals[0] = AngleSine3D<T>(vp1_params, vp2_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<VPParallelismCostFunctor, 1, 3, 3>(
        new VPParallelismCostFunctor());
  }
};

// Plane-Plane normal parallelism cost: penalizes deviation from 0° between
// two plane normals. Residual: 1D |sin(angle)| — equals 0 when parallel.
// Plane params are 4D (a, b, c, d); AngleSine3D reads only the first 3
// elements (the normal), so gradient for d is automatically zero.
struct PlaneNormalParallelismCostFunctor {
  template <typename T>
  bool operator()(const T *const p1_params, const T *const p2_params,
                  T *residuals) const {
    residuals[0] = AngleSine3D<T>(p1_params, p2_params);
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<PlaneNormalParallelismCostFunctor, 1,
                                           4, 4>(
        new PlaneNormalParallelismCostFunctor());
  }
};

//============================================================================
// TYPE ALIASES FOR PLANE
//============================================================================

// 3D cost
using PointToPlane3DCostFunctor = PointToSurface3DCostFunctor<PlaneProjection>;

// 2D point costs
template <typename CameraModel>
using PointToPlane2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using PointToPlane2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using PointToPlane2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using PointToPlane2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, PlaneProjection>;

// 2D line costs
template <typename CameraModel>
using LineToPlane2DCostFunctor =
    LineToSurface2DCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using LineToPlane2DConstantPoseCostFunctor =
    LineToSurface2DConstantPoseCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using LineToPlane2DRigCostFunctor =
    LineToSurface2DRigCostFunctor<CameraModel, PlaneProjection>;

template <typename CameraModel>
using LineToPlane2DConstantRigCostFunctor =
    LineToSurface2DConstantRigCostFunctor<CameraModel, PlaneProjection>;

//============================================================================
// TYPE ALIASES FOR SPHERE
// Note: Line costs not supported for spheres (kSupportsLineProjection = false)
//============================================================================

// 3D cost
using PointToSphere3DCostFunctor =
    PointToSurface3DCostFunctor<SphereProjection>;

// 2D point costs
template <typename CameraModel>
using PointToSphere2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, SphereProjection>;

template <typename CameraModel>
using PointToSphere2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, SphereProjection>;

template <typename CameraModel>
using PointToSphere2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, SphereProjection>;

template <typename CameraModel>
using PointToSphere2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, SphereProjection>;

//============================================================================
// TYPE ALIASES FOR CYLINDER
// Note: Line costs not supported for cylinders
//============================================================================

// 3D cost
using PointToCylinder3DCostFunctor =
    PointToSurface3DCostFunctor<CylinderProjection>;

// 2D point costs
template <typename CameraModel>
using PointToCylinder2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, CylinderProjection>;

template <typename CameraModel>
using PointToCylinder2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, CylinderProjection>;

template <typename CameraModel>
using PointToCylinder2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, CylinderProjection>;

template <typename CameraModel>
using PointToCylinder2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, CylinderProjection>;

//============================================================================
// TYPE ALIASES FOR ELLIPSOID
// Note: Line costs not supported for ellipsoids
//============================================================================

// 3D cost
using PointToEllipsoid3DCostFunctor =
    PointToSurface3DCostFunctor<EllipsoidProjection>;

// 2D point costs
template <typename CameraModel>
using PointToEllipsoid2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, EllipsoidProjection>;

template <typename CameraModel>
using PointToEllipsoid2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, EllipsoidProjection>;

template <typename CameraModel>
using PointToEllipsoid2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, EllipsoidProjection>;

template <typename CameraModel>
using PointToEllipsoid2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, EllipsoidProjection>;

//============================================================================
// TYPE ALIASES FOR CUBOID
//============================================================================

// 3D cost
using PointToCuboid3DCostFunctor =
    PointToSurface3DCostFunctor<CuboidProjection>;

// 2D point costs
template <typename CameraModel>
using PointToCuboid2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using PointToCuboid2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using PointToCuboid2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using PointToCuboid2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, CuboidProjection>;

// 2D line costs
template <typename CameraModel>
using LineToCuboid2DCostFunctor =
    LineToSurface2DCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using LineToCuboid2DConstantPoseCostFunctor =
    LineToSurface2DConstantPoseCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using LineToCuboid2DRigCostFunctor =
    LineToSurface2DRigCostFunctor<CameraModel, CuboidProjection>;

template <typename CameraModel>
using LineToCuboid2DConstantRigCostFunctor =
    LineToSurface2DConstantRigCostFunctor<CameraModel, CuboidProjection>;

//============================================================================
// TYPE ALIASES FOR CONE
// Note: Line costs not supported for cones
//============================================================================

// 3D cost
using PointToCone3DCostFunctor = PointToSurface3DCostFunctor<ConeProjection>;

// 2D point costs
template <typename CameraModel>
using PointToCone2DCostFunctor =
    PointToSurface2DCostFunctor<CameraModel, ConeProjection>;

template <typename CameraModel>
using PointToCone2DConstantPoseCostFunctor =
    PointToSurface2DConstantPoseCostFunctor<CameraModel, ConeProjection>;

template <typename CameraModel>
using PointToCone2DRigCostFunctor =
    PointToSurface2DRigCostFunctor<CameraModel, ConeProjection>;

template <typename CameraModel>
using PointToCone2DConstantRigCostFunctor =
    PointToSurface2DConstantRigCostFunctor<CameraModel, ConeProjection>;

} // namespace estimators
} // namespace limap
