#pragma once

#include <ceres/ceres.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>

#include "limap/estimators/bundle_adjustment/cost_functions.h"
#include "limap/geometry/camera_models.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/line2d.h"

namespace limap {
namespace estimators {

//============================================================================
// GENERIC SURFACE COST FUNCTORS
// These work for any surface type (Plane, Sphere, etc.) that provides:
//   - SurfaceProjection::kNumParams (number of parameters)
//   - SurfaceProjection::kSupportsLineProjection (bool)
//   - SurfaceProjection::ProjectPoint(point3D, params, projected)
//   - SurfaceProjection::ProjectLine(dvec, mvec, params, proj_dvec, proj_mvec)
//============================================================================

//----------------------------------------------------------------------------
// 3D Point-to-Surface cost
// Residual: 3 (3D displacement from point to its projection on surface)
// Note: Weight should be applied via ceres::ScaledLoss, not baked into residual
//----------------------------------------------------------------------------
template <typename SurfaceProjection> struct PointToSurface3DCostFunctor {
  PointToSurface3DCostFunctor() = default;

  template <typename T>
  bool operator()(const T *const point3D, const T *const surface_params,
                  T *residuals) const {
    T projected[3];
    SurfaceProjection::ProjectPoint(point3D, surface_params, projected);
    residuals[0] = point3D[0] - projected[0];
    residuals[1] = point3D[1] - projected[1];
    residuals[2] = point3D[2] - projected[2];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<
        PointToSurface3DCostFunctor<SurfaceProjection>, 3, 3,
        SurfaceProjection::kNumParams>(new PointToSurface3DCostFunctor());
  }
};

//----------------------------------------------------------------------------
// 2D Point-to-Surface costs (4 pose variants)
// Note: Weight should be applied via ceres::ScaledLoss, not baked into residual
//----------------------------------------------------------------------------

// Variable pose: params (point3D, cam_from_world, surface_params,
// camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct PointToSurface2DCostFunctor {
  PointToSurface2DCostFunctor() = default;

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_world,
                  const T *const surface_params, const T *const camera_params,
                  T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;

    T xy_orig[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, point3D,
                                     xy_orig);

    T projected[3];
    SurfaceProjection::ProjectPoint(point3D, surface_params, projected);
    T xy_proj[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, projected,
                                     xy_proj);

    residuals[0] = xy_orig[0] - xy_proj[0];
    residuals[1] = xy_orig[1] - xy_proj[1];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<
        PointToSurface2DCostFunctor<CameraModel, SurfaceProjection>, 2, 3, 7,
        SurfaceProjection::kNumParams, CameraModel::num_params>(
        new PointToSurface2DCostFunctor());
  }
};

// Constant pose: params (point3D, surface_params, camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct PointToSurface2DConstantPoseCostFunctor {
  explicit PointToSurface2DConstantPoseCostFunctor(
      const colmap::Rigid3d &cam_from_world)
      : cam_from_world_(cam_from_world) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const surface_params,
                  const T *const camera_params, T *residuals) const {
    const Eigen::Quaterniond &q = cam_from_world_.rotation();
    const Eigen::Vector3d &t = cam_from_world_.translation();
    T qvec[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
    T tvec[3] = {T(t.x()), T(t.y()), T(t.z())};

    T xy_orig[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, point3D,
                                     xy_orig);

    T projected[3];
    SurfaceProjection::ProjectPoint(point3D, surface_params, projected);
    T xy_proj[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, projected,
                                     xy_proj);

    residuals[0] = xy_orig[0] - xy_proj[0];
    residuals[1] = xy_orig[1] - xy_proj[1];
    return true;
  }

  static ceres::CostFunction *Create(const colmap::Rigid3d &cam_from_world) {
    return new ceres::AutoDiffCostFunction<
        PointToSurface2DConstantPoseCostFunctor<CameraModel, SurfaceProjection>,
        2, 3, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new PointToSurface2DConstantPoseCostFunctor(cam_from_world));
  }

private:
  const colmap::Rigid3d cam_from_world_;
};

// Rig: params (point3D, cam_from_rig, rig_from_world, surface_params,
//              camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct PointToSurface2DRigCostFunctor {
  PointToSurface2DRigCostFunctor() = default;

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_rig,
                  const T *const rig_from_world, const T *const surface_params,
                  const T *const camera_params, T *residuals) const {
    const T *cam_from_rig_rot = cam_from_rig;
    const T *cam_from_rig_trans = cam_from_rig + 4;
    const T *rig_from_world_rot = rig_from_world;
    const T *rig_from_world_trans = rig_from_world + 4;

    // Compose transforms to get cam_from_world
    Eigen::Map<const Eigen::Quaternion<T>> q_rig(rig_from_world_rot);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_rig(rig_from_world_trans);
    Eigen::Map<const Eigen::Quaternion<T>> q_cam(cam_from_rig_rot);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam(cam_from_rig_trans);

    Eigen::Quaternion<T> q_composed = q_cam * q_rig;
    Eigen::Matrix<T, 3, 1> t_composed = q_cam * t_rig + t_cam;

    T qvec[4] = {q_composed.x(), q_composed.y(), q_composed.z(),
                 q_composed.w()};
    T tvec[3] = {t_composed[0], t_composed[1], t_composed[2]};

    T xy_orig[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, point3D,
                                     xy_orig);

    T projected[3];
    SurfaceProjection::ProjectPoint(point3D, surface_params, projected);
    T xy_proj[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, projected,
                                     xy_proj);

    residuals[0] = xy_orig[0] - xy_proj[0];
    residuals[1] = xy_orig[1] - xy_proj[1];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<
        PointToSurface2DRigCostFunctor<CameraModel, SurfaceProjection>, 2, 3, 7,
        7, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new PointToSurface2DRigCostFunctor());
  }
};

// Constant rig: params (point3D, rig_from_world, surface_params,
//                       camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct PointToSurface2DConstantRigCostFunctor {
  explicit PointToSurface2DConstantRigCostFunctor(
      const colmap::Rigid3d &cam_from_rig)
      : cam_from_rig_(cam_from_rig) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const rig_from_world,
                  const T *const surface_params, const T *const camera_params,
                  T *residuals) const {
    const T *rig_from_world_rot = rig_from_world;
    const T *rig_from_world_trans = rig_from_world + 4;

    // Convert constant cam_from_rig to T
    const Eigen::Quaterniond &q_cfr = cam_from_rig_.rotation();
    const Eigen::Vector3d &t_cfr = cam_from_rig_.translation();
    T cam_from_rig_rot[4] = {T(q_cfr.x()), T(q_cfr.y()), T(q_cfr.z()),
                             T(q_cfr.w())};
    T cam_from_rig_trans[3] = {T(t_cfr.x()), T(t_cfr.y()), T(t_cfr.z())};

    // Compose transforms to get cam_from_world
    Eigen::Map<const Eigen::Quaternion<T>> q_rig(rig_from_world_rot);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_rig(rig_from_world_trans);
    Eigen::Map<const Eigen::Quaternion<T>> q_cam(cam_from_rig_rot);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam(cam_from_rig_trans);

    Eigen::Quaternion<T> q_composed = q_cam * q_rig;
    Eigen::Matrix<T, 3, 1> t_composed = q_cam * t_rig + t_cam;

    T qvec[4] = {q_composed.x(), q_composed.y(), q_composed.z(),
                 q_composed.w()};
    T tvec[3] = {t_composed[0], t_composed[1], t_composed[2]};

    T xy_orig[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, point3D,
                                     xy_orig);

    T projected[3];
    SurfaceProjection::ProjectPoint(point3D, surface_params, projected);
    T xy_proj[2];
    ProjectPointToPixel<CameraModel>(camera_params, qvec, tvec, projected,
                                     xy_proj);

    residuals[0] = xy_orig[0] - xy_proj[0];
    residuals[1] = xy_orig[1] - xy_proj[1];
    return true;
  }

  static ceres::CostFunction *Create(const colmap::Rigid3d &cam_from_rig) {
    return new ceres::AutoDiffCostFunction<
        PointToSurface2DConstantRigCostFunctor<CameraModel, SurfaceProjection>,
        2, 3, 7, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new PointToSurface2DConstantRigCostFunctor(cam_from_rig));
  }

private:
  const colmap::Rigid3d cam_from_rig_;
};

//----------------------------------------------------------------------------
// 2D Line-to-Surface costs (4 pose variants)
//----------------------------------------------------------------------------

// Variable pose: params (line_params, cam_from_world, surface_params,
// camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct LineToSurface2DCostFunctor {
  explicit LineToSurface2DCostFunctor(const Line2d &observed)
      : observed_(observed) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_world,
                  const T *const surface_params, const T *const camera_params,
                  T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;

    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    T r_orig[2];
    ComputeLineReprojectionCost<CameraModel>(camera_params, qvec, tvec, dvec,
                                             mvec, observed_, r_orig);

    T proj_dvec[3], proj_mvec[3];
    SurfaceProjection::ProjectLine(dvec, mvec, surface_params, proj_dvec,
                                   proj_mvec);
    T r_surf[2];
    ComputeLineReprojectionCost<CameraModel>(
        camera_params, qvec, tvec, proj_dvec, proj_mvec, observed_, r_surf);

    residuals[0] = r_orig[0] - r_surf[0];
    residuals[1] = r_orig[1] - r_surf[1];
    return true;
  }

  static ceres::CostFunction *Create(const Line2d &observed) {
    return new ceres::AutoDiffCostFunction<
        LineToSurface2DCostFunctor<CameraModel, SurfaceProjection>, 2, 6, 7,
        SurfaceProjection::kNumParams, CameraModel::num_params>(
        new LineToSurface2DCostFunctor(observed));
  }

private:
  const Line2d observed_;
};

// Constant pose: params (line_params, surface_params, camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct LineToSurface2DConstantPoseCostFunctor {
  LineToSurface2DConstantPoseCostFunctor(const Line2d &observed,
                                         const colmap::Rigid3d &cam_from_world)
      : observed_(observed), cam_from_world_(cam_from_world) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const surface_params,
                  const T *const camera_params, T *residuals) const {
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    T r_orig[2];
    ComputeLineReprojectionCostConstantPose<CameraModel>(
        camera_params, cam_from_world_, dvec, mvec, observed_, r_orig);

    T proj_dvec[3], proj_mvec[3];
    SurfaceProjection::ProjectLine(dvec, mvec, surface_params, proj_dvec,
                                   proj_mvec);
    T r_surf[2];
    ComputeLineReprojectionCostConstantPose<CameraModel>(
        camera_params, cam_from_world_, proj_dvec, proj_mvec, observed_,
        r_surf);

    residuals[0] = r_orig[0] - r_surf[0];
    residuals[1] = r_orig[1] - r_surf[1];
    return true;
  }

  static ceres::CostFunction *Create(const Line2d &observed,
                                     const colmap::Rigid3d &cam_from_world) {
    return new ceres::AutoDiffCostFunction<
        LineToSurface2DConstantPoseCostFunctor<CameraModel, SurfaceProjection>,
        2, 6, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new LineToSurface2DConstantPoseCostFunctor(observed, cam_from_world));
  }

private:
  const Line2d observed_;
  const colmap::Rigid3d cam_from_world_;
};

// Rig: params (line_params, cam_from_rig, rig_from_world, surface_params,
//              camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct LineToSurface2DRigCostFunctor {
  explicit LineToSurface2DRigCostFunctor(const Line2d &observed)
      : observed_(observed) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_rig,
                  const T *const rig_from_world, const T *const surface_params,
                  const T *const camera_params, T *residuals) const {
    const T *cam_from_rig_rot = cam_from_rig;
    const T *cam_from_rig_trans = cam_from_rig + 4;
    const T *rig_from_world_rot = rig_from_world;
    const T *rig_from_world_trans = rig_from_world + 4;

    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    T r_orig[2];
    ComputeLineReprojectionCostRig<CameraModel>(
        cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
        rig_from_world_trans, dvec, mvec, camera_params, observed_, r_orig);

    T proj_dvec[3], proj_mvec[3];
    SurfaceProjection::ProjectLine(dvec, mvec, surface_params, proj_dvec,
                                   proj_mvec);
    T r_surf[2];
    ComputeLineReprojectionCostRig<CameraModel>(
        cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
        rig_from_world_trans, proj_dvec, proj_mvec, camera_params, observed_,
        r_surf);

    residuals[0] = r_orig[0] - r_surf[0];
    residuals[1] = r_orig[1] - r_surf[1];
    return true;
  }

  static ceres::CostFunction *Create(const Line2d &observed) {
    return new ceres::AutoDiffCostFunction<
        LineToSurface2DRigCostFunctor<CameraModel, SurfaceProjection>, 2, 6, 7,
        7, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new LineToSurface2DRigCostFunctor(observed));
  }

private:
  const Line2d observed_;
};

// Constant rig: params (line_params, rig_from_world, surface_params,
//                       camera_params)
template <typename CameraModel, typename SurfaceProjection>
struct LineToSurface2DConstantRigCostFunctor {
  LineToSurface2DConstantRigCostFunctor(const Line2d &observed,
                                        const colmap::Rigid3d &cam_from_rig)
      : observed_(observed), cam_from_rig_(cam_from_rig) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const rig_from_world,
                  const T *const surface_params, const T *const camera_params,
                  T *residuals) const {
    const T *rig_from_world_rot = rig_from_world;
    const T *rig_from_world_trans = rig_from_world + 4;

    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

    T r_orig[2];
    ComputeLineReprojectionCostConstantRig<CameraModel>(
        cam_from_rig_, rig_from_world_rot, rig_from_world_trans, dvec, mvec,
        camera_params, observed_, r_orig);

    T proj_dvec[3], proj_mvec[3];
    SurfaceProjection::ProjectLine(dvec, mvec, surface_params, proj_dvec,
                                   proj_mvec);
    T r_surf[2];
    ComputeLineReprojectionCostConstantRig<CameraModel>(
        cam_from_rig_, rig_from_world_rot, rig_from_world_trans, proj_dvec,
        proj_mvec, camera_params, observed_, r_surf);

    residuals[0] = r_orig[0] - r_surf[0];
    residuals[1] = r_orig[1] - r_surf[1];
    return true;
  }

  static ceres::CostFunction *Create(const Line2d &observed,
                                     const colmap::Rigid3d &cam_from_rig) {
    return new ceres::AutoDiffCostFunction<
        LineToSurface2DConstantRigCostFunctor<CameraModel, SurfaceProjection>,
        2, 6, 7, SurfaceProjection::kNumParams, CameraModel::num_params>(
        new LineToSurface2DConstantRigCostFunctor(observed, cam_from_rig));
  }

private:
  const Line2d observed_;
  const colmap::Rigid3d cam_from_rig_;
};

//============================================================================
// GENERIC FACTORY FUNCTION for 2D costs (dispatch on camera model)
//============================================================================

template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction *
CreateSurfaceCameraCostFunction2D(const colmap::CameraModelId camera_model_id,
                                  Args &&...args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error(
        "Camera model not supported for surface projection");
  }
}

} // namespace estimators
} // namespace limap
