#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <colmap/estimators/cost_functions/reprojection_error.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>

#include "limap/estimators/bundle_adjustment/analytical_line_cost_functions.h"
#include "limap/geometry/camera_models.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/line2d.h"
#include "limap/util/eigen_types.h"

namespace limap {
namespace estimators {

//============================================================================
// STATIC PROJECTION HELPERS
//============================================================================

// Project a 3D point to 2D pixel coordinates (no observed comparison).
// qvec is in Eigen xyzw format (from Eigen::Quaternion::coeffs()).
template <typename CameraModel, typename T>
inline void ProjectPointToPixel(const T *camera_params, const T *qvec,
                                const T *tvec, const T *point3D, T *xy) {
  Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(point3D);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);

  Eigen::Matrix<T, 3, 1> point_cam = q * p + t;

  CameraModel::ImgFromCam(camera_params, point_cam[0], point_cam[1],
                          point_cam[2], &xy[0], &xy[1]);
}

//============================================================================
// STATIC REPROJECTION COST FUNCTIONS
// These can be called directly from within cost functors to compute residuals.
//============================================================================

// Point reprojection cost with variable pose
// Computes: world → camera → 2D, returns residual vs observed
// Note: qvec is in Eigen xyzw format (from Eigen::Quaternion::coeffs())
template <typename CameraModel, typename T>
inline bool ComputePointReprojectionCost(const T *camera_params, const T *qvec,
                                         const T *tvec, const T *point3D,
                                         const Eigen::Vector2d &observed,
                                         double weight, T *residuals) {
  // Transform to camera frame using Eigen operations (qvec is in xyzw format)
  Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(point3D);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);

  Eigen::Matrix<T, 3, 1> point_cam = q * p + t;

  // Project to 2D
  T xy[2];
  CameraModel::ImgFromCam(camera_params, point_cam[0], point_cam[1],
                          point_cam[2], &xy[0], &xy[1]);
  residuals[0] = T(weight) * (xy[0] - T(observed.x()));
  residuals[1] = T(weight) * (xy[1] - T(observed.y()));
  return true;
}

// Point reprojection cost with constant pose
template <typename CameraModel, typename T>
inline bool ComputePointReprojectionCostConstantPose(
    const T *camera_params, const colmap::Rigid3d &cam_from_world,
    const T *point3D, const Eigen::Vector2d &observed, double weight,
    T *residuals) {
  const Eigen::Quaterniond &q = cam_from_world.rotation();
  const Eigen::Vector3d &t = cam_from_world.translation();
  // Use xyzw format (Eigen's coeffs() order)
  T qvec[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T tvec[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputePointReprojectionCost<CameraModel>(
      camera_params, qvec, tvec, point3D, observed, weight, residuals);
}

// Point reprojection cost with rig (variable cam_from_rig and rig_from_world)
// Note: quaternions are in Eigen xyzw format (from Eigen::Quaternion::coeffs())
template <typename CameraModel, typename T>
inline bool ComputePointReprojectionCostRig(
    const T *cam_from_rig_rot, const T *cam_from_rig_trans,
    const T *rig_from_world_rot, const T *rig_from_world_trans,
    const T *point3D, const T *camera_params, const Eigen::Vector2d &observed,
    double weight, T *residuals) {
  // Transform using Eigen operations (quaternions in xyzw format)
  Eigen::Map<const Eigen::Quaternion<T>> q_rig_from_world(rig_from_world_rot);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_rig_from_world(
      rig_from_world_trans);
  Eigen::Map<const Eigen::Quaternion<T>> q_cam_from_rig(cam_from_rig_rot);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam_from_rig(cam_from_rig_trans);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(point3D);

  // Transform to rig frame, then to camera frame
  Eigen::Matrix<T, 3, 1> point_rig = q_rig_from_world * p + t_rig_from_world;
  Eigen::Matrix<T, 3, 1> point_cam =
      q_cam_from_rig * point_rig + t_cam_from_rig;

  // Project to 2D
  T xy[2];
  CameraModel::ImgFromCam(camera_params, point_cam[0], point_cam[1],
                          point_cam[2], &xy[0], &xy[1]);
  residuals[0] = T(weight) * (xy[0] - T(observed.x()));
  residuals[1] = T(weight) * (xy[1] - T(observed.y()));
  return true;
}

// Point reprojection cost with constant rig
template <typename CameraModel, typename T>
inline bool ComputePointReprojectionCostConstantRig(
    const colmap::Rigid3d &cam_from_rig, const T *rig_from_world_rot,
    const T *rig_from_world_trans, const T *point3D, const T *camera_params,
    const Eigen::Vector2d &observed, double weight, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_rig.rotation();
  const Eigen::Vector3d &t = cam_from_rig.translation();
  // Use xyzw format (Eigen's coeffs() order)
  T cam_from_rig_rot[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T cam_from_rig_trans[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputePointReprojectionCostRig<CameraModel>(
      cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
      rig_from_world_trans, point3D, camera_params, observed, weight,
      residuals);
}

//============================================================================
// STATIC LINE REPROJECTION COST FUNCTIONS
// These take Plucker coordinates (dvec, mvec) directly.
// For MinimalPlucker input, first convert using MinimalPluckerToPlucker.
//============================================================================

// Helper: compute line reprojection residual from Plucker coordinates.
// Projects 3D line to 2D and computes point-to-line distance for endpoints.
template <typename CameraModel, typename T>
inline bool ComputeLineReprojectionCostFromPlucker(const T *camera_params,
                                                   const T *qvec, const T *tvec,
                                                   const T *dvec, const T *mvec,
                                                   const Line2d &observed,
                                                   T *residuals) {
  // Convert camera params to kvec [fx, fy, cx, cy]
  T kvec[4];
  ParamsToKvec<T>(CameraModel::model_id, camera_params, kvec);

  // Project 3D line to 2D homogeneous line coordinates [a, b, c]
  T line_coor[3];
  Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, line_coor);

  // Point-to-line distance for observed endpoints
  T denom =
      ceres::sqrt(line_coor[0] * line_coor[0] + line_coor[1] * line_coor[1]);

  if (denom < T(1e-12)) {
    residuals[0] = T(0);
    residuals[1] = T(0);
  } else {
    residuals[0] = (line_coor[0] * T(observed.start.x()) +
                    line_coor[1] * T(observed.start.y()) + line_coor[2]) /
                   denom;
    residuals[1] = (line_coor[0] * T(observed.end.x()) +
                    line_coor[1] * T(observed.end.y()) + line_coor[2]) /
                   denom;
  }
  return true;
}

// Line reprojection cost with variable pose (from Plucker)
template <typename CameraModel, typename T>
inline bool ComputeLineReprojectionCost(const T *camera_params, const T *qvec,
                                        const T *tvec, const T *dvec,
                                        const T *mvec, const Line2d &observed,
                                        T *residuals) {
  return ComputeLineReprojectionCostFromPlucker<CameraModel>(
      camera_params, qvec, tvec, dvec, mvec, observed, residuals);
}

// Line reprojection cost with constant pose (from Plucker)
template <typename CameraModel, typename T>
inline bool ComputeLineReprojectionCostConstantPose(
    const T *camera_params, const colmap::Rigid3d &cam_from_world,
    const T *dvec, const T *mvec, const Line2d &observed, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_world.rotation();
  const Eigen::Vector3d &t = cam_from_world.translation();
  // Note: Line_WorldToPixel expects xyzw quaternion format
  T qvec[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T tvec[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputeLineReprojectionCost<CameraModel>(
      camera_params, qvec, tvec, dvec, mvec, observed, residuals);
}

// Line reprojection cost with rig (from Plucker)
// Composes cam_from_rig * rig_from_world then projects
template <typename CameraModel, typename T>
inline bool ComputeLineReprojectionCostRig(
    const T *cam_from_rig_rot, const T *cam_from_rig_trans,
    const T *rig_from_world_rot, const T *rig_from_world_trans, const T *dvec,
    const T *mvec, const T *camera_params, const Line2d &observed,
    T *residuals) {
  // Compose transforms using Eigen quaternions (xyzw format)
  Eigen::Map<const Eigen::Quaternion<T>> q_rig_from_world(rig_from_world_rot);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_rig_from_world(
      rig_from_world_trans);
  Eigen::Map<const Eigen::Quaternion<T>> q_cam_from_rig(cam_from_rig_rot);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam_from_rig(cam_from_rig_trans);

  Eigen::Quaternion<T> q_cam_from_world = q_cam_from_rig * q_rig_from_world;
  Eigen::Matrix<T, 3, 1> t_cam_from_world =
      q_cam_from_rig * t_rig_from_world + t_cam_from_rig;

  T qvec[4] = {q_cam_from_world.x(), q_cam_from_world.y(), q_cam_from_world.z(),
               q_cam_from_world.w()};
  T tvec[3] = {t_cam_from_world[0], t_cam_from_world[1], t_cam_from_world[2]};

  return ComputeLineReprojectionCost<CameraModel>(
      camera_params, qvec, tvec, dvec, mvec, observed, residuals);
}

// Line reprojection cost with constant rig (from Plucker)
template <typename CameraModel, typename T>
inline bool ComputeLineReprojectionCostConstantRig(
    const colmap::Rigid3d &cam_from_rig, const T *rig_from_world_rot,
    const T *rig_from_world_trans, const T *dvec, const T *mvec,
    const T *camera_params, const Line2d &observed, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_rig.rotation();
  const Eigen::Vector3d &t = cam_from_rig.translation();
  T cam_from_rig_rot[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T cam_from_rig_trans[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputeLineReprojectionCostRig<CameraModel>(
      cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
      rig_from_world_trans, dvec, mvec, camera_params, observed, residuals);
}

//============================================================================

// Note: For point reprojection, we use COLMAP's cost functions directly:
// - colmap::ReprojErrorCostFunctor
// - colmap::ReprojErrorConstantPoseCostFunctor
// - colmap::RigReprojErrorCostFunctor
// - colmap::RigReprojErrorConstantRigCostFunctor
// Use colmap::CreateCameraCostFunction<...>() to create them.

// Forward declarations for analytical dispatch in factory
template <typename CameraModel> struct LineReprojectionCostFunctor;
template <typename CameraModel> struct LineReprojectionConstantPoseCostFunctor;

// Factory function for creating line reprojection cost functions based on
// camera model ID. Only supports undistorted camera models (line projection
// requires undistorted images).
template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction *
CreateLineCameraCostFunction(const colmap::CameraModelId camera_model_id,
                             Args &&...args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    if constexpr (std::is_same_v<CostFunctor<CameraModel>,                     \
                                 LineReprojectionCostFunctor<CameraModel>>) {  \
      return new AnalyticalLineReprojCostFunction<CameraModel>(                \
          std::forward<Args>(args)...);                                        \
    } else if constexpr (std::is_same_v<                                       \
                             CostFunctor<CameraModel>,                         \
                             LineReprojectionConstantPoseCostFunctor<          \
                                 CameraModel>>) {                              \
      return new AnalyticalLineReprojConstantPoseCostFunction<CameraModel>(    \
          std::forward<Args>(args)...);                                        \
    } else {                                                                   \
      return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);    \
    }

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error("Camera model not supported for line projection");
  }
}

// Line reprojection cost functor templated on camera model (2 residuals).
// Computes point-to-line distance for observed 2D line endpoints to the
// projected 3D line.
template <typename CameraModel> struct LineReprojectionCostFunctor {
  explicit LineReprojectionCostFunctor(const Line2d &observed)
      : observed_(observed) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);
    return ComputeLineReprojectionCost<CameraModel>(
        camera_params, qvec, tvec, dvec, mvec, observed_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed) {
    return new ceres::AutoDiffCostFunction<LineReprojectionCostFunctor, 2, 6, 7,
                                           CameraModel::num_params>(
        new LineReprojectionCostFunctor(observed));
  }

private:
  const Line2d observed_;
};

// Line reprojection with constant pose
template <typename CameraModel> struct LineReprojectionConstantPoseCostFunctor {
  LineReprojectionConstantPoseCostFunctor(const Line2d &observed,
                                          const colmap::Rigid3d &cam_from_world)
      : observed_(observed), cam_from_world_(cam_from_world) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const camera_params,
                  T *residuals) const {
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);
    return ComputeLineReprojectionCostConstantPose<CameraModel>(
        camera_params, cam_from_world_, dvec, mvec, observed_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed,
                                     const colmap::Rigid3d &cam_from_world) {
    return new ceres::AutoDiffCostFunction<
        LineReprojectionConstantPoseCostFunctor, 2, 6, CameraModel::num_params>(
        new LineReprojectionConstantPoseCostFunctor(observed, cam_from_world));
  }

private:
  const Line2d observed_;
  const colmap::Rigid3d cam_from_world_;
};

// Line reprojection with constant line (for localization: optimize pose only)
// Use MinimalInfiniteLine3d(line3d).data to get the minimal Plucker params.
template <typename CameraModel> struct LineReprojectionConstantLineCostFunctor {
  LineReprojectionConstantLineCostFunctor(const Line2d &observed,
                                          const V6D &params)
      : observed_(observed), params_(params) {}

  template <typename T>
  bool operator()(const T *const cam_from_world, const T *const camera_params,
                  T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;
    T line_params[6];
    for (int i = 0; i < 6; ++i) {
      line_params[i] = T(params_(i));
    }
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);
    return ComputeLineReprojectionCost<CameraModel>(
        camera_params, qvec, tvec, dvec, mvec, observed_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed,
                                     const V6D &params) {
    return new ceres::AutoDiffCostFunction<
        LineReprojectionConstantLineCostFunctor, 2, 7, CameraModel::num_params>(
        new LineReprojectionConstantLineCostFunctor(observed, params));
  }

private:
  const Line2d observed_;
  const V6D params_;
};

////////////////////////////////////////////////////////////////////////////////
// Rig cost functors for lines
////////////////////////////////////////////////////////////////////////////////

// Rig line reprojection with variable cam_from_rig and rig_from_world
template <typename CameraModel> struct RigLineReprojErrorCostFunctor {
  explicit RigLineReprojErrorCostFunctor(const Line2d &observed)
      : observed_(observed) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_rig,
                  const T *const rig_from_world, const T *const camera_params,
                  T *residuals) const {
    const T *cam_from_rig_rotation = cam_from_rig;
    const T *cam_from_rig_translation = cam_from_rig + 4;
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);
    return ComputeLineReprojectionCostRig<CameraModel>(
        cam_from_rig_rotation, cam_from_rig_translation,
        rig_from_world_rotation, rig_from_world_translation, dvec, mvec,
        camera_params, observed_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed) {
    return new ceres::AutoDiffCostFunction<RigLineReprojErrorCostFunctor, 2, 6,
                                           7, 7, CameraModel::num_params>(
        new RigLineReprojErrorCostFunctor(observed));
  }

private:
  const Line2d observed_;
};

// Rig line reprojection with constant cam_from_rig
template <typename CameraModel>
struct RigLineReprojErrorConstantRigCostFunctor {
  RigLineReprojErrorConstantRigCostFunctor(const Line2d &observed,
                                           const colmap::Rigid3d &cam_from_rig)
      : observed_(observed), cam_from_rig_(cam_from_rig) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const rig_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(line_params, dvec, mvec);
    return ComputeLineReprojectionCostConstantRig<CameraModel>(
        cam_from_rig_, rig_from_world_rotation, rig_from_world_translation,
        dvec, mvec, camera_params, observed_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed,
                                     const colmap::Rigid3d &cam_from_rig) {
    return new ceres::AutoDiffCostFunction<
        RigLineReprojErrorConstantRigCostFunctor, 2, 6, 7,
        CameraModel::num_params>(
        new RigLineReprojErrorConstantRigCostFunctor(observed, cam_from_rig));
  }

private:
  const Line2d observed_;
  const colmap::Rigid3d cam_from_rig_;
};

} // namespace estimators
} // namespace limap
