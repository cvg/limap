#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/line2d.h"
#include "limap/util/eigen_types.h"

namespace limap {
namespace estimators {

//============================================================================
// POINT-TO-LINE COST FUNCTIONS
// Reproject 3D point, measure distance vector to observed 2D line (2 residuals)
// Residuals: perpendicular displacement vector (d*a, d*b) where d is signed
// distance and (a,b) is the normalized line normal.
//============================================================================

// Point-to-line cost with variable pose
// Parameters: (camera_params, qvec, tvec, point3D)
// Observed: 2D line from image (ground truth)
// Residuals: 2D perpendicular distance vector from reprojected point to line
template <typename CameraModel, typename T>
inline bool ComputePointToLineCost(const T *camera_params, const T *qvec,
                                   const T *tvec, const T *point3D,
                                   const Line2d &observed_line, T *residuals) {
  // Transform to camera frame using Eigen operations (qvec is in xyzw format)
  Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(point3D);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);

  Eigen::Matrix<T, 3, 1> point_cam = q * p + t;

  // Project to 2D
  T xy[2];
  CameraModel::ImgFromCam(camera_params, point_cam[0], point_cam[1],
                          point_cam[2], &xy[0], &xy[1]);

  // Compute homogeneous line coordinates [a, b, c] from observed line endpoints
  // Line equation: a*x + b*y + c = 0
  const T a = T(observed_line.end.y() - observed_line.start.y());
  const T b = T(observed_line.start.x() - observed_line.end.x());
  const T c = T(observed_line.end.x() * observed_line.start.y() -
                observed_line.start.x() * observed_line.end.y());

  // Normalize line coefficients
  const T denom = ceres::sqrt(a * a + b * b);
  if (denom < T(1e-12)) {
    residuals[0] = T(0);
    residuals[1] = T(0);
    return true;
  }

  const T a_norm = a / denom;
  const T b_norm = b / denom;
  const T c_norm = c / denom;

  // Signed distance from point to line
  const T d = a_norm * xy[0] + b_norm * xy[1] + c_norm;

  // Perpendicular distance vector: (d*a, d*b)
  residuals[0] = d * a_norm;
  residuals[1] = d * b_norm;
  return true;
}

// Point-to-line cost with constant pose
template <typename CameraModel, typename T>
inline bool ComputePointToLineCostConstantPose(
    const T *camera_params, const colmap::Rigid3d &cam_from_world,
    const T *point3D, const Line2d &observed_line, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_world.rotation();
  const Eigen::Vector3d &t = cam_from_world.translation();
  T qvec[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T tvec[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputePointToLineCost<CameraModel>(camera_params, qvec, tvec, point3D,
                                             observed_line, residuals);
}

// Point-to-line cost with rig (variable cam_from_rig and rig_from_world)
template <typename CameraModel, typename T>
inline bool ComputePointToLineCostRig(const T *cam_from_rig_rot,
                                      const T *cam_from_rig_trans,
                                      const T *rig_from_world_rot,
                                      const T *rig_from_world_trans,
                                      const T *point3D, const T *camera_params,
                                      const Line2d &observed_line,
                                      T *residuals) {
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

  // Compute homogeneous line coordinates
  const T a = T(observed_line.end.y() - observed_line.start.y());
  const T b = T(observed_line.start.x() - observed_line.end.x());
  const T c = T(observed_line.end.x() * observed_line.start.y() -
                observed_line.start.x() * observed_line.end.y());

  const T denom = ceres::sqrt(a * a + b * b);
  if (denom < T(1e-12)) {
    residuals[0] = T(0);
    residuals[1] = T(0);
    return true;
  }

  const T a_norm = a / denom;
  const T b_norm = b / denom;
  const T c_norm = c / denom;

  const T d = a_norm * xy[0] + b_norm * xy[1] + c_norm;
  residuals[0] = d * a_norm;
  residuals[1] = d * b_norm;
  return true;
}

// Point-to-line cost with constant rig
template <typename CameraModel, typename T>
inline bool ComputePointToLineCostConstantRig(
    const colmap::Rigid3d &cam_from_rig, const T *rig_from_world_rot,
    const T *rig_from_world_trans, const T *point3D, const T *camera_params,
    const Line2d &observed_line, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_rig.rotation();
  const Eigen::Vector3d &t = cam_from_rig.translation();
  T cam_from_rig_rot[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T cam_from_rig_trans[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputePointToLineCostRig<CameraModel>(
      cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
      rig_from_world_trans, point3D, camera_params, observed_line, residuals);
}

//============================================================================
// LINE-TO-POINT COST FUNCTIONS
// Reproject 3D line, measure distance vector from observed 2D point (2
// residuals) Residuals: perpendicular displacement vector (d*a, d*b)
//============================================================================

// Line-to-point cost with variable pose
// Parameters: (camera_params, qvec, tvec, line_params)
// Observed: 2D point from image (ground truth)
// Residuals: 2D perpendicular distance vector from observed point to line
template <typename CameraModel, typename T>
inline bool ComputeLineToPointCost(const T *camera_params, const T *qvec,
                                   const T *tvec, const T *line_params,
                                   const Eigen::Vector2d &observed_point,
                                   T *residuals) {
  // Convert minimal Plucker to Plucker
  T dvec[3], mvec[3];
  MinimalPluckerToPlucker<T>(line_params, dvec, mvec);

  // Convert camera params to kvec [fx, fy, cx, cy]
  T kvec[4];
  ParamsToKvec<T>(CameraModel::model_id, camera_params, kvec);

  // Project 3D line to 2D homogeneous line coordinates [a, b, c]
  T line_coor[3];
  Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, line_coor);

  // Normalize line coefficients
  T denom =
      ceres::sqrt(line_coor[0] * line_coor[0] + line_coor[1] * line_coor[1]);
  if (denom < T(1e-12)) {
    residuals[0] = T(0);
    residuals[1] = T(0);
    return true;
  }

  const T a_norm = line_coor[0] / denom;
  const T b_norm = line_coor[1] / denom;
  const T c_norm = line_coor[2] / denom;

  // Signed distance from observed point to line
  const T d =
      a_norm * T(observed_point.x()) + b_norm * T(observed_point.y()) + c_norm;

  // Perpendicular distance vector
  residuals[0] = d * a_norm;
  residuals[1] = d * b_norm;
  return true;
}

// Line-to-point cost with constant pose
template <typename CameraModel, typename T>
inline bool ComputeLineToPointCostConstantPose(
    const T *camera_params, const colmap::Rigid3d &cam_from_world,
    const T *line_params, const Eigen::Vector2d &observed_point, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_world.rotation();
  const Eigen::Vector3d &t = cam_from_world.translation();
  T qvec[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T tvec[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputeLineToPointCost<CameraModel>(
      camera_params, qvec, tvec, line_params, observed_point, residuals);
}

// Line-to-point cost with rig (variable cam_from_rig and rig_from_world)
template <typename CameraModel, typename T>
inline bool ComputeLineToPointCostRig(
    const T *cam_from_rig_rot, const T *cam_from_rig_trans,
    const T *rig_from_world_rot, const T *rig_from_world_trans,
    const T *line_params, const T *camera_params,
    const Eigen::Vector2d &observed_point, T *residuals) {
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

  return ComputeLineToPointCost<CameraModel>(
      camera_params, qvec, tvec, line_params, observed_point, residuals);
}

// Line-to-point cost with constant rig
template <typename CameraModel, typename T>
inline bool ComputeLineToPointCostConstantRig(
    const colmap::Rigid3d &cam_from_rig, const T *rig_from_world_rot,
    const T *rig_from_world_trans, const T *line_params, const T *camera_params,
    const Eigen::Vector2d &observed_point, T *residuals) {
  const Eigen::Quaterniond &q = cam_from_rig.rotation();
  const Eigen::Vector3d &t = cam_from_rig.translation();
  T cam_from_rig_rot[4] = {T(q.x()), T(q.y()), T(q.z()), T(q.w())};
  T cam_from_rig_trans[3] = {T(t.x()), T(t.y()), T(t.z())};
  return ComputeLineToPointCostRig<CameraModel>(
      cam_from_rig_rot, cam_from_rig_trans, rig_from_world_rot,
      rig_from_world_trans, line_params, camera_params, observed_point,
      residuals);
}

//============================================================================
// COST FUNCTORS
// Note: Weight should be applied via ceres::ScaledLoss when adding residuals
//============================================================================

// Point-to-Line: Variable pose (point3D, cam_from_world, camera_params) -> 2
// residuals
template <typename CameraModel> struct PointToLineCostFunctor {
  explicit PointToLineCostFunctor(const Line2d &observed_line)
      : observed_line_(observed_line) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;
    return ComputePointToLineCost<CameraModel>(
        camera_params, qvec, tvec, point3D, observed_line_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed_line) {
    return new ceres::AutoDiffCostFunction<PointToLineCostFunctor, 2, 3, 7,
                                           CameraModel::num_params>(
        new PointToLineCostFunctor(observed_line));
  }

private:
  const Line2d observed_line_;
};

// Point-to-Line: Constant pose (point3D, camera_params) -> 2 residuals
template <typename CameraModel> struct PointToLineConstantPoseCostFunctor {
  PointToLineConstantPoseCostFunctor(const Line2d &observed_line,
                                     const colmap::Rigid3d &cam_from_world)
      : observed_line_(observed_line), cam_from_world_(cam_from_world) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const camera_params,
                  T *residuals) const {
    return ComputePointToLineCostConstantPose<CameraModel>(
        camera_params, cam_from_world_, point3D, observed_line_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed_line,
                                     const colmap::Rigid3d &cam_from_world) {
    return new ceres::AutoDiffCostFunction<PointToLineConstantPoseCostFunctor,
                                           2, 3, CameraModel::num_params>(
        new PointToLineConstantPoseCostFunctor(observed_line, cam_from_world));
  }

private:
  const Line2d observed_line_;
  const colmap::Rigid3d cam_from_world_;
};

// Point-to-Line: Rig with variable cam_from_rig and rig_from_world
template <typename CameraModel> struct PointToLineRigCostFunctor {
  explicit PointToLineRigCostFunctor(const Line2d &observed_line)
      : observed_line_(observed_line) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_rig,
                  const T *const rig_from_world, const T *const camera_params,
                  T *residuals) const {
    const T *cam_from_rig_rotation = cam_from_rig;
    const T *cam_from_rig_translation = cam_from_rig + 4;
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    return ComputePointToLineCostRig<CameraModel>(
        cam_from_rig_rotation, cam_from_rig_translation,
        rig_from_world_rotation, rig_from_world_translation, point3D,
        camera_params, observed_line_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed_line) {
    return new ceres::AutoDiffCostFunction<PointToLineRigCostFunctor, 2, 3, 7,
                                           7, CameraModel::num_params>(
        new PointToLineRigCostFunctor(observed_line));
  }

private:
  const Line2d observed_line_;
};

// Point-to-Line: Constant rig (point3D, rig_from_world, camera_params) -> 2
// residuals
template <typename CameraModel> struct PointToLineConstantRigCostFunctor {
  PointToLineConstantRigCostFunctor(const Line2d &observed_line,
                                    const colmap::Rigid3d &cam_from_rig)
      : observed_line_(observed_line), cam_from_rig_(cam_from_rig) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const rig_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    return ComputePointToLineCostConstantRig<CameraModel>(
        cam_from_rig_, rig_from_world_rotation, rig_from_world_translation,
        point3D, camera_params, observed_line_, residuals);
  }

  static ceres::CostFunction *Create(const Line2d &observed_line,
                                     const colmap::Rigid3d &cam_from_rig) {
    return new ceres::AutoDiffCostFunction<PointToLineConstantRigCostFunctor, 2,
                                           3, 7, CameraModel::num_params>(
        new PointToLineConstantRigCostFunctor(observed_line, cam_from_rig));
  }

private:
  const Line2d observed_line_;
  const colmap::Rigid3d cam_from_rig_;
};

// Line-to-Point: Variable pose (line_params, cam_from_world, camera_params) ->
// 2 residuals
template <typename CameraModel> struct LineToPointCostFunctor {
  explicit LineToPointCostFunctor(const Eigen::Vector2d &observed_point)
      : observed_point_(observed_point) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *qvec = cam_from_world;
    const T *tvec = cam_from_world + 4;
    return ComputeLineToPointCost<CameraModel>(
        camera_params, qvec, tvec, line_params, observed_point_, residuals);
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &observed_point) {
    return new ceres::AutoDiffCostFunction<LineToPointCostFunctor, 2, 6, 7,
                                           CameraModel::num_params>(
        new LineToPointCostFunctor(observed_point));
  }

private:
  const Eigen::Vector2d observed_point_;
};

// Line-to-Point: Constant pose (line_params, camera_params) -> 2 residuals
template <typename CameraModel> struct LineToPointConstantPoseCostFunctor {
  LineToPointConstantPoseCostFunctor(const Eigen::Vector2d &observed_point,
                                     const colmap::Rigid3d &cam_from_world)
      : observed_point_(observed_point), cam_from_world_(cam_from_world) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const camera_params,
                  T *residuals) const {
    return ComputeLineToPointCostConstantPose<CameraModel>(
        camera_params, cam_from_world_, line_params, observed_point_,
        residuals);
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &observed_point,
                                     const colmap::Rigid3d &cam_from_world) {
    return new ceres::AutoDiffCostFunction<LineToPointConstantPoseCostFunctor,
                                           2, 6, CameraModel::num_params>(
        new LineToPointConstantPoseCostFunctor(observed_point, cam_from_world));
  }

private:
  const Eigen::Vector2d observed_point_;
  const colmap::Rigid3d cam_from_world_;
};

// Line-to-Point: Rig with variable cam_from_rig and rig_from_world
template <typename CameraModel> struct LineToPointRigCostFunctor {
  explicit LineToPointRigCostFunctor(const Eigen::Vector2d &observed_point)
      : observed_point_(observed_point) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const cam_from_rig,
                  const T *const rig_from_world, const T *const camera_params,
                  T *residuals) const {
    const T *cam_from_rig_rotation = cam_from_rig;
    const T *cam_from_rig_translation = cam_from_rig + 4;
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    return ComputeLineToPointCostRig<CameraModel>(
        cam_from_rig_rotation, cam_from_rig_translation,
        rig_from_world_rotation, rig_from_world_translation, line_params,
        camera_params, observed_point_, residuals);
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &observed_point) {
    return new ceres::AutoDiffCostFunction<LineToPointRigCostFunctor, 2, 6, 7,
                                           7, CameraModel::num_params>(
        new LineToPointRigCostFunctor(observed_point));
  }

private:
  const Eigen::Vector2d observed_point_;
};

// Line-to-Point: Constant rig
template <typename CameraModel> struct LineToPointConstantRigCostFunctor {
  LineToPointConstantRigCostFunctor(const Eigen::Vector2d &observed_point,
                                    const colmap::Rigid3d &cam_from_rig)
      : observed_point_(observed_point), cam_from_rig_(cam_from_rig) {}

  template <typename T>
  bool operator()(const T *const line_params, const T *const rig_from_world,
                  const T *const camera_params, T *residuals) const {
    const T *rig_from_world_rotation = rig_from_world;
    const T *rig_from_world_translation = rig_from_world + 4;
    return ComputeLineToPointCostConstantRig<CameraModel>(
        cam_from_rig_, rig_from_world_rotation, rig_from_world_translation,
        line_params, camera_params, observed_point_, residuals);
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &observed_point,
                                     const colmap::Rigid3d &cam_from_rig) {
    return new ceres::AutoDiffCostFunction<LineToPointConstantRigCostFunctor, 2,
                                           6, 7, CameraModel::num_params>(
        new LineToPointConstantRigCostFunctor(observed_point, cam_from_rig));
  }

private:
  const Eigen::Vector2d observed_point_;
  const colmap::Rigid3d cam_from_rig_;
};

//============================================================================
// FACTORY FUNCTIONS
//============================================================================

// Factory function for point-to-line cost functions
template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction *
CreatePointToLineCostFunction(const colmap::CameraModelId camera_model_id,
                              Args &&...args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error(
        "Camera model not supported for wireframe projection");
  }
}

// Factory function for line-to-point cost functions
template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction *
CreateLineToPointCostFunction(const colmap::CameraModelId camera_model_id,
                              Args &&...args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error(
        "Camera model not supported for wireframe projection");
  }
}

} // namespace estimators
} // namespace limap
