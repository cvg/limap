#pragma once

#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>
#include <colmap/util/enum_utils.h>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace limap {

// Group type enumeration
MAKE_ENUM_CLASS_OVERLOAD_STREAM(GroupType, -1, INVALID, VP, PLANE, SPHERE,
                                CYLINDER, ELLIPSOID, CUBOID, CONE);

// Abstract base class for group-specific implementations
class BaseGroup {
public:
  virtual ~BaseGroup() = default;

  // Get default inlier threshold for RANSAC during initialization
  // Units depend on group type:
  //   VP: angular error (radians)
  //   Plane/Sphere: distance to surface (world units)
  virtual double GetDefaultThreshold() const = 0;

  // Get the expected number of parameters for this group type in 2D
  virtual size_t GetNumParamsIn2D() const = 0;

  // Get the expected number of parameters for this group type in 3D
  virtual size_t GetNumParamsIn3D() const = 0;

  // Get default/identity parameters for this group type in 2D.
  // Returns valid parameters that can be used as initialization.
  virtual std::vector<double> GetDefaultParams2D() const = 0;

  // Get default/identity parameters for this group type in 3D.
  // Returns valid, normalized parameters that can be used as initialization.
  virtual std::vector<double> GetDefaultParams3D() const = 0;

  // Normalize 3D params in-place to canonical form.
  // Called by Group3d on construction and SetParams.
  // During optimization, manifolds maintain these constraints.
  virtual void NormalizeParams3D(double *params) const = 0;

  // Check if 3D params are valid and normalized (within tolerance).
  // Returns true if valid, false otherwise.
  virtual bool CheckParams3D(const double *params, double tol = 1e-6) const = 0;

  //==========================================================================
  // SURFACE PROJECTION (runtime evaluation, not Ceres)
  //==========================================================================
  // Project a 3D point onto the group surface. Default: identity (no-op).
  // Override in surface groups (Plane, Sphere, Cylinder, Ellipsoid).
  virtual V3D ProjectPoint(const V3D &point, const double *params) const {
    return point;
  }

  // Project a 3D line onto the group surface. Default: identity (no-op).
  // Override in groups that support line projection (Plane).
  virtual Line3d ProjectLine(const Line3d &line, const double *params) const {
    return line;
  }

  //==========================================================================
  // 3D COST FACTORIES (world space, no camera transformation)
  //==========================================================================
  // These create Ceres cost functions for group constraints in 3D.
  // Parameter blocks:
  //   - CreateLineCost3D: (line_params[6], group_params[N])
  //   - CreatePointCost3D: (point3D[3], group_params[N])
  // Returns nullptr if this constraint type is not applicable.
  // Ownership: Caller takes ownership of returned CostFunction*.
  // Note: Weight should be applied via ceres::ScaledLoss, not in cost functor.

  virtual ceres::CostFunction *CreateLineCost3D() const { return nullptr; }

  virtual ceres::CostFunction *CreatePointCost3D() const { return nullptr; }

  virtual int NumLineResiduals3D() const { return 4; }
  virtual int NumPointResiduals3D() const { return 3; }

  //==========================================================================
  // MANIFOLD FACTORY (for group parameter optimization)
  //==========================================================================
  // Creates the appropriate Ceres manifold for this group's 3D parameters.
  // Ownership: Caller takes ownership of returned Manifold*.

  virtual ceres::Manifold *CreateManifold3D() const { return nullptr; }

  //==========================================================================
  // 2D COST FACTORIES (image space, requires camera pose)
  //==========================================================================
  // These project 3D features through the group surface to 2D and compare
  // with observed. Uses PlaneProjection then standard camera projection.
  // Note: Weight should be applied via ceres::ScaledLoss, not in cost functor.
  //
  // Variable pose:   params (camera_params, qvec, tvec, feature, group)
  // Constant pose:   params (camera_params, feature, group)
  // Rig:             params (cam_from_rig_rot/trans, rig_from_world_rot/trans,
  //                         feature, group, camera_params)
  // Constant rig:    params (rig_from_world_rot/trans, feature, group,
  //                         camera_params)

  // Point: residual = reproject(point) - reproject(surface_project(point))
  virtual ceres::CostFunction *
  CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreatePointCost2DConstantPose(colmap::CameraModelId camera_model_id,
                                const colmap::Rigid3d &cam_from_world) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreatePointCost2DConstantRig(colmap::CameraModelId camera_model_id,
                               const colmap::Rigid3d &cam_from_rig) const {
    return nullptr;
  }

  virtual int NumPointResiduals2D() const { return 2; }

  // Line: residual = line_reproj(orig) - line_reproj(surface_project(orig))
  // observed_2d endpoints are used as evaluation points for point-to-line dist
  virtual ceres::CostFunction *
  CreateLineCost2D(colmap::CameraModelId camera_model_id,
                   const Line2d &observed_2d) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreateLineCost2DConstantPose(colmap::CameraModelId camera_model_id,
                               const Line2d &observed_2d,
                               const colmap::Rigid3d &cam_from_world) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreateLineCost2DRig(colmap::CameraModelId camera_model_id,
                      const Line2d &observed_2d) const {
    return nullptr;
  }

  virtual ceres::CostFunction *
  CreateLineCost2DConstantRig(colmap::CameraModelId camera_model_id,
                              const Line2d &observed_2d,
                              const colmap::Rigid3d &cam_from_rig) const {
    return nullptr;
  }

  virtual int NumLineResiduals2D() const { return 2; }
};

// Factory function to get the appropriate implementation based on group type
std::unique_ptr<BaseGroup> GetGroup(GroupType type);

// Get all valid group types (excludes INVALID)
inline std::vector<GroupType> GetAllGroupTypes() {
  return {GroupType::VP,       GroupType::PLANE,     GroupType::SPHERE,
          GroupType::CYLINDER, GroupType::ELLIPSOID, GroupType::CUBOID,
          GroupType::CONE};
}

// Get number of parameters for the corresponding group type in 2D
size_t GetNumParamsIn2DByGroupType(GroupType type);

// Get number of parameters for the corresponding group type in 3D
size_t GetNumParamsIn3DByGroupType(GroupType type);

// Normalize 3D group parameters in-place to canonical form
void NormalizeGroupParams3D(GroupType type, double *params);

// Check if 3D group parameters are valid and normalized
bool CheckGroupParams3D(GroupType type, const double *params,
                        double tol = 1e-6);

// Get default 2D parameters for a group type
std::vector<double> GetDefaultGroupParams2D(GroupType type);

// Get default 3D parameters for a group type
std::vector<double> GetDefaultGroupParams3D(GroupType type);

} // namespace limap
