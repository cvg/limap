#pragma once

#include <ceres/ceres.h>
#include <memory>
#include <vector>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/custom_tolerance_callbacks.h" // AngleResidualInfo
#include "limap/geometry/groups.h"
#include "limap/geometry/groups/vp.h"
#include "limap/scene/group3d.h"
#include "limap/util/types.h"

namespace limap {
namespace estimators {

// Options for group bundle adjustment
// Extends PointLineBundleAdjustmentOptions with group-specific parameters
struct GroupBundleAdjustmentOptions : public PointLineBundleAdjustmentOptions {
  // Group refinement options
  bool refine_groups = true;

  // Weights for different group constraint types
  double weight_vp = 5e3;
  double weight_plane = 0.5;

  // VP residual formulation: "sin" or "cosine"
  std::string vp_residual_type = "sin";

  // Loss function for angle-based group constraints (VP: residuals in [0, 1])
  LossFunctionType loss_function_type_angle = LossFunctionType::CAUCHY;
  double loss_function_scale_angle = 0.2;

  // Loss function for reprojection-based group constraints (Plane, etc.: pixel
  // residuals)
  LossFunctionType loss_function_type_reprojection = LossFunctionType::CAUCHY;
  double loss_function_scale_reprojection = 1.5;

  // Minimum active associations for a group to be included in BA.
  // Groups below this threshold are silently skipped (not added to the
  // problem). Set to 0 to include all groups.
  size_t min_active_group_associations = 10;

  // Minimum reliable associations for a group to be included in step 2 BA.
  // Counts line associations in the reliable set + active point associations.
  // Groups with fewer reliable constraints may have poorly-estimated geometry
  // and can pull unreliable lines to wrong positions. Set to 0 to disable.
  size_t min_reliable_group_associations_step2 = 20;

  // Angular constraints (Manhattan world prior)
  // Constrains near-orthogonal pairs to stay orthogonal and near-parallel
  // pairs to stay parallel during BA.
  // Shared threshold/covisibility/loss settings:
  double angular_constraint_threshold_deg = 5.0; // max deviation from 90° or 0°
  size_t angular_constraint_min_covisible_images = 3;
  LossFunctionType loss_function_type_angular_constraint =
      LossFunctionType::CAUCHY;
  double loss_function_scale_angular_constraint = 0.1;
  // Per-type enable/weight:
  bool enforce_vp_angular_constraints = true;
  double weight_vp_angular_constraint = 1e5;
  bool enforce_plane_angular_constraints = true;
  double weight_plane_angular_constraint = 1e5;
};

// Configuration for group bundle adjustment
// Extends PointLineBundleAdjustmentConfig with group-specific management
class GroupBundleAdjustmentConfig : public PointLineBundleAdjustmentConfig {
public:
  GroupBundleAdjustmentConfig() = default;

  size_t NumGroups() const {
    return variable_group3D_ids_.size() + constant_group3D_ids_.size();
  }

  // Groups
  void AddVariableGroup(group3D_t group3D_id);
  void AddConstantGroup(group3D_t group3D_id);
  bool HasGroup(group3D_t group3D_id) const;
  bool HasVariableGroup(group3D_t group3D_id) const;
  bool HasConstantGroup(group3D_t group3D_id) const;
  void RemoveVariableGroup(group3D_t group3D_id);
  void RemoveConstantGroup(group3D_t group3D_id);

  const NodeHashSet<group3D_t> &VariableGroups() const {
    return variable_group3D_ids_;
  }
  const NodeHashSet<group3D_t> &ConstantGroups() const {
    return constant_group3D_ids_;
  }

private:
  NodeHashSet<group3D_t> variable_group3D_ids_;
  NodeHashSet<group3D_t> constant_group3D_ids_;
};

// Group bundle adjuster extending BasePointLineBundleAdjuster
// Adds group constraints (VP, Plane) to point-line bundle adjustment
class GroupBundleAdjuster : public BasePointLineBundleAdjuster {
public:
  GroupBundleAdjuster(GroupBundleAdjustmentOptions options,
                      GroupBundleAdjustmentConfig config,
                      HolisticReconstruction &reconstruction);

  ~GroupBundleAdjuster() override = default;

  // Run the optimization (override from base class)
  std::shared_ptr<colmap::BundleAdjustmentSummary> Solve() override;

  // Group-specific options
  bool refine_groups() const { return group_options_.refine_groups; }
  double weight_vp() const { return group_options_.weight_vp; }
  double weight_plane() const { return group_options_.weight_plane; }
  VPGroup::ResidualType vp_residual_type() const { return vp_residual_type_; }

  // Group-specific config access
  const NodeHashSet<group3D_t> &VariableGroups() const {
    return variable_group3D_ids_;
  }
  const NodeHashSet<group3D_t> &ConstantGroups() const {
    return constant_group3D_ids_;
  }
  bool HasVariableGroup(group3D_t group3D_id) const {
    return variable_group3D_ids_.count(group3D_id) > 0;
  }
  bool HasConstantGroup(group3D_t group3D_id) const {
    return constant_group3D_ids_.count(group3D_id) > 0;
  }

protected:
  // Methods for adding group residuals
  virtual void AddGroupToProblem(group3D_t group3D_id,
                                 HolisticReconstruction &reconstruction);

  // Methods for parameterization
  virtual void ParameterizeGroups();

  // Add angular constraints (orthogonality + parallelism) between VP pairs
  void AddVPAngularConstraints(HolisticReconstruction &reconstruction);

  // Add angular constraints (orthogonality + parallelism) between plane pairs
  void AddPlaneAngularConstraints(HolisticReconstruction &reconstruction);

  // Get weight based on group type
  double GetGroupWeight(GroupType type) const;

  // Group options (stored as typed struct; base options_ is sliced)
  GroupBundleAdjustmentOptions group_options_;

  // Computed from group_options_ in constructor
  VPGroup::ResidualType vp_residual_type_;
  double angular_constraint_threshold_cos_; // cos(90° - threshold_deg)

  // Group-specific config (extracted from GroupBundleAdjustmentConfig)
  NodeHashSet<group3D_t> variable_group3D_ids_;
  NodeHashSet<group3D_t> constant_group3D_ids_;

  // Loss functions for angle-based and reprojection-based group constraints
  std::unique_ptr<ceres::LossFunction> angle_loss_function_;
  std::unique_ptr<ceres::LossFunction> reprojection_loss_function_;

  // Angular constraint loss function (shared for orthogonality + parallelism)
  std::unique_ptr<ceres::LossFunction> angular_constraint_loss_function_;

  // Track number of observations per group
  FlatHashMap<group3D_t, size_t> group3D_num_observations_;

  // Storage for scaled loss functions (for proper memory management)
  // Each residual block gets its own ScaledLoss wrapping the appropriate loss
  std::vector<std::unique_ptr<ceres::LossFunction>> group_losses_;

  // Angle residual infos for AngleCostAwareToleranceCallback.
  // Collected during construction at VP line-to-VP, VP angular constraint,
  // and plane angular constraint residual addition sites.
  std::vector<AngleResidualInfo> angle_residual_infos_;
};

// Factory function to create a default group bundle adjuster
std::unique_ptr<GroupBundleAdjuster>
CreateGroupBundleAdjuster(GroupBundleAdjustmentOptions options,
                          GroupBundleAdjustmentConfig config,
                          HolisticReconstruction &reconstruction);

// Refine group parameters by minimizing 2D reprojection error.
// Runs a small Ceres problem where only group_params are variable;
// all camera params, poses, and point3D positions are held constant.
// VP groups are skipped (returns false) since they have no 2D point cost.
// Returns true if refinement ran successfully, false if skipped.
bool RefineGroupByReprojError(Group3d &group,
                              const HolisticReconstruction &recon);

} // namespace estimators
} // namespace limap
