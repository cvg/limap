#include "limap/estimators/bundle_adjustment/group_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/ceres_angle_utils.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/groups/vp.h"
#include "limap/geometry/minimal_inf_line3d.h"
#include "limap/scene/group3d_with_active_labels.h"
#include "limap/scene/structure2d.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {
namespace estimators {

////////////////////////////////////////////////////////////////////////////////
// GroupBundleAdjustmentConfig (group-specific methods)
////////////////////////////////////////////////////////////////////////////////

void GroupBundleAdjustmentConfig::AddVariableGroup(group3D_t group3D_id) {
  variable_group3D_ids_.insert(group3D_id);
}

void GroupBundleAdjustmentConfig::AddConstantGroup(group3D_t group3D_id) {
  constant_group3D_ids_.insert(group3D_id);
}

bool GroupBundleAdjustmentConfig::HasGroup(group3D_t group3D_id) const {
  return HasVariableGroup(group3D_id) || HasConstantGroup(group3D_id);
}

bool GroupBundleAdjustmentConfig::HasVariableGroup(group3D_t group3D_id) const {
  return variable_group3D_ids_.count(group3D_id) > 0;
}

bool GroupBundleAdjustmentConfig::HasConstantGroup(group3D_t group3D_id) const {
  return constant_group3D_ids_.count(group3D_id) > 0;
}

void GroupBundleAdjustmentConfig::RemoveVariableGroup(group3D_t group3D_id) {
  variable_group3D_ids_.erase(group3D_id);
}

void GroupBundleAdjustmentConfig::RemoveConstantGroup(group3D_t group3D_id) {
  constant_group3D_ids_.erase(group3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// GroupBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

GroupBundleAdjuster::GroupBundleAdjuster(GroupBundleAdjustmentOptions options,
                                         GroupBundleAdjustmentConfig config,
                                         HolisticReconstruction &reconstruction)
    : BasePointLineBundleAdjuster(options, config, reconstruction),
      group_options_(options), variable_group3D_ids_(config.VariableGroups()),
      constant_group3D_ids_(config.ConstantGroups()),
      angle_loss_function_(CreateLossFunction(
          options.loss_function_type_angle, options.loss_function_scale_angle)),
      reprojection_loss_function_(
          CreateLossFunction(options.loss_function_type_reprojection,
                             options.loss_function_scale_reprojection)) {
  // Parse VP residual type
  if (options.vp_residual_type == "sin") {
    vp_residual_type_ = VPGroup::ResidualType::SINE;
  } else if (options.vp_residual_type == "cosine") {
    vp_residual_type_ = VPGroup::ResidualType::COSINE;
  } else {
    LOG(WARNING) << "Unknown VP residual type: " << options.vp_residual_type
                 << ", using 'sin'";
    vp_residual_type_ = VPGroup::ResidualType::SINE;
  }

  // Filter groups below active-association threshold
  const size_t min_active = options.min_active_group_associations;
  if (min_active > 0) {
    const auto &srec = reconstruction.StructureRecon();
    auto filter = [&](NodeHashSet<group3D_t> &ids) {
      for (auto it = ids.begin(); it != ids.end();) {
        const auto &group = srec.Group(*it);
        if (group.CountActiveAssociations() < min_active) {
          it = ids.erase(it);
        } else {
          ++it;
        }
      }
    };
    filter(variable_group3D_ids_);
    filter(constant_group3D_ids_);
  }

  // Add groups to problem (params are optimized in place in Group3d)
  for (const group3D_t group3D_id : variable_group3D_ids_) {
    AddGroupToProblem(group3D_id, reconstruction);
  }
  for (const group3D_t group3D_id : constant_group3D_ids_) {
    AddGroupToProblem(group3D_id, reconstruction);
  }

  // Angular constraints — add BEFORE ParameterizeGroups so that
  // parameter blocks introduced solely by angular constraint residuals (e.g.,
  // a VP with 0 line residuals in Step 1 BA) get their manifold set correctly.
  angular_constraint_threshold_cos_ =
      std::sin(group_options_.angular_constraint_threshold_deg * M_PI / 180.0);
  angular_constraint_loss_function_ =
      CreateLossFunction(group_options_.loss_function_type_angular_constraint,
                         group_options_.loss_function_scale_angular_constraint);

  if (group_options_.enforce_vp_angular_constraints) {
    AddVPAngularConstraints(reconstruction);
  }

  if (group_options_.enforce_plane_angular_constraints) {
    AddPlaneAngularConstraints(reconstruction);
  }

  // Parameterize groups (after all residuals are added, so all parameter
  // blocks are registered and receive their manifold + constant flags)
  ParameterizeGroups();
}

void GroupBundleAdjuster::AddGroupToProblem(
    group3D_t group3D_id, HolisticReconstruction &reconstruction) {
  Group3d &group3d = reconstruction.StructureRecon().Group(group3D_id);

  // Check if group has valid parameters
  double *group_params = group3d.GetParamsData();
  THROW_CHECK(group_params != nullptr && !group3d.GetParams().empty())
      << "Group " << group3D_id << " has no parameters";
  THROW_CHECK(group3d.CheckParams())
      << "Group " << group3D_id << " has invalid/unnormalized params";

  // Get group implementation for cost creation
  auto group_impl = GetGroup(group3d.type);
  THROW_CHECK(group_impl != nullptr)
      << "Cannot get implementation for group type " << group3d.type;

  // Configure VP group if needed
  if (group3d.type == GroupType::VP) {
    auto *vp_impl = dynamic_cast<VPGroup *>(group_impl.get());
    if (vp_impl) {
      vp_impl->SetResidualType(vp_residual_type_);
    }
  }

  double group_weight = GetGroupWeight(group3d.type);

  // Select loss function based on whether this group uses angle or reprojection
  // residuals. VP groups use angle-based loss; all others use reprojection
  // loss.
  ceres::LossFunction *base_loss = (group3d.type == GroupType::VP)
                                       ? angle_loss_function_.get()
                                       : reprojection_loss_function_.get();

  // Helper lambda to create scaled loss with optional length-based weighting.
  // For 2D line observations, pass line2d_length > 0 to scale by length.
  auto create_scaled_loss = [&](double line2d_length =
                                    0) -> ceres::LossFunction * {
    double weight = group_weight;
    if (line2d_length > 0 &&
        base_options_.line_weight_normalization_length > 0.0) {
      weight *= line2d_length / base_options_.line_weight_normalization_length;
    }
    group_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
        base_loss, weight, ceres::DO_NOT_TAKE_OWNERSHIP));
    return group_losses_.back().get();
  };

  size_t num_observations = 0;

  if (group3d.type == GroupType::VP) {
    // Add 3D line-to-VP constraints
    for (const auto &assoc_line : group3d.lines) {
      if (line_params_.find(assoc_line.idx) == line_params_.end()) {
        continue;
      }
      auto &line_param = line_params_.at(assoc_line.idx);

      // Create cost without weight, apply group_weight via ScaledLoss
      ceres::CostFunction *cost = group_impl->CreateLineCost3D();
      if (cost) {
        problem_->AddResidualBlock(cost, create_scaled_loss(),
                                   line_param.data.data(), group_params);
        angle_residual_infos_.push_back(
            {AngleResidualInfo::PARALLELISM, line_param.data.data(),
             group_params, group_weight, angle_loss_function_.get()});
        num_observations += 1;
      }
    }
  } else {
    // Non-VP groups (Plane, Sphere, etc.): use 2D reprojection costs
    // Add 2D point-to-group costs
    // Skip when disable_point_residuals: point3D.xyz is not in the problem
    if (!base_options_.disable_point_residuals) {
      for (const auto &assoc_point : group3d.points) {
        if (!reconstruction.PointRecon().ExistsPoint3D(assoc_point.idx)) {
          continue;
        }
        colmap::Point3D &point3D =
            reconstruction.PointRecon().Point3D(assoc_point.idx);

        // Iterate over track elements (2D observations)
        for (const auto &track_el : point3D.track.Elements()) {
          colmap::Image &image =
              reconstruction.PointRecon().Image(track_el.image_id);
          colmap::Camera &camera =
              reconstruction.PointRecon().Camera(image.CameraId());
          const colmap::Point2D &point2D = image.Point2D(track_el.point2D_idx);

          const colmap::sensor_t sensor_id = image.CameraPtr()->SensorId();
          const bool is_ref_sensor =
              image.FramePtr()->RigPtr()->IsRefSensor(sensor_id);
          const bool constant_sensor_from_rig =
              !options_.refine_sensor_from_rig ||
              config_.HasConstantSensorFromRigPose(sensor_id);
          const bool constant_rig_from_world =
              !options_.refine_rig_from_world ||
              config_.HasConstantRigFromWorldPose(image.FrameId());

          ceres::CostFunction *cost = nullptr;

          // Create cost without weight, apply group_weight via ScaledLoss
          if (is_ref_sensor) {
            // Reference sensor: cam_from_world = rig_from_world
            colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
            if (constant_rig_from_world) {
              cost = group_impl->CreatePointCost2DConstantPose(
                  camera.model_id, image.CamFromWorld());
              if (cost) {
                problem_->AddResidualBlock(cost, create_scaled_loss(),
                                           point3D.xyz.data(), group_params,
                                           camera.params.data());
              }
            } else {
              cost = group_impl->CreatePointCost2D(camera.model_id);
              if (cost) {
                problem_->AddResidualBlock(cost, create_scaled_loss(),
                                           point3D.xyz.data(),
                                           rig_from_world.params.data(),
                                           group_params, camera.params.data());
              }
            }
          } else {
            // Non-reference sensor: need to handle cam_from_rig
            colmap::Rigid3d &cam_from_rig =
                image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
            colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();

            if (constant_sensor_from_rig && constant_rig_from_world) {
              cost = group_impl->CreatePointCost2DConstantPose(
                  camera.model_id, cam_from_rig * rig_from_world);
              if (cost) {
                problem_->AddResidualBlock(cost, create_scaled_loss(),
                                           point3D.xyz.data(), group_params,
                                           camera.params.data());
              }
            } else if (!constant_rig_from_world && constant_sensor_from_rig) {
              cost = group_impl->CreatePointCost2DConstantRig(camera.model_id,
                                                              cam_from_rig);
              if (cost) {
                problem_->AddResidualBlock(cost, create_scaled_loss(),
                                           point3D.xyz.data(),
                                           rig_from_world.params.data(),
                                           group_params, camera.params.data());
              }
            } else {
              cost = group_impl->CreatePointCost2DRig(camera.model_id);
              if (cost) {
                problem_->AddResidualBlock(
                    cost, create_scaled_loss(), point3D.xyz.data(),
                    cam_from_rig.params.data(), rig_from_world.params.data(),
                    group_params, camera.params.data());
              }
            }
          }

          if (cost) {
            num_observations += 1;
          }
        }
      }
    } // !disable_point_residuals

    // Add 2D line-to-group costs
    for (const auto &assoc_line : group3d.lines) {
      if (line_params_.find(assoc_line.idx) == line_params_.end()) {
        continue;
      }
      auto &line_param = line_params_.at(assoc_line.idx);
      const auto &line3d = reconstruction.StructureRecon().Line(assoc_line.idx);

      // Iterate over track elements (2D observations)
      for (const auto &track_el : line3d.track.Elements()) {

        colmap::Image &image =
            reconstruction.PointRecon().Image(track_el.image_id);
        colmap::Camera &camera =
            reconstruction.PointRecon().Camera(image.CameraId());
        const Line2d &line2d = reconstruction.StructureRecon()
                                   .Structure2d(track_el.image_id)
                                   .Line(track_el.point2D_idx);

        const colmap::sensor_t sensor_id = image.CameraPtr()->SensorId();
        const bool is_ref_sensor =
            image.FramePtr()->RigPtr()->IsRefSensor(sensor_id);
        const bool constant_sensor_from_rig =
            !options_.refine_sensor_from_rig ||
            config_.HasConstantSensorFromRigPose(sensor_id);
        const bool constant_rig_from_world =
            !options_.refine_rig_from_world ||
            config_.HasConstantRigFromWorldPose(image.FrameId());

        ceres::CostFunction *cost = nullptr;
        const double line_len = line2d.Length();

        if (is_ref_sensor) {
          // Reference sensor: cam_from_world = rig_from_world
          colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
          if (constant_rig_from_world) {
            cost = group_impl->CreateLineCost2DConstantPose(
                camera.model_id, line2d, image.CamFromWorld());
            if (cost) {
              problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                         line_param.data.data(), group_params,
                                         camera.params.data());
            }
          } else {
            cost = group_impl->CreateLineCost2D(camera.model_id, line2d);
            if (cost) {
              problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                         line_param.data.data(),
                                         rig_from_world.params.data(),
                                         group_params, camera.params.data());
            }
          }
        } else {
          // Non-reference sensor: need to handle cam_from_rig
          colmap::Rigid3d &cam_from_rig =
              image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
          colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();

          if (constant_sensor_from_rig && constant_rig_from_world) {
            cost = group_impl->CreateLineCost2DConstantPose(
                camera.model_id, line2d, cam_from_rig * rig_from_world);
            if (cost) {
              problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                         line_param.data.data(), group_params,
                                         camera.params.data());
            }
          } else if (!constant_rig_from_world && constant_sensor_from_rig) {
            cost = group_impl->CreateLineCost2DConstantRig(
                camera.model_id, line2d, cam_from_rig);
            if (cost) {
              problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                         line_param.data.data(),
                                         rig_from_world.params.data(),
                                         group_params, camera.params.data());
            }
          } else {
            cost = group_impl->CreateLineCost2DRig(camera.model_id, line2d);
            if (cost) {
              problem_->AddResidualBlock(
                  cost, create_scaled_loss(line_len), line_param.data.data(),
                  cam_from_rig.params.data(), rig_from_world.params.data(),
                  group_params, camera.params.data());
            }
          }
        }

        if (cost) {
          num_observations += 1;
        }
      }
    }
  }

  group3D_num_observations_[group3D_id] = num_observations;
}

void GroupBundleAdjuster::ParameterizeGroups() {
  // Iterate over all groups in the config
  auto parameterize_group = [&](group3D_t group3D_id) {
    Group3d &group3d = reconstruction_.StructureRecon().Group(group3D_id);
    double *group_params = group3d.GetParamsData();

    if (!problem_->HasParameterBlock(group_params)) {
      return;
    }

    // Set appropriate manifold
    auto group_impl = GetGroup(group3d.type);
    if (group_impl) {
      ceres::Manifold *manifold = group_impl->CreateManifold3D();
      if (manifold) {
        problem_->SetManifold(group_params, manifold);
      }
    }

    // Set constant if not refining or if explicitly marked constant
    if (!group_options_.refine_groups || HasConstantGroup(group3D_id)) {
      problem_->SetParameterBlockConstant(group_params);
    }
  };

  for (const group3D_t group3D_id : variable_group3D_ids_) {
    parameterize_group(group3D_id);
  }
  for (const group3D_t group3D_id : constant_group3D_ids_) {
    parameterize_group(group3D_id);
  }
}

void GroupBundleAdjuster::AddVPAngularConstraints(
    HolisticReconstruction &reconstruction) {
  auto &srec = reconstruction.StructureRecon();

  // Collect VP group IDs that have observation residuals in the problem.
  // Groups with 0 observations (e.g., VPs in Step 1 BA with 0 reliable lines)
  // are excluded — angular constraints alone can't anchor a group.
  std::vector<group3D_t> vp_ids;
  auto collect_vps = [&](const NodeHashSet<group3D_t> &ids) {
    for (const group3D_t id : ids) {
      if (srec.Group(id).type == GroupType::VP &&
          problem_->HasParameterBlock(srec.Group(id).GetParamsData())) {
        vp_ids.push_back(id);
      }
    }
  };
  collect_vps(variable_group3D_ids_);
  collect_vps(constant_group3D_ids_);

  if (vp_ids.size() < 2) {
    return;
  }

  // Threshold values:
  // angular_constraint_threshold_cos_ = sin(threshold_deg) — upper bound for
  // orthogonal pairs cos(threshold_deg) — lower bound for parallel pairs
  const double parallel_threshold_cos =
      std::cos(group_options_.angular_constraint_threshold_deg * M_PI / 180.0);

  size_t num_ortho = 0, num_parallel = 0;
  for (size_t i = 0; i < vp_ids.size(); ++i) {
    Group3d &vp_i = srec.Group(vp_ids[i]);
    double *params_i = vp_i.GetParamsData();

    for (size_t j = i + 1; j < vp_ids.size(); ++j) {
      Group3d &vp_j = srec.Group(vp_ids[j]);
      double *params_j = vp_j.GetParamsData();

      const double cos_angle = AngleCosine3D<double>(params_i, params_j);
      const bool is_orthogonal = cos_angle < angular_constraint_threshold_cos_;
      const bool is_parallel = cos_angle > parallel_threshold_cos;

      if (!is_orthogonal && !is_parallel) {
        continue;
      }

      // Check covisibility: intersection of track image sets
      FlatHashSet<colmap::image_t> images_i;
      for (const auto &el : vp_i.track.Elements()) {
        images_i.insert(el.image_id);
      }
      size_t covisible = 0;
      for (const auto &el : vp_j.track.Elements()) {
        if (images_i.count(el.image_id)) {
          ++covisible;
        }
      }
      if (covisible < group_options_.angular_constraint_min_covisible_images) {
        continue;
      }

      if (is_orthogonal) {
        // Orthogonality: residual = |cos(angle)| → 0 when orthogonal
        ceres::CostFunction *cost = VPOrthogonalityCostFunctor::Create();
        group_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            angular_constraint_loss_function_.get(),
            group_options_.weight_vp_angular_constraint,
            ceres::DO_NOT_TAKE_OWNERSHIP));
        problem_->AddResidualBlock(cost, group_losses_.back().get(), params_i,
                                   params_j);
        angle_residual_infos_.push_back(
            {AngleResidualInfo::ORTHOGONALITY, params_i, params_j,
             group_options_.weight_vp_angular_constraint,
             angular_constraint_loss_function_.get()});
        ++num_ortho;
      } else {
        // Parallelism: residual = |sin(angle)| → 0 when parallel
        ceres::CostFunction *cost = VPParallelismCostFunctor::Create();
        group_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            angular_constraint_loss_function_.get(),
            group_options_.weight_vp_angular_constraint,
            ceres::DO_NOT_TAKE_OWNERSHIP));
        problem_->AddResidualBlock(cost, group_losses_.back().get(), params_i,
                                   params_j);
        angle_residual_infos_.push_back(
            {AngleResidualInfo::VP_PARALLELISM, params_i, params_j,
             group_options_.weight_vp_angular_constraint,
             angular_constraint_loss_function_.get()});
        ++num_parallel;
      }
    }
  }

  if (num_ortho > 0 || num_parallel > 0) {
    LOG(INFO) << "Added " << num_ortho << " VP orthogonality + " << num_parallel
              << " VP parallelism constraint(s)";
  }
}

void GroupBundleAdjuster::AddPlaneAngularConstraints(
    HolisticReconstruction &reconstruction) {
  auto &srec = reconstruction.StructureRecon();

  // Collect PLANE group IDs that have observation residuals in the problem.
  // Groups with 0 observations are excluded — angular constraints alone can't
  // anchor a group.
  std::vector<group3D_t> plane_ids;
  auto collect_planes = [&](const NodeHashSet<group3D_t> &ids) {
    for (const group3D_t id : ids) {
      if (srec.Group(id).type == GroupType::PLANE &&
          problem_->HasParameterBlock(srec.Group(id).GetParamsData())) {
        plane_ids.push_back(id);
      }
    }
  };
  collect_planes(variable_group3D_ids_);
  collect_planes(constant_group3D_ids_);

  if (plane_ids.size() < 2) {
    return;
  }

  // Threshold values:
  // angular_constraint_threshold_cos_ = sin(threshold_deg) — upper bound for
  // orthogonal pairs cos(threshold_deg) — lower bound for parallel pairs
  const double parallel_threshold_cos =
      std::cos(group_options_.angular_constraint_threshold_deg * M_PI / 180.0);

  size_t num_ortho = 0, num_parallel = 0;
  for (size_t i = 0; i < plane_ids.size(); ++i) {
    Group3d &plane_i = srec.Group(plane_ids[i]);
    double *params_i = plane_i.GetParamsData();

    for (size_t j = i + 1; j < plane_ids.size(); ++j) {
      Group3d &plane_j = srec.Group(plane_ids[j]);
      double *params_j = plane_j.GetParamsData();

      const double cos_angle = AngleCosine3D<double>(params_i, params_j);
      const bool is_orthogonal = cos_angle < angular_constraint_threshold_cos_;
      const bool is_parallel = cos_angle > parallel_threshold_cos;

      if (!is_orthogonal && !is_parallel) {
        continue;
      }

      // Check covisibility: intersection of track image sets
      FlatHashSet<colmap::image_t> images_i;
      for (const auto &el : plane_i.track.Elements()) {
        images_i.insert(el.image_id);
      }
      size_t covisible = 0;
      for (const auto &el : plane_j.track.Elements()) {
        if (images_i.count(el.image_id)) {
          ++covisible;
        }
      }
      if (covisible < group_options_.angular_constraint_min_covisible_images) {
        continue;
      }

      if (is_orthogonal) {
        // Orthogonality: residual = |cos(angle)| → 0 when orthogonal
        ceres::CostFunction *cost =
            PlaneNormalOrthogonalityCostFunctor::Create();
        group_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            angular_constraint_loss_function_.get(),
            group_options_.weight_plane_angular_constraint,
            ceres::DO_NOT_TAKE_OWNERSHIP));
        problem_->AddResidualBlock(cost, group_losses_.back().get(), params_i,
                                   params_j);
        angle_residual_infos_.push_back(
            {AngleResidualInfo::ORTHOGONALITY, params_i, params_j,
             group_options_.weight_plane_angular_constraint,
             angular_constraint_loss_function_.get()});
        ++num_ortho;
      } else {
        // Parallelism: residual = |sin(angle)| → 0 when parallel
        ceres::CostFunction *cost = PlaneNormalParallelismCostFunctor::Create();
        group_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            angular_constraint_loss_function_.get(),
            group_options_.weight_plane_angular_constraint,
            ceres::DO_NOT_TAKE_OWNERSHIP));
        problem_->AddResidualBlock(cost, group_losses_.back().get(), params_i,
                                   params_j);
        angle_residual_infos_.push_back(
            {AngleResidualInfo::PLANE_PARALLELISM, params_i, params_j,
             group_options_.weight_plane_angular_constraint,
             angular_constraint_loss_function_.get()});
        ++num_parallel;
      }
    }
  }

  if (num_ortho > 0 || num_parallel > 0) {
    LOG(INFO) << "Added " << num_ortho << " plane orthogonality + "
              << num_parallel << " plane parallelism constraint(s)";
  }
}

double GroupBundleAdjuster::GetGroupWeight(GroupType type) const {
  switch (type) {
  case GroupType::VP:
    return group_options_.weight_vp;
  case GroupType::PLANE:
    return group_options_.weight_plane;
  default:
    return 1.0;
  }
}

std::shared_ptr<colmap::BundleAdjustmentSummary> GroupBundleAdjuster::Solve() {
  if (problem_->NumResiduals() == 0) {
    return std::make_shared<colmap::BundleAdjustmentSummary>();
  }

  ceres::Solver::Options solver_options =
      options_.ceres->CreateSolverOptions(config_, *problem_);
  SetParameterBlockOrdering(solver_options);

  const bool vp_use_cosine =
      (vp_residual_type_ == VPGroup::ResidualType::COSINE);

  AngleCostAwareToleranceCallback tolerance_callback(
      base_options_.custom_function_tolerance, angle_residual_infos_,
      vp_use_cosine);
  if (base_options_.custom_function_tolerance > 0.0) {
    solver_options.callbacks.push_back(&tolerance_callback);
    solver_options.function_tolerance = 0.0;
    // The callback reads parameter values via raw pointers (to compute angle
    // cost). Ceres normally works with an internal copy and only writes back
    // after Solve(). This flag forces per-iteration writeback so the callback
    // sees current parameter values.
    solver_options.update_state_every_iteration = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);

  // Write back optimized line parameters
  WriteBackLines(reconstruction_);

  // Group params are optimized in place via GetParamsData().
  // Normalization is maintained by manifolds during optimization:
  // - VP: SphereManifold<3> ensures ||direction|| = 1
  // - Plane: ProductManifold<SphereManifold<3>, EuclideanManifold<1>>
  //          ensures ||(a,b,c)|| = 1, d is free
  // Verify params are still valid after optimization
  for (const group3D_t group3D_id : variable_group3D_ids_) {
    Group3d &group = reconstruction_.StructureRecon().Group(group3D_id);
    if (!group.CheckParams()) {
      LOG(WARNING) << "Group " << group3D_id
                   << " has invalid params after optimization";
    }
  }

  if (options_.print_summary) {
    colmap::PrintSolverSummary(summary, "Group bundle adjustment report");
  }

  return colmap::CeresBundleAdjustmentSummary::Create(summary);
}

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<GroupBundleAdjuster>
CreateGroupBundleAdjuster(GroupBundleAdjustmentOptions options,
                          GroupBundleAdjustmentConfig config,
                          HolisticReconstruction &reconstruction) {
  return std::make_unique<GroupBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

////////////////////////////////////////////////////////////////////////////////
// RefineGroupByReprojError (free function)
////////////////////////////////////////////////////////////////////////////////

bool RefineGroupByReprojError(Group3d &group,
                              const HolisticReconstruction &recon) {
  // VP groups have no 2D point-on-surface cost — skip
  if (group.type == GroupType::VP) {
    return false;
  }

  auto group_impl = GetGroup(group.type);
  if (!group_impl) {
    return false;
  }

  // Copy group params (Ceres will optimize these)
  std::vector<double> group_params = group.GetParams();

  ceres::Problem problem;

  // Storage for copies of constant parameter blocks (points).
  struct PointConstBlock {
    std::vector<double> cam_params;
    std::vector<double> point3d; // size 3
  };
  std::vector<PointConstBlock> point_const_blocks;
  point_const_blocks.reserve(group.points.size() * 4);

  // Storage for copies of constant parameter blocks (lines).
  // line_params stored as std::vector<double> (heap-allocated) rather than
  // inline V6D so that pointers survive vector reallocation.
  struct LineConstBlock {
    std::vector<double> cam_params;
    std::vector<double> line_params; // size 6
  };
  std::vector<LineConstBlock> line_const_blocks;
  line_const_blocks.reserve(group.lines.size() * 4);

  const auto &point_recon = recon.PointRecon();
  const auto &struct_recon = recon.StructureRecon();

  // Add point residuals
  for (const auto &assoc : group.points) {
    if (!point_recon.ExistsPoint3D(assoc.idx)) {
      continue;
    }
    const colmap::Point3D &point3D = point_recon.Point3D(assoc.idx);

    for (const auto &track_el : point3D.track.Elements()) {
      if (!point_recon.ExistsImage(track_el.image_id)) {
        continue;
      }
      const colmap::Image &image = point_recon.Image(track_el.image_id);
      const colmap::Camera &camera = point_recon.Camera(image.CameraId());

      ceres::CostFunction *cost = group_impl->CreatePointCost2DConstantPose(
          camera.model_id, image.CamFromWorld());
      if (!cost) {
        continue;
      }

      point_const_blocks.emplace_back();
      auto &cb = point_const_blocks.back();
      cb.cam_params.assign(camera.params.begin(), camera.params.end());
      cb.point3d = {point3D.xyz.x(), point3D.xyz.y(), point3D.xyz.z()};

      problem.AddResidualBlock(cost, new ceres::HuberLoss(6.0),
                               cb.point3d.data(), group_params.data(),
                               cb.cam_params.data());
      problem.SetParameterBlockConstant(cb.cam_params.data());
      problem.SetParameterBlockConstant(cb.point3d.data());
    }
  }

  // Add line residuals (only for group types that support it, e.g. Plane)
  for (const auto &assoc : group.lines) {
    if (!struct_recon.ExistsLine3D(assoc.idx)) {
      continue;
    }
    const auto &line3d = struct_recon.Line(assoc.idx);

    for (const auto &track_el : line3d.track.Elements()) {

      if (!point_recon.ExistsImage(track_el.image_id)) {
        continue;
      }
      const colmap::Image &image = point_recon.Image(track_el.image_id);
      if (!struct_recon.ExistsStructure2D(track_el.image_id)) {
        continue;
      }
      const Structure2d &s2d = struct_recon.Structure2d(track_el.image_id);
      line2D_t line2D_idx = static_cast<line2D_t>(track_el.point2D_idx);
      if (line2D_idx >= static_cast<line2D_t>(s2d.NumLines())) {
        continue;
      }
      const Line2d &obs = s2d.Line(line2D_idx);
      const colmap::Camera &camera = point_recon.Camera(image.CameraId());

      ceres::CostFunction *cost = group_impl->CreateLineCost2DConstantPose(
          camera.model_id, obs, image.CamFromWorld());
      if (!cost) {
        continue;
      }

      line_const_blocks.emplace_back();
      auto &cb = line_const_blocks.back();
      cb.cam_params.assign(camera.params.begin(), camera.params.end());
      MinimalInfiniteLine3d mil(line3d);
      cb.line_params.assign(mil.data.data(), mil.data.data() + 6);

      problem.AddResidualBlock(cost, new ceres::HuberLoss(6.0),
                               cb.line_params.data(), group_params.data(),
                               cb.cam_params.data());
      problem.SetParameterBlockConstant(cb.cam_params.data());
      problem.SetParameterBlockConstant(cb.line_params.data());
    }
  }

  if (problem.NumResidualBlocks() == 0) {
    return false;
  }

  // Set manifold on group params
  ceres::Manifold *manifold = group_impl->CreateManifold3D();
  if (manifold) {
    problem.SetManifold(group_params.data(), manifold);
  }

  // Solve
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = 10;
  solver_options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // Write back refined params
  group.SetParams(group_params);
  return true;
}

} // namespace estimators
} // namespace limap
