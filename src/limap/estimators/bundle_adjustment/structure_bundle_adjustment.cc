#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/analytical_wireframe_cost_functions.h"
#include "limap/estimators/bundle_adjustment/wireframe_cost_functions.h"

#include <colmap/util/logging.h>

namespace limap {
namespace estimators {

////////////////////////////////////////////////////////////////////////////////
// StructureBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

StructureBundleAdjuster::StructureBundleAdjuster(
    StructureBundleAdjustmentOptions options,
    StructureBundleAdjustmentConfig config,
    HolisticReconstruction &reconstruction)
    : GroupBundleAdjuster(options, config, reconstruction),
      structure_options_(options), wireframe_loss_function_(CreateLossFunction(
                                       options.loss_function_type_wireframe,
                                       options.loss_function_scale_wireframe)) {
  // Add wireframe edges to problem (if enabled)
  if (structure_options_.refine_wireframe) {
    // Reconstruct wireframe from 2D if forced or if empty
    if (options.force_reconstruct_wireframe3d ||
        reconstruction.StructureRecon().Wireframe().CountEdges() == 0) {
      reconstruction.StructureRecon().Wireframe().Clear();
      reconstruction.StructureRecon().ConstructWireframe3dFrom2d(
          options.wireframe_voting);
    }

    const Wireframe3d &wireframe = reconstruction.StructureRecon().Wireframe();
    for (const auto &edge : wireframe.GetAllEdges()) {
      AddWireframeEdgeToProblem(edge.point_idx, edge.line_idx, edge.w,
                                reconstruction);
    }
  }
}

void StructureBundleAdjuster::AddWireframeEdgeToProblem(
    point3D_t point3D_id, line3D_t line3D_id, double edge_weight,
    HolisticReconstruction &reconstruction) {

  // Check that the point exists in the reconstruction
  if (!reconstruction.PointRecon().ExistsPoint3D(point3D_id)) {
    return;
  }

  // Check that the line exists in our line_params (i.e., it's in the problem)
  if (line_params_.find(line3D_id) == line_params_.end()) {
    return;
  }

  colmap::Point3D &point3D = reconstruction.PointRecon().Point3D(point3D_id);
  const auto &line3d = reconstruction.StructureRecon().Line(line3D_id);
  auto &line_param = line_params_.at(line3D_id);

  // Helper lambda to create scaled loss with optional length-based weighting.
  // The final weight is: structure_options_.weight_wireframe * line2d_length /
  // norm_length (or just structure_options_.weight_wireframe if length-based
  // weighting is disabled).
  auto create_scaled_loss = [&](double line2d_length) -> ceres::LossFunction * {
    double weight = structure_options_.weight_wireframe;
    if (base_options_.line_weight_normalization_length > 0.0) {
      weight *= line2d_length / base_options_.line_weight_normalization_length;
    }
    wireframe_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
        wireframe_loss_function_.get(), weight, ceres::DO_NOT_TAKE_OWNERSHIP));
    return wireframe_losses_.back().get();
  };

  // Build image ID sets for both tracks. Wireframe residuals are only added in
  // images where the OTHER feature is NOT already observed — those images
  // already have direct reprojection residuals which provide the same
  // information. Wireframe's value is cross-view: constraining a feature from
  // views where only its partner is observed.
  FlatHashSet<colmap::image_t> point_image_ids;
  for (const auto &el : point3D.track.Elements()) {
    point_image_ids.insert(el.image_id);
  }
  FlatHashSet<colmap::image_t> line_image_ids;
  for (const auto &el : line3d.track.Elements()) {
    line_image_ids.insert(el.image_id);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Point-to-Line residuals: iterate over line's track
  // Only in images where the point is NOT observed (cross-view constraint)
  // Skip when disable_point_residuals: point3D.xyz is not in the problem
  //////////////////////////////////////////////////////////////////////////////
  if (!base_options_.disable_point_residuals) {
    for (const auto &track_el : line3d.track.Elements()) {
      // Skip images where the point already has a direct reprojection residual
      if (point_image_ids.count(track_el.image_id)) {
        continue;
      }

      colmap::Image &image =
          reconstruction.PointRecon().Image(track_el.image_id);
      colmap::Camera &camera =
          reconstruction.PointRecon().Camera(image.CameraId());

      // Get observed 2D line
      const Line2d &observed_line = reconstruction.StructureRecon()
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

      const double line_len = observed_line.Length();

      if (is_ref_sensor) {
        // Reference sensor: cam_from_world = rig_from_world
        colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
        if (constant_rig_from_world) {
          cost = CreateAnalyticalPointToLineConstantPose(
              camera.model_id, observed_line, image.CamFromWorld());
          if (cost) {
            problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                       point3D.xyz.data(),
                                       camera.params.data());
          }
        } else {
          cost = CreatePointToLineCostFunction<PointToLineCostFunctor>(
              camera.model_id, observed_line);
          if (cost) {
            problem_->AddResidualBlock(
                cost, create_scaled_loss(line_len), point3D.xyz.data(),
                rig_from_world.params.data(), camera.params.data());
          }
        }
      } else {
        // Non-reference sensor: need to handle cam_from_rig
        colmap::Rigid3d &cam_from_rig =
            image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
        colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();

        if (constant_sensor_from_rig && constant_rig_from_world) {
          cost = CreateAnalyticalPointToLineConstantPose(
              camera.model_id, observed_line, cam_from_rig * rig_from_world);
          if (cost) {
            problem_->AddResidualBlock(cost, create_scaled_loss(line_len),
                                       point3D.xyz.data(),
                                       camera.params.data());
          }
        } else if (!constant_rig_from_world && constant_sensor_from_rig) {
          cost =
              CreatePointToLineCostFunction<PointToLineConstantRigCostFunctor>(
                  camera.model_id, observed_line, cam_from_rig);
          if (cost) {
            problem_->AddResidualBlock(
                cost, create_scaled_loss(line_len), point3D.xyz.data(),
                rig_from_world.params.data(), camera.params.data());
          }
        } else {
          cost = CreatePointToLineCostFunction<PointToLineRigCostFunctor>(
              camera.model_id, observed_line);
          if (cost) {
            problem_->AddResidualBlock(
                cost, create_scaled_loss(line_len), point3D.xyz.data(),
                cam_from_rig.params.data(), rig_from_world.params.data(),
                camera.params.data());
          }
        }
      }

      if (cost) {
        wireframe_num_observations_ += 1;
      }
    }
  } // !disable_point_residuals

  //////////////////////////////////////////////////////////////////////////////
  // Line-to-Point residuals: iterate over point's track
  // Only in images where the line is NOT observed (cross-view constraint)
  //////////////////////////////////////////////////////////////////////////////
  for (const auto &track_el : point3D.track.Elements()) {
    // Skip images where the line already has a direct reprojection residual
    if (line_image_ids.count(track_el.image_id)) {
      continue;
    }
    colmap::Image &image = reconstruction.PointRecon().Image(track_el.image_id);
    colmap::Camera &camera =
        reconstruction.PointRecon().Camera(image.CameraId());

    // Get observed 2D point
    const colmap::Point2D &observed_point2D =
        image.Point2D(track_el.point2D_idx);
    const Eigen::Vector2d observed_point = observed_point2D.xy;

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

    if (is_ref_sensor) {
      // Reference sensor: cam_from_world = rig_from_world
      colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
      if (constant_rig_from_world) {
        cost = CreateAnalyticalLineToPointConstantPose(
            camera.model_id, observed_point, image.CamFromWorld());
        if (cost) {
          problem_->AddResidualBlock(cost, create_scaled_loss(0),
                                     line_param.data.data(),
                                     camera.params.data());
        }
      } else {
        cost = CreateLineToPointCostFunction<LineToPointCostFunctor>(
            camera.model_id, observed_point);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(0), line_param.data.data(),
              rig_from_world.params.data(), camera.params.data());
        }
      }
    } else {
      // Non-reference sensor: need to handle cam_from_rig
      colmap::Rigid3d &cam_from_rig =
          image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
      colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();

      if (constant_sensor_from_rig && constant_rig_from_world) {
        cost = CreateAnalyticalLineToPointConstantPose(
            camera.model_id, observed_point, cam_from_rig * rig_from_world);
        if (cost) {
          problem_->AddResidualBlock(cost, create_scaled_loss(0),
                                     line_param.data.data(),
                                     camera.params.data());
        }
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        cost = CreateLineToPointCostFunction<LineToPointConstantRigCostFunctor>(
            camera.model_id, observed_point, cam_from_rig);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(0), line_param.data.data(),
              rig_from_world.params.data(), camera.params.data());
        }
      } else {
        cost = CreateLineToPointCostFunction<LineToPointRigCostFunctor>(
            camera.model_id, observed_point);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(0), line_param.data.data(),
              cam_from_rig.params.data(), rig_from_world.params.data(),
              camera.params.data());
        }
      }
    }

    if (cost) {
      wireframe_num_observations_ += 1;
    }
  }
}

std::shared_ptr<colmap::BundleAdjustmentSummary>
StructureBundleAdjuster::Solve() {
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
  // Verify params are still valid after optimization
  for (const group3D_t group3D_id : variable_group3D_ids_) {
    Group3d &group = reconstruction_.StructureRecon().Group(group3D_id);
    if (!group.CheckParams()) {
      LOG(WARNING) << "Group " << group3D_id
                   << " has invalid params after optimization";
    }
  }

  if (options_.print_summary) {
    colmap::PrintSolverSummary(summary, "Structure bundle adjustment report");
  }

  return colmap::CeresBundleAdjustmentSummary::Create(summary);
}

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<BasePointLineBundleAdjuster>
CreateStructureBundleAdjuster(StructureBundleAdjustmentOptions options,
                              StructureBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction) {
  return std::make_unique<StructureBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

} // namespace estimators
} // namespace limap
