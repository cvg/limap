#include "limap/estimators/bundle_adjustment/wireframe_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/analytical_wireframe_cost_functions.h"
#include "limap/estimators/bundle_adjustment/wireframe_cost_functions.h"

#include <colmap/util/logging.h>

namespace limap {
namespace estimators {

////////////////////////////////////////////////////////////////////////////////
// WireframeBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

WireframeBundleAdjuster::WireframeBundleAdjuster(
    WireframeBundleAdjustmentOptions options,
    PointLineBundleAdjustmentConfig config,
    HolisticReconstruction &reconstruction)
    : BasePointLineBundleAdjuster(options, config, reconstruction),
      wireframe_options_(options), wireframe_loss_function_(CreateLossFunction(
                                       options.loss_function_type_wireframe,
                                       options.loss_function_scale_wireframe)) {
  // Add wireframe edges to problem (if enabled)
  if (wireframe_options_.refine_wireframe) {
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

void WireframeBundleAdjuster::AddWireframeEdgeToProblem(
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

  // Helper lambda to create and store scaled loss function
  // Note: edge_weight is used for filtering during wireframe construction,
  // but not for weighting residuals (all edges treated equally in BA)
  // TODO: Investigate whether using edge_weight to scale residuals is better
  auto create_scaled_loss = [&]() -> ceres::LossFunction * {
    wireframe_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
        wireframe_loss_function_.get(), wireframe_options_.weight_wireframe,
        ceres::DO_NOT_TAKE_OWNERSHIP));
    return wireframe_losses_.back().get();
  };

  //////////////////////////////////////////////////////////////////////////////
  // Point-to-Line residuals: iterate over line's track
  // For each line observation, reproject point and measure distance to line
  //////////////////////////////////////////////////////////////////////////////
  for (const auto &track_el : line3d.track.Elements()) {

    colmap::Image &image = reconstruction.PointRecon().Image(track_el.image_id);
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

    if (is_ref_sensor) {
      // Reference sensor: cam_from_world = rig_from_world
      colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
      if (constant_rig_from_world) {
        cost = CreateAnalyticalPointToLineConstantPose(
            camera.model_id, observed_line, image.CamFromWorld());
        if (cost) {
          problem_->AddResidualBlock(cost, create_scaled_loss(),
                                     point3D.xyz.data(), camera.params.data());
        }
      } else {
        cost = CreatePointToLineCostFunction<PointToLineCostFunctor>(
            camera.model_id, observed_line);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), point3D.xyz.data(),
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
          problem_->AddResidualBlock(cost, create_scaled_loss(),
                                     point3D.xyz.data(), camera.params.data());
        }
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        cost = CreatePointToLineCostFunction<PointToLineConstantRigCostFunctor>(
            camera.model_id, observed_line, cam_from_rig);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), point3D.xyz.data(),
              rig_from_world.params.data(), camera.params.data());
        }
      } else {
        cost = CreatePointToLineCostFunction<PointToLineRigCostFunctor>(
            camera.model_id, observed_line);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), point3D.xyz.data(),
              cam_from_rig.params.data(), rig_from_world.params.data(),
              camera.params.data());
        }
      }
    }

    if (cost) {
      wireframe_num_observations_ += 1;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Line-to-Point residuals: iterate over point's track
  // For each point observation, reproject line and measure distance from point
  //////////////////////////////////////////////////////////////////////////////
  for (const auto &track_el : point3D.track.Elements()) {
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
          problem_->AddResidualBlock(cost, create_scaled_loss(),
                                     line_param.data.data(),
                                     camera.params.data());
        }
      } else {
        cost = CreateLineToPointCostFunction<LineToPointCostFunctor>(
            camera.model_id, observed_point);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), line_param.data.data(),
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
          problem_->AddResidualBlock(cost, create_scaled_loss(),
                                     line_param.data.data(),
                                     camera.params.data());
        }
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        cost = CreateLineToPointCostFunction<LineToPointConstantRigCostFunctor>(
            camera.model_id, observed_point, cam_from_rig);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), line_param.data.data(),
              rig_from_world.params.data(), camera.params.data());
        }
      } else {
        cost = CreateLineToPointCostFunction<LineToPointRigCostFunctor>(
            camera.model_id, observed_point);
        if (cost) {
          problem_->AddResidualBlock(
              cost, create_scaled_loss(), line_param.data.data(),
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
WireframeBundleAdjuster::Solve() {
  if (problem_->NumResiduals() == 0) {
    return std::make_shared<colmap::BundleAdjustmentSummary>();
  }

  ceres::Solver::Options solver_options =
      options_.ceres->CreateSolverOptions(config_, *problem_);
  SetParameterBlockOrdering(solver_options);

  CustomToleranceCallback custom_tolerance_callback(
      base_options_.custom_function_tolerance);
  if (base_options_.custom_function_tolerance > 0.0) {
    solver_options.callbacks.push_back(&custom_tolerance_callback);
    solver_options.function_tolerance = 0.0;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);

  // Write back optimized line parameters
  WriteBackLines(reconstruction_);

  if (options_.print_summary) {
    colmap::PrintSolverSummary(summary, "Wireframe bundle adjustment report");
  }

  return colmap::CeresBundleAdjustmentSummary::Create(summary);
}

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WireframeBundleAdjuster>
CreateWireframeBundleAdjuster(WireframeBundleAdjustmentOptions options,
                              PointLineBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction) {
  return std::make_unique<WireframeBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

} // namespace estimators
} // namespace limap
