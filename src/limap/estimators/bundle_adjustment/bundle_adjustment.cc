#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/cost_functions.h"

#include <colmap/estimators/cost_functions/manifold.h>
#include <colmap/scene/camera.h>
#include <colmap/scene/frame.h>
#include <colmap/scene/image.h>
#include <colmap/scene/point2d.h>
#include <colmap/scene/point3d.h>
#include <colmap/scene/rig.h>

namespace limap {
namespace estimators {

////////////////////////////////////////////////////////////////////////////////
// PointLineBundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

void PointLineBundleAdjustmentConfig::AddVariableLine(line3D_t line3D_id) {
  variable_line3D_ids_.insert(line3D_id);
}

void PointLineBundleAdjustmentConfig::AddConstantLine(line3D_t line3D_id) {
  constant_line3D_ids_.insert(line3D_id);
}

bool PointLineBundleAdjustmentConfig::HasLine(line3D_t line3D_id) const {
  return HasVariableLine(line3D_id) || HasConstantLine(line3D_id);
}

bool PointLineBundleAdjustmentConfig::HasVariableLine(
    line3D_t line3D_id) const {
  return variable_line3D_ids_.count(line3D_id) > 0;
}

bool PointLineBundleAdjustmentConfig::HasConstantLine(
    line3D_t line3D_id) const {
  return constant_line3D_ids_.count(line3D_id) > 0;
}

void PointLineBundleAdjustmentConfig::RemoveVariableLine(line3D_t line3D_id) {
  variable_line3D_ids_.erase(line3D_id);
}

void PointLineBundleAdjustmentConfig::RemoveConstantLine(line3D_t line3D_id) {
  constant_line3D_ids_.erase(line3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BasePointLineBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BasePointLineBundleAdjuster::BasePointLineBundleAdjuster(
    PointLineBundleAdjustmentOptions options,
    PointLineBundleAdjustmentConfig config,
    HolisticReconstruction &reconstruction)
    : colmap::CeresBundleAdjuster(options, config), base_options_(options),
      variable_line3D_ids_(config.VariableLines()),
      constant_line3D_ids_(config.ConstantLines()),
      reconstruction_(reconstruction),
      loss_function_(options_.ceres->CreateLossFunction()),
      line_loss_function_(CreateLossFunction(
          options.loss_function_type_line, options.loss_function_scale_line)) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_shared<ceres::Problem>(problem_options);

  // Initialize line parameters only for configured (variable + constant) lines.
  // This ensures AddImageToProblem only adds residuals for lines in the config,
  // not all lines in the reconstruction (important for 2-step BA).
  for (const line3D_t line_id : variable_line3D_ids_) {
    if (reconstruction_.StructureRecon().ExistsLine3D(line_id)) {
      const auto &line = reconstruction_.StructureRecon().Line(line_id);
      if (line.Length() <= 0.0) {
        LOG(WARNING) << "Skipping zero-length variable line " << line_id;
        continue;
      }
      line_params_[line_id] = MinimalInfiniteLine3d(line);
    }
  }
  for (const line3D_t line_id : constant_line3D_ids_) {
    if (reconstruction_.StructureRecon().ExistsLine3D(line_id)) {
      const auto &line = reconstruction_.StructureRecon().Line(line_id);
      if (line.Length() <= 0.0) {
        LOG(WARNING) << "Skipping zero-length constant line " << line_id;
        continue;
      }
      line_params_[line_id] = MinimalInfiniteLine3d(line);
    }
  }

  // Set up problem
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first
  for (const colmap::image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction);
  }
  for (const colmap::point3D_t point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction);
  }
  for (const colmap::point3D_t point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction);
  }
  for (const line3D_t line3D_id : variable_line3D_ids_) {
    AddLineToProblem(line3D_id, reconstruction);
  }
  for (const line3D_t line3D_id : constant_line3D_ids_) {
    AddLineToProblem(line3D_id, reconstruction);
  }

  ParameterizeCameras(reconstruction);
  ParameterizeImages(reconstruction);
  ParameterizePoints(reconstruction);
  ParameterizeLines();
}

ceres::LossFunction *
BasePointLineBundleAdjuster::CreateLineLoss(double line2d_length) {
  // Use the dedicated line loss function (default: CAUCHY 1.0).
  // Falls back to the shared loss if line_loss_function_ is null (TRIVIAL).
  ceres::LossFunction *base_loss =
      line_loss_function_ ? line_loss_function_.get() : loss_function_.get();

  if (base_options_.line_weight_normalization_length <= 0.0) {
    // Length-based weighting disabled; use the base loss directly
    return base_loss;
  }
  const double weight = base_options_.weight_line * line2d_length /
                        base_options_.line_weight_normalization_length;
  line_scaled_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
      base_loss, weight, ceres::DO_NOT_TAKE_OWNERSHIP));
  return line_scaled_losses_.back().get();
}

void BasePointLineBundleAdjuster::AddImageToProblem(
    colmap::image_t image_id, HolisticReconstruction &reconstruction) {
  colmap::Image &image = reconstruction.PointRecon().Image(image_id);
  colmap::Camera &camera = reconstruction.PointRecon().Camera(image.CameraId());

  const colmap::sensor_t sensor_id = image.CameraPtr()->SensorId();

  // Check if this camera is the reference sensor of the rig
  // Reference sensor has identity cam_from_rig, so we can use simpler cost
  // functions
  const bool is_ref_sensor = image.FramePtr()->RigPtr()->IsRefSensor(sensor_id);

  size_t num_observations = 0;

  if (is_ref_sensor) {
    // Reference sensor case: cam_from_world = rig_from_world
    colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
    rig_from_world.rotation().normalize();

    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // Add point residuals
    if (!base_options_.disable_point_residuals) {
      for (const colmap::Point2D &point2D : image.Points2D()) {
        if (!point2D.HasPoint3D()) {
          continue;
        }

        num_observations += 1;
        point3D_num_observations_[point2D.point3D_id] += 1;

        colmap::Point3D &point3D =
            reconstruction.PointRecon().Point3D(point2D.point3D_id);

        if (constant_rig_from_world) {
          problem_->AddResidualBlock(
              colmap::CreateCameraCostFunction<
                  colmap::ReprojErrorConstantPoseCostFunctor>(
                  camera.model_id, point2D.xy, image.CamFromWorld()),
              loss_function_.get(), point3D.xyz.data(), camera.params.data());
        } else {
          problem_->AddResidualBlock(
              colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
                  camera.model_id, point2D.xy),
              loss_function_.get(), point3D.xyz.data(),
              rig_from_world.params.data(), camera.params.data());
        }
      }
    }

    // Add line residuals (only active observations)
    const Structure2d &structure2d =
        reconstruction.StructureRecon().Structure2d(image_id);
    for (size_t line2d_idx = 0; line2d_idx < structure2d.NumLines();
         ++line2d_idx) {
      const Line2d &line2d = structure2d.Line(line2d_idx);
      if (!line2d.HasLine3D()) {
        continue;
      }

      line3D_t line3D_id = line2d.line3D_id;
      if (line_params_.find(line3D_id) == line_params_.end()) {
        continue;
      }

      num_observations += 1;
      line3D_num_observations_[line3D_id] += 1;

      auto &line_param = line_params_.at(line3D_id);

      ceres::LossFunction *line_loss = CreateLineLoss(line2d.Length());
      if (constant_rig_from_world) {
        problem_->AddResidualBlock(
            CreateLineCameraCostFunction<
                LineReprojectionConstantPoseCostFunctor>(
                camera.model_id, line2d, image.CamFromWorld()),
            line_loss, line_param.data.data(), camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateLineCameraCostFunction<LineReprojectionCostFunctor>(
                camera.model_id, line2d),
            line_loss, line_param.data.data(), rig_from_world.params.data(),
            camera.params.data());
      }
    }
  } else {
    // Non-reference sensor case: need to handle cam_from_rig
    colmap::Rigid3d &cam_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
    rig_from_world.rotation().normalize();
    cam_from_rig.rotation().normalize();

    const bool constant_sensor_from_rig =
        !options_.refine_sensor_from_rig ||
        config_.HasConstantSensorFromRigPose(sensor_id);
    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // Add point residuals
    if (!base_options_.disable_point_residuals) {
      for (const colmap::Point2D &point2D : image.Points2D()) {
        if (!point2D.HasPoint3D()) {
          continue;
        }

        num_observations += 1;
        point3D_num_observations_[point2D.point3D_id] += 1;

        colmap::Point3D &point3D =
            reconstruction.PointRecon().Point3D(point2D.point3D_id);

        // The !constant_sensor_from_rig && constant_rig_from_world case is
        // intentionally not handled (uncommon in practice)
        if (constant_sensor_from_rig && constant_rig_from_world) {
          problem_->AddResidualBlock(
              colmap::CreateCameraCostFunction<
                  colmap::ReprojErrorConstantPoseCostFunctor>(
                  camera.model_id, point2D.xy, cam_from_rig * rig_from_world),
              loss_function_.get(), point3D.xyz.data(), camera.params.data());
        } else if (!constant_rig_from_world && constant_sensor_from_rig) {
          problem_->AddResidualBlock(
              colmap::CreateCameraCostFunction<
                  colmap::RigReprojErrorConstantRigCostFunctor>(
                  camera.model_id, point2D.xy, cam_from_rig),
              loss_function_.get(), point3D.xyz.data(),
              rig_from_world.params.data(), camera.params.data());
        } else {
          problem_->AddResidualBlock(colmap::CreateCameraCostFunction<
                                         colmap::RigReprojErrorCostFunctor>(
                                         camera.model_id, point2D.xy),
                                     loss_function_.get(), point3D.xyz.data(),
                                     cam_from_rig.params.data(),
                                     rig_from_world.params.data(),
                                     camera.params.data());
        }
      }
    }

    // Add line residuals (only active observations)
    const Structure2d &structure2d =
        reconstruction.StructureRecon().Structure2d(image_id);
    for (size_t line2d_idx = 0; line2d_idx < structure2d.NumLines();
         ++line2d_idx) {
      const Line2d &line2d = structure2d.Line(line2d_idx);
      if (!line2d.HasLine3D()) {
        continue;
      }

      line3D_t line3D_id = line2d.line3D_id;
      if (line_params_.find(line3D_id) == line_params_.end()) {
        continue;
      }

      num_observations += 1;
      line3D_num_observations_[line3D_id] += 1;

      auto &line_param = line_params_.at(line3D_id);

      ceres::LossFunction *line_loss = CreateLineLoss(line2d.Length());
      if (constant_sensor_from_rig && constant_rig_from_world) {
        problem_->AddResidualBlock(
            CreateLineCameraCostFunction<
                LineReprojectionConstantPoseCostFunctor>(
                camera.model_id, line2d, cam_from_rig * rig_from_world),
            line_loss, line_param.data.data(), camera.params.data());
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        problem_->AddResidualBlock(
            CreateLineCameraCostFunction<
                RigLineReprojErrorConstantRigCostFunctor>(camera.model_id,
                                                          line2d, cam_from_rig),
            line_loss, line_param.data.data(), rig_from_world.params.data(),
            camera.params.data());
      } else {
        problem_->AddResidualBlock(
            CreateLineCameraCostFunction<RigLineReprojErrorCostFunctor>(
                camera.model_id, line2d),
            line_loss, line_param.data.data(), cam_from_rig.params.data(),
            rig_from_world.params.data(), camera.params.data());
      }
    }
  }

  if (num_observations > 0) {
    parameterized_camera_ids_.insert(image.CameraId());
    parameterized_image_ids_.insert(image.ImageId());
  }
}

void BasePointLineBundleAdjuster::AddPointToProblem(
    colmap::point3D_t point3D_id, HolisticReconstruction &reconstruction) {
  colmap::Point3D &point3D = reconstruction.PointRecon().Point3D(point3D_id);

  size_t &num_observations = point3D_num_observations_[point3D_id];

  // Is 3D point already fully contained in the problem?
  if (num_observations == point3D.track.Length()) {
    return;
  }

  for (const auto &track_el : point3D.track.Elements()) {
    // Skip observations that were already added in AddImageToProblem
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    num_observations += 1;

    colmap::Image &image = reconstruction.PointRecon().Image(track_el.image_id);
    colmap::Camera &camera =
        reconstruction.PointRecon().Camera(image.CameraId());
    const colmap::Point2D &point2D = image.Point2D(track_el.point2D_idx);

    // Use CamFromWorld() by value - these images are not in the config,
    // so their poses are constant
    problem_->AddResidualBlock(
        colmap::CreateCameraCostFunction<
            colmap::ReprojErrorConstantPoseCostFunctor>(
            camera.model_id, point2D.xy, image.CamFromWorld()),
        loss_function_.get(), point3D.xyz.data(), camera.params.data());

    // Do not optimize intrinsics if the corresponding images were not included
    if (parameterized_camera_ids_.insert(image.CameraId()).second) {
      config_.SetConstantCamIntrinsics(image.CameraId());
    }
  }
}

void BasePointLineBundleAdjuster::AddLineToProblem(
    line3D_t line3D_id, HolisticReconstruction &reconstruction) {
  const auto &line3d = reconstruction.StructureRecon().Line(line3D_id);
  auto &line_param = line_params_.at(line3D_id);

  size_t &num_observations = line3D_num_observations_[line3D_id];

  // Is 3D line already fully contained in the problem?
  if (num_observations == line3d.track.Length()) {
    return;
  }

  for (const auto &track_el : line3d.track.Elements()) {
    // Skip observations that were already added in AddImageToProblem
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    colmap::Image &image = reconstruction.PointRecon().Image(track_el.image_id);

    num_observations += 1;
    colmap::Camera &camera =
        reconstruction.PointRecon().Camera(image.CameraId());

    // Get 2D line observation
    const Line2d &line2d = reconstruction.StructureRecon()
                               .Structure2d(track_el.image_id)
                               .Line(track_el.point2D_idx);

    // Use CamFromWorld() by value - these images are not in the config,
    // so their poses are constant
    ceres::LossFunction *line_loss = CreateLineLoss(line2d.Length());
    problem_->AddResidualBlock(
        CreateLineCameraCostFunction<LineReprojectionConstantPoseCostFunctor>(
            camera.model_id, line2d, image.CamFromWorld()),
        line_loss, line_param.data.data(), camera.params.data());

    // Do not optimize intrinsics if the corresponding images were not included
    if (parameterized_camera_ids_.insert(image.CameraId()).second) {
      config_.SetConstantCamIntrinsics(image.CameraId());
    }
  }
}

void BasePointLineBundleAdjuster::ParameterizeCameras(
    HolisticReconstruction &reconstruction) {
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;

  for (const colmap::camera_t camera_id : parameterized_camera_ids_) {
    colmap::Camera &camera = reconstruction.PointRecon().Camera(camera_id);

    if (constant_camera || config_.HasConstantCamIntrinsics(camera_id)) {
      problem_->SetParameterBlockConstant(camera.params.data());
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const auto params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const auto params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const auto params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        colmap::SetManifold(
            problem_.get(), camera.params.data(),
            colmap::CreateSubsetManifold(static_cast<int>(camera.params.size()),
                                         const_camera_params));
      }
    }
  }
}

void BasePointLineBundleAdjuster::ParameterizeImages(
    HolisticReconstruction &reconstruction) {
  FlatHashSet<colmap::rig_t> parameterized_rig_ids;
  FlatHashSet<colmap::sensor_t> parameterized_sensor_ids;
  FlatHashSet<colmap::frame_t> parameterized_frame_ids;

  for (const colmap::image_t image_id : parameterized_image_ids_) {
    colmap::Image &image = reconstruction.PointRecon().Image(image_id);
    parameterized_rig_ids.insert(image.FramePtr()->RigId());

    // Parameterize sensor_from_rig
    const colmap::sensor_t sensor_id = image.CameraPtr()->SensorId();
    const bool not_parameterized_before =
        parameterized_sensor_ids.insert(sensor_id).second;
    if (not_parameterized_before && !image.IsRefInFrame()) {
      colmap::Rigid3d &sensor_from_rig =
          image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
      // CostFunction assumes unit quaternions
      sensor_from_rig.rotation().normalize();
      if (problem_->HasParameterBlock(sensor_from_rig.params.data())) {
        colmap::SetManifold(problem_.get(), sensor_from_rig.params.data(),
                            colmap::CreateProductManifold(
                                colmap::CreateEigenQuaternionManifold(),
                                colmap::CreateEuclideanManifold<3>()));
        if (!options_.refine_sensor_from_rig ||
            config_.HasConstantSensorFromRigPose(sensor_id)) {
          problem_->SetParameterBlockConstant(sensor_from_rig.params.data());
        }
      }
    }

    // Parameterize rig_from_world
    if (parameterized_frame_ids.insert(image.FrameId()).second) {
      colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();
      // CostFunction assumes unit quaternions
      rig_from_world.rotation().normalize();
      if (problem_->HasParameterBlock(rig_from_world.params.data())) {
        colmap::SetManifold(problem_.get(), rig_from_world.params.data(),
                            colmap::CreateProductManifold(
                                colmap::CreateEigenQuaternionManifold(),
                                colmap::CreateEuclideanManifold<3>()));
        if (!options_.refine_rig_from_world ||
            config_.HasConstantRigFromWorldPose(image.FrameId())) {
          problem_->SetParameterBlockConstant(rig_from_world.params.data());
        }
      }
    }
  }

  // Set the rig poses as constant if the reference sensor is not part of the
  // problem. Otherwise, the relative pose between the sensors is not well
  // constrained.
  for (const colmap::rig_t rig_id : parameterized_rig_ids) {
    colmap::Rig &rig = reconstruction.PointRecon().Rig(rig_id);
    if (parameterized_sensor_ids.count(rig.RefSensorId()) != 0) {
      continue;
    }
    for (auto &[_, sensor_from_rig] : rig.NonRefSensors()) {
      if (sensor_from_rig.has_value() &&
          problem_->HasParameterBlock(sensor_from_rig->params.data())) {
        problem_->SetParameterBlockConstant(sensor_from_rig->params.data());
      }
    }
  }
}

void BasePointLineBundleAdjuster::ParameterizePoints(
    HolisticReconstruction &reconstruction) {
  for (const auto &[point3D_id, num_observations] : point3D_num_observations_) {
    colmap::Point3D &point3D = reconstruction.PointRecon().Point3D(point3D_id);
    // Set constant if:
    // 1. refine_points is false, OR
    // 2. point has more observations than added to the problem
    if (!base_options_.refine_points ||
        point3D.track.Length() > num_observations) {
      problem_->SetParameterBlockConstant(point3D.xyz.data());
    }
  }

  for (const colmap::point3D_t point3D_id : config_.ConstantPoints()) {
    colmap::Point3D &point3D = reconstruction.PointRecon().Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.xyz.data());
  }
}

void BasePointLineBundleAdjuster::ParameterizeLines() {
  for (const auto &[line3D_id, num_obs] : line3D_num_observations_) {
    auto &line_param = line_params_.at(line3D_id);

    if (problem_->HasParameterBlock(line_param.data.data())) {
      problem_->SetManifold(line_param.data.data(),
                            new MinimalInfiniteLine3dManifold());
    }

    if (!base_options_.refine_lines || HasConstantLine(line3D_id)) {
      problem_->SetParameterBlockConstant(line_param.data.data());
    }
  }
}

void BasePointLineBundleAdjuster::WriteBackLines(
    HolisticReconstruction &reconstruction) {
  for (const auto &[line_id, line_param] : line_params_) {
    InfiniteLine3d inf_line = line_param.GetInfiniteLine();

    // Get all 2D observations for this line
    Line3d &line3d = reconstruction.StructureRecon().Line(line_id);
    std::vector<colmap::Image> images;
    std::vector<Line2d> line2ds;

    for (const auto &track_el : line3d.track.Elements()) {
      const colmap::Image &image =
          reconstruction.PointRecon().Image(track_el.image_id);
      images.push_back(image);
      line2ds.push_back(reconstruction.StructureRecon()
                            .Structure2d(track_el.image_id)
                            .Line(track_el.point2D_idx));
    }

    // Re-estimate segment endpoints from the optimized infinite line
    const int num_outliers = std::min(2, static_cast<int>(images.size()) - 1);
    Line3d new_line3d = GetLineSegmentFromInfiniteLine3d(inf_line, images,
                                                         line2ds, num_outliers);
    if (new_line3d.Length() <= 0.0) {
      LOG(WARNING) << "WriteBackLines: line " << line_id
                   << " collapsed to zero length (num_images=" << images.size()
                   << ", num_outliers=" << num_outliers
                   << "), keeping old endpoints";
      continue;
    }
    line3d.start = new_line3d.start;
    line3d.end = new_line3d.end;
  }
}

// TODO: Move the base_options_.use_parameter_block_ordering check to the caller
// so this method unconditionally sets the ordering when called.
void BasePointLineBundleAdjuster::SetParameterBlockOrdering(
    ceres::Solver::Options &solver_options) {
  if (!base_options_.use_parameter_block_ordering) {
    return;
  }
  if (solver_options.linear_solver_type != ceres::DENSE_SCHUR &&
      solver_options.linear_solver_type != ceres::SPARSE_SCHUR &&
      solver_options.linear_solver_type != ceres::ITERATIVE_SCHUR) {
    return;
  }

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();

  // Group 0 (eliminated via Schur complement): point xyz + line params.
  // These form an independent set — no residual connects two landmarks.
  // Group params (VP, plane) go to group 1 because VP residuals connect
  // VP params with line params, violating the independent set requirement.
  FlatHashSet<double *> landmark_blocks;

  for (const auto &[point3D_id, _] : point3D_num_observations_) {
    double *ptr = reconstruction_.PointRecon().Point3D(point3D_id).xyz.data();
    if (problem_->HasParameterBlock(ptr)) {
      ordering->AddElementToGroup(ptr, 0);
      landmark_blocks.insert(ptr);
    }
  }
  for (auto &[line3D_id, line_param] : line_params_) {
    double *ptr = line_param.data.data();
    if (problem_->HasParameterBlock(ptr)) {
      ordering->AddElementToGroup(ptr, 0);
      landmark_blocks.insert(ptr);
    }
  }

  // Everything else (poses, cameras, group params) → group 1
  std::vector<double *> all_blocks;
  problem_->GetParameterBlocks(&all_blocks);
  for (double *ptr : all_blocks) {
    if (landmark_blocks.count(ptr) == 0) {
      ordering->AddElementToGroup(ptr, 1);
    }
  }

  VLOG(1) << "Schur ordering: group 0 (eliminate) = " << landmark_blocks.size()
          << " blocks (" << point3D_num_observations_.size() << " points + "
          << line_params_.size() << " lines), group 1 (keep) = "
          << (all_blocks.size() - landmark_blocks.size()) << " blocks";

  solver_options.linear_solver_ordering = std::move(ordering);
}

std::shared_ptr<colmap::BundleAdjustmentSummary>
BasePointLineBundleAdjuster::Solve() {
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
    // Disable Ceres's built-in function_tolerance to avoid double-triggering
    solver_options.function_tolerance = 0.0;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);

  // Write back optimized line parameters
  WriteBackLines(reconstruction_);

  if (options_.print_summary) {
    colmap::PrintSolverSummary(summary, "Point-line bundle adjustment report");
  }

  return colmap::CeresBundleAdjustmentSummary::Create(summary);
}

namespace {

// Concrete implementation of point-line bundle adjustment
class PointLineBundleAdjuster : public BasePointLineBundleAdjuster {
public:
  using BasePointLineBundleAdjuster::BasePointLineBundleAdjuster;
};

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<BasePointLineBundleAdjuster>
CreatePointLineBundleAdjuster(PointLineBundleAdjustmentOptions options,
                              PointLineBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction) {
  return std::make_unique<PointLineBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

std::unique_ptr<BasePointLineBundleAdjuster>
CreateLineBundleAdjuster(PointLineBundleAdjustmentOptions options,
                         PointLineBundleAdjustmentConfig config,
                         HolisticReconstruction &reconstruction) {
  options.disable_point_residuals = true;
  return std::make_unique<PointLineBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

std::unique_ptr<ceres::LossFunction>
CreateLossFunction(colmap::CeresBundleAdjustmentOptions::LossFunctionType type,
                   double scale) {
  using LossFunctionType =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType;
  switch (type) {
  case LossFunctionType::TRIVIAL:
    return nullptr;
  case LossFunctionType::SOFT_L1:
    return std::make_unique<ceres::SoftLOneLoss>(scale);
  case LossFunctionType::CAUCHY:
    return std::make_unique<ceres::CauchyLoss>(scale);
  case LossFunctionType::HUBER:
    return std::make_unique<ceres::HuberLoss>(scale);
  default:
    return nullptr;
  }
}

} // namespace estimators
} // namespace limap
