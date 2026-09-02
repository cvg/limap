#include "limap/sfm/structure_incremental_mapper.h"

#include <colmap/estimators/pose.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>

#include "limap/sfm/structure_observation_manager.h"

namespace limap {

bool StructureIncrementalMapper::Options::Check() const {
  CHECK_OPTION_GE(line_visibility_weight, 0);
  CHECK_OPTION(triangulation.Check());
  return true;
}

StructureIncrementalMapper::StructureIncrementalMapper(
    std::shared_ptr<const colmap::DatabaseCache> database_cache,
    std::shared_ptr<const StructureDatabaseCache> structure_db_cache)
    : database_cache_(std::move(database_cache)),
      structure_db_cache_(std::move(structure_db_cache)),
      colmap_mapper_(database_cache_) {}

// === Lifecycle ===

void StructureIncrementalMapper::BeginReconstruction(
    std::shared_ptr<HolisticReconstruction> reconstruction) {
  filtered_frames_.clear();
  num_reg_trials_.clear();
  num_structure_less_reg_trials_.clear();
  num_shared_reg_images_ = 0;
  reconstruction_ = std::move(reconstruction);

  // Aliasing shared_ptr: points to PointRecon(), shares lifetime with
  // reconstruction_
  auto point_recon_ptr = std::shared_ptr<colmap::Reconstruction>(
      reconstruction_, &reconstruction_->PointRecon());
  colmap_mapper_.BeginReconstruction(point_recon_ptr);

  // Register any already-existing frames (for continuing from a previous
  // reconstruction). Mirrors COLMAP's BeginReconstruction logic.
  for (const colmap::frame_t frame_id :
       reconstruction_->PointRecon().RegFrameIds()) {
    RegisterFrameEvent(frame_id);
  }

  // Load structure data and create triangulator
  reconstruction_->StructureRecon().Load(*structure_db_cache_);
  reconstruction_->StructureRecon().InitializeAllWireframes();
  structure_corr_graph_ = structure_db_cache_->StructureCorrespondenceGraph();
  // Non-owning shared_ptr to the COLMAP mapper's ObservationManager.
  // The structure's point triangulator must use the same ObservationManager
  // so that pair statistics (num_tri_corrs) stay balanced when points are
  // created during triangulation and later removed during filtering.
  auto point_obs_manager = std::shared_ptr<colmap::ObservationManager>(
      &colmap_mapper_.ObservationManager(),
      [](colmap::ObservationManager *) {});
  structure_triangulator_ = std::make_unique<IncrementalStructureTriangulator>(
      database_cache_->CorrespondenceGraph(), structure_corr_graph_,
      *reconstruction_, std::move(point_obs_manager));
}

void StructureIncrementalMapper::EndReconstruction(bool discard) {
  structure_triangulator_.reset();
  structure_corr_graph_.reset();
  colmap_mapper_.EndReconstruction(discard);
  if (discard) {
    reconstruction_.reset();
  }
}

// === Initialization (pure delegation) ===

bool StructureIncrementalMapper::FindInitialImagePair(
    const colmap::IncrementalMapper::Options &options,
    colmap::image_t &image_id1, colmap::image_t &image_id2,
    colmap::Rigid3d &cam2_from_cam1) {
  return colmap_mapper_.FindInitialImagePair(options, image_id1, image_id2,
                                             cam2_from_cam1);
}

void StructureIncrementalMapper::RegisterInitialImagePair(
    const colmap::IncrementalMapper::Options &options,
    colmap::image_t image_id1, colmap::image_t image_id2,
    const colmap::Rigid3d &cam2_from_cam1) {
  colmap_mapper_.RegisterInitialImagePair(options, image_id1, image_id2,
                                          cam2_from_cam1);
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;
  auto &point_recon = reconstruction_->PointRecon();
  RegisterFrameEvent(point_recon.Image(image_id1).FrameId());
  RegisterFrameEvent(point_recon.Image(image_id2).FrameId());
}

bool StructureIncrementalMapper::EstimateInitialTwoViewGeometry(
    const colmap::IncrementalMapper::Options &options,
    colmap::image_t image_id1, colmap::image_t image_id2,
    colmap::Rigid3d &cam2_from_cam1) {
  return colmap_mapper_.EstimateInitialTwoViewGeometry(
      options, image_id1, image_id2, cam2_from_cam1);
}

// === Image selection ===

std::vector<colmap::image_t> StructureIncrementalMapper::FindNextImages(
    const colmap::IncrementalMapper::Options &options,
    const Options &structure_options, bool structure_less) {
  // Get candidate images from COLMAP
  std::vector<colmap::image_t> candidates =
      colmap_mapper_.FindNextImages(options, structure_less);

  if (candidates.empty()) {
    return candidates;
  }

  const auto &point_recon = reconstruction_->PointRecon();
  const size_t max_reg_trials = static_cast<size_t>(options.max_reg_trials);

  // Select the appropriate trial counter (regular vs structure-less have
  // independent budgets, matching COLMAP's separated counters).
  auto &trial_counter =
      structure_less ? num_structure_less_reg_trials_ : num_reg_trials_;

  // Score and bucket candidates (matching COLMAP's two-bucket approach):
  // - Bucket 1: fresh candidates (not filtered, not previously tried)
  // - Bucket 2: filtered or previously tried candidates
  // - Excluded: images exceeding max_reg_trials
  struct ScoredImage {
    colmap::image_t image_id;
    double score;
  };
  std::vector<ScoredImage> primary;
  std::vector<ScoredImage> deprioritized;

  for (const colmap::image_t image_id : candidates) {
    // Hard exclude images that exceeded max registration trials
    const size_t trials = trial_counter[image_id];
    if (trials >= max_reg_trials) {
      continue;
    }

    double score;
    if (structure_less) {
      // Structure-less: rank by visible correspondences (matching COLMAP)
      score = static_cast<double>(
          colmap_mapper_.ObservationManager().NumVisibleCorrespondences(
              image_id));
    } else {
      // Structure-based: rank by visible 3D points + lines
      score = static_cast<double>(
          colmap_mapper_.ObservationManager().NumVisiblePoints3D(image_id));
      if (structure_options.line_visibility_weight > 0.0) {
        const size_t num_visible_lines = NumVisibleLines3D(image_id);
        score += structure_options.line_visibility_weight *
                 static_cast<double>(num_visible_lines);
      }
    }

    // Deprioritize filtered or previously tried images
    const auto &image = point_recon.Image(image_id);
    const bool is_filtered = filtered_frames_.count(image.FrameId()) > 0;
    if (is_filtered || trials > 0) {
      deprioritized.push_back({image_id, score});
    } else {
      primary.push_back({image_id, score});
    }
  }

  auto cmp = [](const ScoredImage &a, const ScoredImage &b) {
    return a.score > b.score;
  };
  std::sort(primary.begin(), primary.end(), cmp);
  std::sort(deprioritized.begin(), deprioritized.end(), cmp);

  // Primary bucket first, then deprioritized
  std::vector<colmap::image_t> result;
  result.reserve(primary.size() + deprioritized.size());
  for (const auto &s : primary) {
    result.push_back(s.image_id);
  }
  for (const auto &s : deprioritized) {
    result.push_back(s.image_id);
  }
  return result;
}

// === Registration ===

bool StructureIncrementalMapper::RegisterNextStructureLessImage(
    const colmap::IncrementalMapper::Options &mapper_options,
    colmap::image_t image_id) {
  num_structure_less_reg_trials_[image_id] += 1;
  return colmap_mapper_.RegisterNextStructureLessImage(mapper_options,
                                                       image_id);
}

bool StructureIncrementalMapper::RegisterNextImage(
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options, colmap::image_t image_id) {
  THROW_CHECK(reconstruction_);
  auto &point_recon = reconstruction_->PointRecon();
  THROW_CHECK_GT(point_recon.NumRegFrames(), 0);

  num_reg_trials_[image_id] += 1;

  colmap::Image &image = point_recon.Image(image_id);
  colmap::Camera &camera = *image.CameraPtr();

  // Gather point 2D-3D correspondences (same logic as COLMAP)
  std::vector<std::pair<colmap::point2D_t, colmap::point3D_t>> tri_corrs;
  std::vector<V2D> points2d;
  std::vector<V3D> points3d;

  const auto corr_graph = database_cache_->CorrespondenceGraph();
  FlatHashSet<colmap::point3D_t> corr_point3D_ids;

  for (colmap::point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const auto &point2D = image.Point2D(point2D_idx);

    corr_point3D_ids.clear();
    const auto corr_range =
        corr_graph->FindCorrespondences(image_id, point2D_idx);
    for (const auto *corr = corr_range.beg; corr < corr_range.end; ++corr) {
      const auto &corr_image = point_recon.Image(corr->image_id);
      if (!corr_image.HasPose()) {
        continue;
      }
      const auto &corr_point2D = corr_image.Point2D(corr->point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }
      if (corr_point3D_ids.count(corr_point2D.point3D_id) > 0) {
        continue;
      }
      if (corr_image.CameraPtr()->HasBogusParams(
              mapper_options.min_focal_length_ratio,
              mapper_options.max_focal_length_ratio,
              mapper_options.max_extra_param)) {
        continue;
      }
      const auto &point3D = point_recon.Point3D(corr_point2D.point3D_id);
      tri_corrs.emplace_back(point2D_idx, corr_point2D.point3D_id);
      corr_point3D_ids.insert(corr_point2D.point3D_id);
      points2d.push_back(point2D.xy);
      points3d.push_back(point3D.xyz);
    }
  }

  // Gather line 2D-3D correspondences
  std::vector<Line3d> lines3d;
  std::vector<Line2d> lines2d;
  if (options.use_lines_for_registration) {
    FindLineCorrespondences(image_id, lines3d, lines2d);
  }

  // Check minimum correspondences (points + lines combined)
  const size_t num_point_corrs = points2d.size();
  const size_t num_line_corrs = lines2d.size();
  if (num_point_corrs + num_line_corrs <
      static_cast<size_t>(mapper_options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Insufficient correspondences for hybrid registration ("
            << num_point_corrs << " points + " << num_line_corrs << " lines < "
            << mapper_options.abs_pose_min_num_inliers << ")";
    return false;
  }

  // Whether to solve for the focal length too, decided from the state of the
  // camera as COLMAP's IncrementalMapper::RegisterNextImage does: a camera
  // with no focal prior gets one estimated, but only while it is still
  // unconstrained, so that a camera shared by several images is not
  // re-estimated (and degraded) once other images have pinned it down.
  // Turn it off through mapper_options.abs_pose_refine_focal_length, the same
  // knob as COLMAP. options.registration.estimate_focal_length is
  // deliberately not consulted: requiring it would leave an uncalibrated
  // reconstruction silently unable to recover its focal lengths.
  auto registration_options = options.registration;
  {
    const bool constant_camera =
        mapper_options.constant_cameras.count(image.CameraId()) > 0;
    const bool bogus_params = camera.HasBogusParams(
        mapper_options.min_focal_length_ratio,
        mapper_options.max_focal_length_ratio, mapper_options.max_extra_param);
    // Other images already registered against this camera have constrained
    // it, unless its parameters have gone bogus.
    const auto &point_recon = reconstruction_->PointRecon();
    bool camera_constrained = false;
    for (const colmap::image_t reg_id : point_recon.RegImageIds()) {
      if (reg_id != image_id &&
          point_recon.Image(reg_id).CameraId() == image.CameraId()) {
        camera_constrained = true;
        break;
      }
    }
    registration_options.estimate_focal_length =
        !constant_camera && !camera.has_prior_focal_length &&
        mapper_options.abs_pose_refine_focal_length &&
        (!camera_constrained || bogus_params);
  }

  // Without lines there is nothing for the hybrid estimator to add, so
  // solving for the focal is COLMAP's job: EstimateAbsolutePose runs P4Pf
  // RANSAC and writes the focal into the camera itself. Deferring to it
  // keeps the points-only path identical to COLMAP's, exactly as the point
  // side of the pipeline does elsewhere.
  if (registration_options.estimate_focal_length && lines2d.empty()) {
    colmap::AbsolutePoseEstimationOptions abs_options;
    abs_options.estimate_focal_length = true;
    abs_options.ransac_options.max_error = registration_options.max_error_point;
    abs_options.ransac_options.min_num_trials =
        registration_options.min_iterations;
    abs_options.ransac_options.max_num_trials =
        registration_options.max_iterations;
    abs_options.ransac_options.confidence = registration_options.success_prob;
    if (!registration_options.random_seed) {
      abs_options.ransac_options.random_seed = registration_options.seed;
    }

    estimators::absolute_pose::PointLineAbsolutePoseResult result;
    size_t num_inliers = 0;
    const bool success = colmap::EstimateAbsolutePose(
        abs_options, points2d, points3d, &result.pose, &camera, &num_inliers,
        &result.inliers_points);
    if (!success || static_cast<int>(num_inliers) <
                        mapper_options.abs_pose_min_num_inliers) {
      VLOG(2) << "PnPF registration failed or too few inliers (" << num_inliers
              << ")";
      return false;
    }
    result.num_inliers = num_inliers;
    result.num_inliers_points = num_inliers;
    result.camera = camera;
    result.success = true;
    return FinishRegistration(image, result, tri_corrs);
  }

  // Hybrid PnPL estimation
  auto result = estimators::absolute_pose::EstimatePointLineAbsolutePose(
      lines3d, lines2d, points3d, points2d, camera, registration_options);

  if (!result.success) {
    VLOG(2) << "Hybrid absolute pose estimation failed";
    return false;
  }

  if (registration_options.estimate_focal_length) {
    THROW_CHECK(result.camera.has_value());
    // Copy per focal parameter rather than through FocalLength(), which is
    // only defined for models with a single one (PINHOLE has fx and fy).
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] = result.camera->params[idx];
    }
  }

  if (static_cast<int>(result.num_inliers) <
      mapper_options.abs_pose_min_num_inliers) {
    VLOG(2) << "Hybrid pose estimation: insufficient inliers ("
            << result.num_inliers << " < "
            << mapper_options.abs_pose_min_num_inliers << ")";
    return false;
  }

  return FinishRegistration(image, result, tri_corrs);
}

bool StructureIncrementalMapper::FinishRegistration(
    colmap::Image &image,
    const estimators::absolute_pose::PointLineAbsolutePoseResult &result,
    const std::vector<std::pair<colmap::point2D_t, colmap::point3D_t>>
        &tri_corrs) {
  const colmap::image_t image_id = image.ImageId();

  // Set pose on the frame, then register it with the reconstruction.
  // ObservationManager::RegisterFrame does two things:
  //   1. Updates num_visible_correspondences for other images (needed for
  //      structure-less fallback ranking in FindNextImages).
  //   2. Calls Reconstruction::RegisterFrame which adds the frame to
  //      reg_frame_ids_, making NumRegFrames()/NumRegImages() correct.
  // Without this, RegFrameIds() would only contain the initial pair, and
  // global BA / filtering / reporting would miss all subsequently registered
  // images.
  //
  image.FramePtr()->SetCamFromWorld(image.CameraId(), result.pose);
  auto &obs_manager = colmap_mapper_.ObservationManager();
  obs_manager.RegisterFrame(image.FrameId());
  RegisterFrameEvent(image.FrameId());

  // Continue point tracks for inlier correspondences.
  // This mirrors COLMAP's RegisterNextImage post-estimation logic.
  for (size_t i = 0; i < result.inliers_points.size(); ++i) {
    if (result.inliers_points[i]) {
      const auto [point2D_idx, point3D_id] = tri_corrs[i];
      const auto &point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const colmap::TrackElement track_el(image_id, point2D_idx);
        obs_manager.AddObservation(point3D_id, track_el);
        colmap_mapper_.Triangulator().AddModifiedPoint3D(point3D_id);
      }
    }
  }

  // Line track continuation is handled by TriangulateImage (which is called
  // right after registration in the pipeline). The incremental line
  // triangulator's Continue step will add this image's 2D lines to existing
  // 3D line tracks.

  LOG(INFO) << "  => Hybrid registration: " << result.num_inliers_points
            << " point inliers, " << result.num_inliers_lines
            << " line inliers";

  return true;
}

// === Triangulation ===

size_t StructureIncrementalMapper::TriangulateImage(const Options &options,
                                                    colmap::image_t image_id) {
  THROW_CHECK(structure_triangulator_);
  return structure_triangulator_->TriangulateImage(options.triangulation,
                                                   image_id);
}

// === Bundle Adjustment ===

StructureIncrementalMapper::LocalBundleAdjustmentReport
StructureIncrementalMapper::AdjustLocalBundle(
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options, colmap::image_t image_id) {
  THROW_CHECK(reconstruction_);
  LocalBundleAdjustmentReport report;

  // Find local bundle (same logic as COLMAP)
  const std::vector<colmap::image_t> local_bundle =
      colmap_mapper_.FindLocalBundle(mapper_options, image_id);

  if (local_bundle.empty()) {
    return report;
  }

  auto &point_recon = reconstruction_->PointRecon();
  auto &srec = reconstruction_->StructureRecon();

  // Build structure BA config
  estimators::StructureBundleAdjustmentConfig ba_config;
  ba_config.FixGauge(colmap::BundleAdjustmentGauge::THREE_POINTS);

  // Add images from local bundle
  std::set<colmap::frame_t> frame_ids;
  const auto &image = point_recon.Image(image_id);
  frame_ids.insert(image.FrameId());
  for (const auto &data_id : image.FramePtr()->ImageIds()) {
    ba_config.AddImage(data_id.id);
  }
  for (const colmap::image_t local_image_id : local_bundle) {
    const auto &local_image = point_recon.Image(local_image_id);
    frame_ids.insert(local_image.FrameId());
    for (const auto &data_id : local_image.FramePtr()->ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  // Fix existing frames if specified
  if (mapper_options.fix_existing_frames) {
    for (const colmap::frame_t frame_id : frame_ids) {
      if (colmap_mapper_.ExistingFrameIds().count(frame_id)) {
        ba_config.SetConstantRigFromWorldPose(frame_id);
      }
    }
  }

  // Fix rig poses if not all frames within local bundle.
  // Compute total counts from the reconstruction directly since
  // RegisterFrameEvent (which updates colmap_mapper_ reg_stats_) is private.
  FlatHashMap<colmap::rig_t, size_t> num_frames_per_rig;
  for (const colmap::frame_t frame_id : frame_ids) {
    const auto &frame = point_recon.Frame(frame_id);
    num_frames_per_rig[frame.RigId()] += 1;
  }
  FlatHashMap<colmap::rig_t, size_t> total_reg_frames_per_rig;
  for (const colmap::frame_t fid : point_recon.RegFrameIds()) {
    total_reg_frames_per_rig[point_recon.Frame(fid).RigId()] += 1;
  }
  for (const auto &[rig_id, num_frames] : num_frames_per_rig) {
    auto it = total_reg_frames_per_rig.find(rig_id);
    if (mapper_options.constant_rigs.count(rig_id) ||
        (it != total_reg_frames_per_rig.end() && num_frames < it->second)) {
      const auto &rig = point_recon.Rig(rig_id);
      for (const auto &[sensor_id, _] : rig.NonRefSensors()) {
        ba_config.SetConstantSensorFromRigPose(sensor_id);
      }
    }
  }

  // Fix camera intrinsics if not all registered images within local bundle.
  FlatHashMap<colmap::camera_t, size_t> num_images_per_camera;
  for (const colmap::image_t img_id : ba_config.Images()) {
    const auto &img = point_recon.Image(img_id);
    num_images_per_camera[img.CameraId()] += 1;
  }
  FlatHashMap<colmap::camera_t, size_t> total_reg_images_per_camera;
  for (const colmap::image_t img_id : point_recon.RegImageIds()) {
    total_reg_images_per_camera[point_recon.Image(img_id).CameraId()] += 1;
  }
  for (const auto &[camera_id, num_images] : num_images_per_camera) {
    auto it = total_reg_images_per_camera.find(camera_id);
    if (mapper_options.constant_cameras.count(camera_id) ||
        (it != total_reg_images_per_camera.end() && num_images < it->second)) {
      ba_config.SetConstantCamIntrinsics(camera_id);
    }
  }

  // Add new/short-track 3D points as variable (same as COLMAP)
  const auto &modified_points = colmap_mapper_.GetModifiedPoints3D();
  for (const colmap::point3D_t point3D_id : modified_points) {
    if (point_recon.ExistsPoint3D(point3D_id)) {
      const auto &point3D = point_recon.Point3D(point3D_id);
      constexpr size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.track.Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
      }
    }
  }

  FlatHashSet<colmap::image_t> local_image_set(ba_config.Images().begin(),
                                               ba_config.Images().end());

  // Use modified lines, filtered by track length. This matches COLMAP's
  // kMaxTrackLength filter for points: long-track lines are already stable
  // and including them in local BA and merge/complete would slow down
  // processing significantly without meaningful quality improvement.
  FlatHashSet<line3D_t> local_line_ids;
  if (structure_triangulator_) {
    constexpr size_t kMaxTrackLength = 15;
    for (const line3D_t lid : structure_triangulator_->GetModifiedLines3D()) {
      if (srec.ExistsLine3D(lid)) {
        const auto &line3d = srec.Line(lid);
        if (line3d.track.Length() <= kMaxTrackLength) {
          local_line_ids.insert(lid);
        }
      }
    }
  }

  // Classify local lines into reliable/unreliable for 2-step BA
  FlatHashSet<line3D_t> reliable_line_ids;
  FlatHashSet<line3D_t> unreliable_line_ids;
  if (options.use_two_step_ba) {
    for (const line3D_t lid : local_line_ids) {
      if (srec.Line(lid).IsReliable(options.min_active_line_observations)) {
        reliable_line_ids.insert(lid);
      } else {
        unreliable_line_ids.insert(lid);
      }
    }
  } else {
    reliable_line_ids = local_line_ids;
  }

  // Step 1: BA with points + reliable lines + groups + wireframes
  for (const line3D_t lid : reliable_line_ids) {
    ba_config.AddVariableLine(lid);
  }

  // Add variable groups visible in local bundle
  if (options.structure_ba.refine_groups) {
    for (const auto &[group3D_id, group3d] : srec.Groups3D()) {
      bool in_local = false;
      for (const auto &elem : group3d.track.Elements()) {
        if (local_image_set.count(elem.image_id)) {
          in_local = true;
          break;
        }
      }
      if (in_local) {
        ba_config.AddVariableGroup(group3D_id);
      }
    }
  }

  // Save image IDs and frame IDs before moving ba_config
  const FlatHashSet<colmap::image_t> ba_image_ids(ba_config.Images().begin(),
                                                  ba_config.Images().end());

  // Solve Step 1 structure BA
  VLOG(1) << "  2-step local BA Step 1: " << reliable_line_ids.size()
          << " reliable lines";
  auto ba_options = options.structure_ba;
  if (VLOG_IS_ON(2)) {
    ba_options.ceres->solver_options.logging_type =
        ceres::PER_MINIMIZER_ITERATION;
  }
  auto adjuster = estimators::CreateStructureBundleAdjuster(
      ba_options, std::move(ba_config), *reconstruction_);
  const auto summary = adjuster->Solve();
  report.num_adjusted_observations = summary->num_residuals / 2;

  // Step 2: Fixed-pose BA for unreliable lines (point-line only, no
  // groups/wireframes)
  if (options.use_two_step_ba && !unreliable_line_ids.empty()) {
    estimators::PointLineBundleAdjustmentConfig ba_config_step2;
    ba_config_step2.FixGauge(colmap::BundleAdjustmentGauge::THREE_POINTS);

    // Add all local images, but fix all poses
    for (const colmap::image_t img_id : ba_image_ids) {
      ba_config_step2.AddImage(img_id);
    }
    for (const colmap::frame_t fid : frame_ids) {
      ba_config_step2.SetConstantRigFromWorldPose(fid);
    }

    // Fix all camera intrinsics
    for (const colmap::image_t img_id : ba_image_ids) {
      ba_config_step2.SetConstantCamIntrinsics(
          point_recon.Image(img_id).CameraId());
    }

    // Add unreliable lines
    for (const line3D_t lid : unreliable_line_ids) {
      if (srec.ExistsLine3D(lid)) {
        ba_config_step2.AddVariableLine(lid);
      }
    }

    // Step 2: only refine lines, everything else fixed
    estimators::PointLineBundleAdjustmentOptions ba_options_step2(
        options.structure_ba);
    ba_options_step2.refine_rig_from_world = false;
    ba_options_step2.refine_sensor_from_rig = false;
    ba_options_step2.refine_focal_length = false;
    ba_options_step2.refine_principal_point = false;
    ba_options_step2.refine_extra_params = false;
    ba_options_step2.refine_points = false;
    ba_options_step2.print_summary = false;

    VLOG(1) << "  2-step local BA Step 2: " << unreliable_line_ids.size()
            << " unreliable lines";
    auto adjuster2 = estimators::CreateLineBundleAdjuster(
        ba_options_step2, std::move(ba_config_step2), *reconstruction_);
    adjuster2->Solve();
  }

  // Post-BA: merge/complete point tracks for modified points
  const auto &tri_options = options.triangulation.point_options;
  FlatHashSet<colmap::point3D_t> variable_point3D_ids(modified_points.begin(),
                                                      modified_points.end());

  auto &triangulator = colmap_mapper_.Triangulator();
  report.num_merged_observations =
      triangulator.MergeTracks(tri_options, variable_point3D_ids);
  report.num_completed_observations =
      triangulator.CompleteTracks(tri_options, variable_point3D_ids);
  report.num_completed_observations +=
      triangulator.CompleteImage(tri_options, image_id);

  // Post-BA: merge/complete structure tracks (scoped to modified lines,
  // matching COLMAP's approach for points which uses variable_point3D_ids)
  if (options.triangulation.triangulate_lines && structure_triangulator_) {
    report.num_merged_line_observations =
        structure_triangulator_->LineTriangulator().MergeTracks(
            options.triangulation.line_options, local_line_ids);
    report.num_completed_line_observations =
        structure_triangulator_->LineTriangulator().CompleteTracks(
            options.triangulation.line_options, local_line_ids);
    report.num_completed_line_observations +=
        structure_triangulator_->LineTriangulator().CompleteImage(
            options.triangulation.line_options, image_id);
  }
  if (options.triangulation.triangulate_groups) {
    report.num_merged_group_observations =
        structure_triangulator_->GroupTriangulator().MergeAllTracks(
            options.triangulation.group_options);
    report.num_completed_group_observations =
        structure_triangulator_->GroupTriangulator().CompleteImage(
            options.triangulation.group_options, image_id);
  }

  // Filter points
  auto &obs_manager = colmap_mapper_.ObservationManager();
  report.num_filtered_observations = obs_manager.FilterPoints3DInImages(
      mapper_options.filter_max_reproj_error,
      mapper_options.filter_min_tri_angle, ba_image_ids);
  report.num_filtered_observations += obs_manager.FilterPoints3D(
      mapper_options.filter_max_reproj_error,
      mapper_options.filter_min_tri_angle, modified_points);

  return report;
}

bool StructureIncrementalMapper::AdjustGlobalBundle(
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options) {
  THROW_CHECK(reconstruction_);

  auto &point_recon = reconstruction_->PointRecon();
  auto &srec = reconstruction_->StructureRecon();

  auto ba_options = options.structure_ba;

  // Filter observations with negative depth
  colmap_mapper_.ObservationManager().FilterObservationsWithNegativeDepth();

  // Build config
  estimators::StructureBundleAdjustmentConfig ba_config;
  for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
    const auto &frame = point_recon.Frame(frame_id);
    for (const auto &data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  THROW_CHECK_GE(ba_config.NumImages(), 2)
      << "At least two images must be registered for global BA";

  // Fix existing frames if specified
  if (mapper_options.fix_existing_frames) {
    for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
      if (colmap_mapper_.ExistingFrameIds().count(frame_id)) {
        ba_config.SetConstantRigFromWorldPose(frame_id);
      }
    }
  }

  for (const auto &rig_id : mapper_options.constant_rigs) {
    const auto &rig = point_recon.Rig(rig_id);
    for (const auto &[sensor_id, _] : rig.NonRefSensors()) {
      ba_config.SetConstantSensorFromRigPose(sensor_id);
    }
  }

  for (const auto &camera_id : mapper_options.constant_cameras) {
    ba_config.SetConstantCamIntrinsics(camera_id);
  }

  // Gauge fixing
  ba_config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  // Add all lines as variable
  for (const auto &[line3D_id, _] : srec.Lines3D()) {
    ba_config.AddVariableLine(line3D_id);
  }

  // Add all groups — BA layer filters by min_active_group_associations
  if (ba_options.refine_groups) {
    for (const auto &[group3D_id, _] : srec.Groups3D()) {
      ba_config.AddVariableGroup(group3D_id);
    }
  }

  if (VLOG_IS_ON(2)) {
    ba_options.ceres->solver_options.logging_type =
        ceres::PER_MINIMIZER_ITERATION;
  }
  auto adjuster = estimators::CreateStructureBundleAdjuster(
      ba_options, std::move(ba_config), *reconstruction_);
  if (!adjuster->Solve()->IsSolutionUsable()) {
    return false;
  }

  // Post-BA: soft-filter group associations + delete supportless groups
  FilterGroupAssociations(*reconstruction_, options.group_verification);
  DeleteSupportlessGroups(*reconstruction_);

  return true;
}

void StructureIncrementalMapper::AdjustGlobalBundleTwoStep(
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options, const FlatHashSet<line3D_t> &reliable_ids,
    const FlatHashSet<line3D_t> &unreliable_ids) {
  THROW_CHECK(reconstruction_);

  auto &point_recon = reconstruction_->PointRecon();
  auto &srec = reconstruction_->StructureRecon();

  auto ba_options = options.structure_ba;

  // Filter observations with negative depth
  colmap_mapper_.ObservationManager().FilterObservationsWithNegativeDepth();

  // ---- Step 1: points + reliable lines + groups + wireframes ----
  {
    estimators::StructureBundleAdjustmentConfig ba_config;
    for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
      const auto &frame = point_recon.Frame(frame_id);
      for (const auto &data_id : frame.ImageIds()) {
        ba_config.AddImage(data_id.id);
      }
    }

    THROW_CHECK_GE(ba_config.NumImages(), 2)
        << "At least two images must be registered for global BA";

    // Fix existing frames if specified
    if (mapper_options.fix_existing_frames) {
      for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
        if (colmap_mapper_.ExistingFrameIds().count(frame_id)) {
          ba_config.SetConstantRigFromWorldPose(frame_id);
        }
      }
    }

    for (const auto &rig_id : mapper_options.constant_rigs) {
      const auto &rig = point_recon.Rig(rig_id);
      for (const auto &[sensor_id, _] : rig.NonRefSensors()) {
        ba_config.SetConstantSensorFromRigPose(sensor_id);
      }
    }

    for (const auto &camera_id : mapper_options.constant_cameras) {
      ba_config.SetConstantCamIntrinsics(camera_id);
    }

    ba_config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

    // Add only reliable lines
    for (const line3D_t lid : reliable_ids) {
      if (srec.ExistsLine3D(lid)) {
        ba_config.AddVariableLine(lid);
      }
    }

    // Add all groups — BA layer filters by min_active_group_associations
    if (ba_options.refine_groups) {
      for (const auto &[group3D_id, _] : srec.Groups3D()) {
        ba_config.AddVariableGroup(group3D_id);
      }
    }

    LOG(INFO) << "  2-step global BA Step 1: " << reliable_ids.size()
              << " reliable lines";
    if (VLOG_IS_ON(2)) {
      ba_options.ceres->solver_options.logging_type =
          ceres::PER_MINIMIZER_ITERATION;
    }
    auto adjuster = estimators::CreateStructureBundleAdjuster(
        ba_options, std::move(ba_config), *reconstruction_);
    adjuster->Solve();
  }

  // Post-Step1: soft-filter group associations + delete supportless groups
  FilterGroupAssociations(*reconstruction_, options.group_verification);
  DeleteSupportlessGroups(*reconstruction_);

  // ---- Step 2: unreliable lines with structure constraints ----
  if (!unreliable_ids.empty()) {
    for (const line3D_t lid : unreliable_ids) {
      if (srec.ExistsLine3D(lid)) {
        srec.Line(lid).ClearInactiveLabels();
      }
    }

    estimators::StructureBundleAdjustmentConfig ba_config_step2;

    for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
      const auto &frame = point_recon.Frame(frame_id);
      for (const auto &data_id : frame.ImageIds()) {
        ba_config_step2.AddImage(data_id.id);
      }
      ba_config_step2.SetConstantRigFromWorldPose(frame_id);
    }
    for (const colmap::image_t img_id : point_recon.RegImageIds()) {
      ba_config_step2.SetConstantCamIntrinsics(
          point_recon.Image(img_id).CameraId());
    }
    ba_config_step2.FixGauge(
        colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

    // Only unreliable lines — everything else automatically skipped
    for (const line3D_t lid : unreliable_ids) {
      if (srec.ExistsLine3D(lid))
        ba_config_step2.AddVariableLine(lid);
    }
    // Only include groups with enough reliable associations in step 2.
    size_t num_step2_groups = 0;
    if (ba_options.refine_groups) {
      const size_t min_reliable =
          ba_options.min_reliable_group_associations_step2;
      for (const auto &[gid, group] : srec.Groups3D()) {
        size_t n_reliable = group.CountActivePoints();
        for (const auto &assoc : group.lines) {
          if (reliable_ids.count(assoc.idx))
            n_reliable++;
        }
        if (n_reliable >= min_reliable) {
          ba_config_step2.AddConstantGroup(gid);
          ++num_step2_groups;
        }
      }
    }

    estimators::StructureBundleAdjustmentOptions ba_opts_step2(ba_options);
    ba_opts_step2.refine_rig_from_world = false;
    ba_opts_step2.refine_sensor_from_rig = false;
    ba_opts_step2.refine_focal_length = false;
    ba_opts_step2.refine_principal_point = false;
    ba_opts_step2.refine_extra_params = false;
    ba_opts_step2.refine_points = false;
    ba_opts_step2.refine_groups = false;
    ba_opts_step2.disable_point_residuals = true;
    ba_opts_step2.force_reconstruct_wireframe3d = false;

    LOG(INFO) << "  2-step global BA Step 2: " << unreliable_ids.size()
              << " unreliable lines, " << num_step2_groups << "/"
              << srec.NumGroups3D() << " groups (min_reliable="
              << ba_options.min_reliable_group_associations_step2 << ")";
    auto adjuster2 = estimators::CreateStructureBundleAdjuster(
        ba_opts_step2, std::move(ba_config_step2), *reconstruction_);
    adjuster2->Solve();
  }
}

void StructureIncrementalMapper::IterativeLocalRefinement(
    int max_num_refinements, double max_refinement_change,
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options, colmap::image_t image_id) {
  auto opts = options;
  // Groups and wireframe only in global BA — local bundle has too few views for
  // reliable voting and the constraints can be noisy.
  opts.structure_ba.refine_groups = false;
  opts.triangulation.triangulate_groups = false;
  opts.structure_ba.refine_wireframe = false;
  StructureObservationManager struct_obs_manager(
      reconstruction_->StructureRecon());

  // Compute local image set (scope filtering to local bundle only)
  auto &point_recon = reconstruction_->PointRecon();
  const std::vector<colmap::image_t> local_bundle =
      colmap_mapper_.FindLocalBundle(mapper_options, image_id);
  FlatHashSet<colmap::image_t> local_image_ids;
  const auto &image = point_recon.Image(image_id);
  for (const auto &data_id : image.FramePtr()->ImageIds()) {
    local_image_ids.insert(data_id.id);
  }
  for (const colmap::image_t local_image_id : local_bundle) {
    const auto &local_image = point_recon.Image(local_image_id);
    for (const auto &data_id : local_image.FramePtr()->ImageIds()) {
      local_image_ids.insert(data_id.id);
    }
  }

  for (int i = 0; i < max_num_refinements; ++i) {
    const auto report = AdjustLocalBundle(mapper_options, opts, image_id);
    VLOG(1) << "=> Merged observations: " << report.num_merged_observations;
    VLOG(1) << "=> Completed observations: "
            << report.num_completed_observations;
    VLOG(1) << "=> Filtered observations: " << report.num_filtered_observations;
    VLOG(1) << "=> Merged lines: " << report.num_merged_line_observations
            << ", Completed lines: " << report.num_completed_line_observations;
    VLOG(1) << "=> Merged groups: " << report.num_merged_group_observations
            << ", Completed groups: "
            << report.num_completed_group_observations;

    // Post-BA: filter line tracks (scoped to local bundle images)
    const size_t num_deleted_lines = struct_obs_manager.FilterLineTracks(
        opts.filter_max_line_angular_error, opts.filter_max_line_perp_error, 0,
        &local_image_ids);
    VLOG(1) << "=> Deleted line tracks: " << num_deleted_lines;

    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    VLOG(1) << colmap::StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration
    opts.structure_ba.ceres->loss_function_type =
        colmap::CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  colmap_mapper_.ClearModifiedPoints3D();
  if (structure_triangulator_) {
    structure_triangulator_->ClearModified();
  }
}

void StructureIncrementalMapper::IterativeGlobalRefinement(
    int max_num_refinements, double max_refinement_change,
    const colmap::IncrementalMapper::Options &mapper_options,
    const Options &options, bool normalize_reconstruction) {
  // Complete and merge all tracks (points + lines + groups)
  CompleteAndMergeTracks(options);

  // Retriangulate
  const size_t num_retriangulated = Retriangulate(options);
  VLOG(1) << "=> Retriangulated observations: " << num_retriangulated;

  auto &point_recon = reconstruction_->PointRecon();
  auto &srec = reconstruction_->StructureRecon();

  // Use robust loss for the first iteration to protect against outlier
  // lines/points that haven't been filtered yet, then switch to TRIVIAL.
  // This mirrors the pattern used in IterativeLocalRefinement.
  auto opts = options;
  opts.structure_ba.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  opts.structure_ba.ceres->loss_function_scale = 1.0;

  StructureObservationManager struct_obs_manager(srec);

  for (int i = 0; i < max_num_refinements; ++i) {
    const size_t num_observations = point_recon.ComputeNumObservations();

    // Count line observations (active only for reporting)
    size_t num_line_observations = 0;
    size_t num_active_line_observations = 0;
    for (const auto &[line3D_id, line3d] : srec.Lines3D()) {
      num_line_observations += line3d.track.Length();
      num_active_line_observations += line3d.CountActiveObservations();
    }

    // Count group associations and types
    size_t num_group_point_assoc = 0;
    size_t num_group_line_assoc = 0;
    size_t num_groups_vp = 0;
    size_t num_groups_plane = 0;
    size_t num_groups_other = 0;
    for (const auto &[group3D_id, group3d] : srec.Groups3D()) {
      num_group_point_assoc += group3d.points.size();
      num_group_line_assoc += group3d.lines.size();
      if (group3d.type == GroupType::VP) {
        num_groups_vp++;
      } else if (group3d.type == GroupType::PLANE) {
        num_groups_plane++;
      } else {
        num_groups_other++;
      }
    }

    // Pre-BA: delete hopeless tracks before BA
    struct_obs_manager.DeleteSupportlessLineTracks(
        opts.min_active_ratio_for_deletion);

    // Classify lines for 2-step BA (uses existing active/inactive labels)
    FlatHashSet<line3D_t> reliable_ids, unreliable_ids;
    if (opts.use_two_step_ba) {
      auto line_loss = estimators::CreateLossFunction(
          opts.structure_ba.loss_function_type_line,
          opts.structure_ba.loss_function_scale_line);
      struct_obs_manager.ClassifyLineTracks(
          opts.min_active_line_observations, opts.pixel_uncertainty_threshold,
          line_loss.get(), reliable_ids, unreliable_ids);
    }

    LOG(INFO) << "  GlobalRefinement iter " << i
              << ": point_obs=" << num_observations
              << ", points3D=" << point_recon.NumPoints3D()
              << ", line_obs=" << num_line_observations
              << " (active=" << num_active_line_observations << ")"
              << ", lines3D=" << srec.NumLines3D()
              << " (reliable=" << reliable_ids.size()
              << ", unreliable=" << unreliable_ids.size() << ")"
              << ", groups3D=" << srec.NumGroups3D() << " (VP=" << num_groups_vp
              << ", PLANE=" << num_groups_plane
              << (num_groups_other > 0
                      ? ", OTHER=" + std::to_string(num_groups_other)
                      : "")
              << ", pt_assoc=" << num_group_point_assoc
              << ", ln_assoc=" << num_group_line_assoc << ")"
              << ", wireframe_edges=" << srec.Wireframe().CountEdges()
              << ", loss="
              << (opts.structure_ba.ceres->loss_function_type ==
                          colmap::CeresBundleAdjustmentOptions::
                              LossFunctionType::TRIVIAL
                      ? "TRIVIAL"
                  : opts.structure_ba.ceres->loss_function_type ==
                          colmap::CeresBundleAdjustmentOptions::
                              LossFunctionType::SOFT_L1
                      ? "SOFT_L1"
                      : "OTHER");

    if (opts.use_two_step_ba) {
      auto step1_opts = opts;
      AdjustGlobalBundleTwoStep(mapper_options, step1_opts, reliable_ids,
                                unreliable_ids);
    } else {
      AdjustGlobalBundle(mapper_options, opts);
    }

    if (normalize_reconstruction && !mapper_options.use_prior_position) {
      reconstruction_->Normalize();
    }

    // Post-BA: complete and merge tracks (points + lines)
    size_t num_changed = CompleteAndMergeTracks(opts);
    const size_t num_filtered_points = FilterPoints(mapper_options);

    // Post-BA: filter line tracks
    // - marks observations active/inactive
    // - hard-deletes individual inactive obs from tracks with >10 active images
    // - deletes entire tracks with <2 obs or active ratio below threshold
    const size_t num_deleted_tracks = struct_obs_manager.FilterLineTracks(
        opts.filter_max_line_angular_error, opts.filter_max_line_perp_error,
        opts.min_active_ratio_for_deletion);
    num_changed += num_filtered_points + num_deleted_tracks;

    // Post-BA: update group associations with latest point/line data.
    // Lines may have been deleted by FilterLineTracks, and new lines may
    // have been completed/merged. Rebuild associations so groups reflect
    // the current state before the next BA iteration.
    if (structure_triangulator_ && opts.triangulation.triangulate_groups) {
      FlatHashSet<group3D_t> all_group_ids;
      for (const auto &[gid, _] : srec.Groups3D()) {
        all_group_ids.insert(gid);
      }
      if (!all_group_ids.empty()) {
        structure_triangulator_->GroupTriangulator().UpdateGroupAssociations(
            opts.triangulation.group_options, all_group_ids);
        structure_triangulator_->GroupTriangulator().ClearModifiedGroups3D();
      }
      // Clean up truly dead groups after association rebuild
      DeleteSupportlessGroups(*reconstruction_);
    }

    LOG(INFO) << "  GlobalRefinement iter " << i
              << " post-filter: filtered_point_obs=" << num_filtered_points
              << ", deleted_line_tracks=" << num_deleted_tracks
              << ", remaining_points3D=" << point_recon.NumPoints3D()
              << ", remaining_lines3D=" << srec.NumLines3D()
              << ", remaining_groups3D=" << srec.NumGroups3D();

    const double changed =
        num_observations == 0
            ? 0
            : num_changed / static_cast<double>(num_observations);
    VLOG(1) << colmap::StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }

    // Only use robust cost function for first iteration
    opts.structure_ba.ceres->loss_function_type =
        colmap::CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  colmap_mapper_.ClearModifiedPoints3D();
  if (structure_triangulator_) {
    structure_triangulator_->ClearModified();
  }
}

// === Filtering ===

size_t StructureIncrementalMapper::FilterFrames(
    const colmap::IncrementalMapper::Options &options) {
  THROW_CHECK(reconstruction_);
  THROW_CHECK(options.Check());

  auto &point_recon = reconstruction_->PointRecon();

  // Same early-exit as COLMAP: don't filter in early stage
  const size_t kMinNumFrames = 20;
  if (point_recon.NumRegFrames() < kMinNumFrames) {
    return 0;
  }

  // Identify frames to filter (same logic as ObservationManager::FilterFrames)
  std::vector<colmap::frame_t> filtered_frame_ids;
  size_t num_bogus_camera = 0;
  size_t num_no_points = 0;
  for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
    const auto &frame = point_recon.Frame(frame_id);
    int num_points3D = 0;
    bool bogus = false;
    for (const auto &data_id : frame.ImageIds()) {
      const auto &image = point_recon.Image(data_id.id);
      num_points3D += image.NumPoints3D();
      if (image.CameraPtr()->HasBogusParams(options.min_focal_length_ratio,
                                            options.max_focal_length_ratio,
                                            options.max_extra_param)) {
        bogus = true;
        num_points3D = 0;
        break;
      }
    }
    if (num_points3D == 0) {
      filtered_frame_ids.push_back(frame_id);
      if (bogus) {
        ++num_bogus_camera;
      } else {
        ++num_no_points;
      }
    }
  }
  if (!filtered_frame_ids.empty()) {
    LOG(INFO) << "FilterFrames: filtering " << filtered_frame_ids.size()
              << " / " << point_recon.NumRegFrames()
              << " frames (bogus_camera=" << num_bogus_camera
              << ", no_points=" << num_no_points << ")";

    if (filtered_frame_ids.size() > 30) {
      LOG(WARNING) << "Large filtering: " << filtered_frame_ids.size() << " / "
                   << point_recon.NumRegFrames()
                   << " frames being filtered (bogus_camera="
                   << num_bogus_camera << ", no_points=" << num_no_points
                   << "). Global BA may be hurting the reconstruction.";
    }
  }

  auto &obs_manager = colmap_mapper_.ObservationManager();
  StructureObservationManager struct_obs_manager(
      reconstruction_->StructureRecon());

  for (const colmap::frame_t frame_id : filtered_frame_ids) {
    // Clean up structure observations (lines + groups) before point cleanup
    struct_obs_manager.DeRegisterFrame(frame_id);
    // Clean up point observations + reset pose
    obs_manager.DeRegisterFrame(frame_id);
    // Update our registration stats
    DeRegisterFrameEvent(frame_id);
    filtered_frames_.insert(frame_id);
  }

  VLOG(1) << "=> Filtered frames: " << filtered_frame_ids.size();
  return filtered_frame_ids.size();
}

size_t StructureIncrementalMapper::FilterPoints(
    const colmap::IncrementalMapper::Options &options) {
  return colmap_mapper_.FilterPoints(options);
}

size_t StructureIncrementalMapper::FilterLines(const Options &options) {
  THROW_CHECK(reconstruction_);
  StructureObservationManager obs_manager(reconstruction_->StructureRecon());
  const size_t num_filtered =
      obs_manager.FilterAllLines3D(options.filter_max_line_angular_error,
                                   options.filter_max_line_perp_error);
  VLOG(1) << "=> Filtered line observations: " << num_filtered;
  return num_filtered;
}

size_t StructureIncrementalMapper::SoftFilterLines(const Options &options) {
  THROW_CHECK(reconstruction_);
  StructureObservationManager obs_manager(reconstruction_->StructureRecon());
  const size_t num_filtered = obs_manager.UpdateLineObservationActivity(
      options.filter_max_line_angular_error,
      options.filter_max_line_perp_error);
  VLOG(1) << "=> Soft-filtered line observations: " << num_filtered;
  return num_filtered;
}

// === Track management ===

size_t
StructureIncrementalMapper::CompleteAndMergeTracks(const Options &options) {
  // COLMAP point tracks
  size_t num_changed = colmap_mapper_.CompleteAndMergeTracks(
      options.triangulation.point_options);

  // Structure tracks
  if (structure_triangulator_) {
    num_changed +=
        structure_triangulator_->CompleteAllTracks(options.triangulation);
    num_changed +=
        structure_triangulator_->MergeAllTracks(options.triangulation);

    // Update group associations after complete/merge
    if (options.triangulation.triangulate_groups) {
      const auto &modified_groups =
          structure_triangulator_->GroupTriangulator().GetModifiedGroups3D();
      if (!modified_groups.empty()) {
        structure_triangulator_->GroupTriangulator().UpdateGroupAssociations(
            options.triangulation.group_options, modified_groups);
        structure_triangulator_->GroupTriangulator().ClearModifiedGroups3D();
      }
    }
  }

  return num_changed;
}

size_t StructureIncrementalMapper::Retriangulate(const Options &options) {
  size_t num_retriangulated =
      colmap_mapper_.Retriangulate(options.triangulation.point_options);

  if (structure_triangulator_ && options.triangulation.triangulate_lines) {
    num_retriangulated +=
        structure_triangulator_->LineTriangulator().Retriangulate(
            options.triangulation.line_options);
  }

  return num_retriangulated;
}

// === Accessors ===

colmap::IncrementalMapper &StructureIncrementalMapper::ColmapMapper() {
  return colmap_mapper_;
}

const colmap::IncrementalMapper &
StructureIncrementalMapper::ColmapMapper() const {
  return colmap_mapper_;
}

IncrementalStructureTriangulator &
StructureIncrementalMapper::StructureTriangulator() {
  THROW_CHECK(structure_triangulator_);
  return *structure_triangulator_;
}

std::shared_ptr<HolisticReconstruction>
StructureIncrementalMapper::Reconstruction() const {
  return reconstruction_;
}

size_t StructureIncrementalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t StructureIncrementalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

void StructureIncrementalMapper::ResetInitializationStats() {
  colmap_mapper_.ResetInitializationStats();
}

void StructureIncrementalMapper::RegisterFrameEvent(colmap::frame_t frame_id) {
  THROW_CHECK(reconstruction_);
  const auto &frame = reconstruction_->PointRecon().Frame(frame_id);
  for (const auto &data_id : frame.ImageIds()) {
    size_t &num_regs = num_registrations_[data_id.id];
    num_regs += 1;
    if (num_regs == 1) {
      num_total_reg_images_ += 1;
    } else if (num_regs > 1) {
      num_shared_reg_images_ += 1;
    }
  }
}

void StructureIncrementalMapper::DeRegisterFrameEvent(
    colmap::frame_t frame_id) {
  THROW_CHECK(reconstruction_);
  const auto &frame = reconstruction_->PointRecon().Frame(frame_id);
  for (const auto &data_id : frame.ImageIds()) {
    size_t &num_regs = num_registrations_[data_id.id];
    num_regs -= 1;
    if (num_regs == 0) {
      num_total_reg_images_ -= 1;
    } else if (num_regs > 0) {
      num_shared_reg_images_ -= 1;
    }
  }
}

// === Private helpers ===

void StructureIncrementalMapper::FindLineCorrespondences(
    colmap::image_t image_id, std::vector<Line3d> &lines3d,
    std::vector<Line2d> &lines2d) const {
  lines3d.clear();
  lines2d.clear();

  if (!structure_corr_graph_ || !reconstruction_) {
    return;
  }

  const auto &srec = reconstruction_->StructureRecon();
  if (!srec.ExistsStructure2D(image_id)) {
    return;
  }

  const auto &s2d = srec.Structure2d(image_id);
  const auto &line_graph = structure_corr_graph_->LineGraph();
  const auto &point_recon = reconstruction_->PointRecon();

  FlatHashSet<line3D_t> seen_line3D_ids;

  for (line2D_t line2D_idx = 0;
       line2D_idx < static_cast<line2D_t>(s2d.NumLines()); ++line2D_idx) {
    const auto &line2d = s2d.Line(line2D_idx);

    // Look up correspondences via the line correspondence graph
    const auto corr_range =
        line_graph.FindCorrespondences(image_id, line2D_idx);
    for (const auto *corr = corr_range.beg; corr < corr_range.end; ++corr) {
      // Check if the corresponding image is registered
      if (!point_recon.ExistsImage(corr->image_id) ||
          !point_recon.Image(corr->image_id).HasPose()) {
        continue;
      }

      // Check if the corresponding line has a 3D line
      if (!srec.ExistsStructure2D(corr->image_id)) {
        continue;
      }
      const auto &corr_s2d = srec.Structure2d(corr->image_id);
      const auto corr_line2D_idx = static_cast<line2D_t>(corr->point2D_idx);
      if (corr_line2D_idx >= static_cast<line2D_t>(corr_s2d.NumLines())) {
        continue;
      }
      const auto &corr_line2d = corr_s2d.Line(corr_line2D_idx);
      if (!corr_line2d.HasLine3D()) {
        continue;
      }

      const line3D_t line3D_id = corr_line2d.line3D_id;
      if (seen_line3D_ids.count(line3D_id) > 0) {
        continue;
      }
      seen_line3D_ids.insert(line3D_id);

      lines3d.push_back(srec.Line(line3D_id));
      lines2d.push_back(line2d);
    }
  }
}

size_t
StructureIncrementalMapper::NumVisibleLines3D(colmap::image_t image_id) const {
  if (!structure_corr_graph_ || !reconstruction_) {
    return 0;
  }

  const auto &srec = reconstruction_->StructureRecon();
  if (!srec.ExistsStructure2D(image_id)) {
    return 0;
  }

  const auto &s2d = srec.Structure2d(image_id);
  const auto &line_graph = structure_corr_graph_->LineGraph();
  const auto &point_recon = reconstruction_->PointRecon();

  FlatHashSet<line3D_t> visible_line3D_ids;

  for (line2D_t line2D_idx = 0;
       line2D_idx < static_cast<line2D_t>(s2d.NumLines()); ++line2D_idx) {
    const auto corr_range =
        line_graph.FindCorrespondences(image_id, line2D_idx);
    for (const auto *corr = corr_range.beg; corr < corr_range.end; ++corr) {
      if (!point_recon.ExistsImage(corr->image_id) ||
          !point_recon.Image(corr->image_id).HasPose()) {
        continue;
      }
      if (!srec.ExistsStructure2D(corr->image_id)) {
        continue;
      }
      const auto &corr_s2d = srec.Structure2d(corr->image_id);
      const auto corr_line2D_idx = static_cast<line2D_t>(corr->point2D_idx);
      if (corr_line2D_idx >= static_cast<line2D_t>(corr_s2d.NumLines())) {
        continue;
      }
      const auto &corr_line2d = corr_s2d.Line(corr_line2D_idx);
      if (corr_line2d.HasLine3D()) {
        visible_line3D_ids.insert(corr_line2d.line3D_id);
      }
    }
  }

  return visible_line3D_ids.size();
}

} // namespace limap
