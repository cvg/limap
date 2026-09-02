#include "limap/sfm/structure_incremental_pipeline.h"

#include <colmap/util/logging.h>
#include <colmap/util/misc.h>

namespace limap {

// === StructureIncrementalPipelineOptions ===

colmap::IncrementalMapper::Options
StructureIncrementalPipelineOptions::Mapper() const {
  return colmap_options.Mapper();
}

bool StructureIncrementalPipelineOptions::IsInitialPairProvided() const {
  return colmap_options.IsInitialPairProvided();
}

bool StructureIncrementalPipelineOptions::Check() const {
  CHECK_OPTION(colmap_options.Check());
  CHECK_OPTION(structure_options.Check());
  return true;
}

estimators::StructureBundleAdjustmentOptions
StructureIncrementalPipelineOptions::LocalStructureBA() const {
  auto opts = structure_options.structure_ba;
  // Merge solver settings from COLMAP's local BA config
  const auto colmap_ba = colmap_options.LocalBundleAdjustment();
  opts.ceres->solver_options.function_tolerance =
      colmap_ba.ceres->solver_options.function_tolerance;
  opts.ceres->solver_options.gradient_tolerance =
      colmap_ba.ceres->solver_options.gradient_tolerance;
  opts.ceres->solver_options.parameter_tolerance =
      colmap_ba.ceres->solver_options.parameter_tolerance;
  opts.ceres->solver_options.max_num_iterations =
      colmap_ba.ceres->solver_options.max_num_iterations;
  opts.ceres->solver_options.max_linear_solver_iterations =
      colmap_ba.ceres->solver_options.max_linear_solver_iterations;
  opts.ceres->solver_options.logging_type =
      colmap_ba.ceres->solver_options.logging_type;
  opts.ceres->solver_options.num_threads = colmap_options.num_threads;
  opts.print_summary = true;
  opts.refine_focal_length = colmap_options.ba_refine_focal_length;
  opts.refine_principal_point = colmap_options.ba_refine_principal_point;
  opts.refine_extra_params = colmap_options.ba_refine_extra_params;
  opts.refine_sensor_from_rig = colmap_options.ba_refine_sensor_from_rig;
  opts.ceres->min_num_residuals_for_cpu_multi_threading =
      colmap_options.ba_min_num_residuals_for_cpu_multi_threading;
  opts.ceres->loss_function_scale = 1.0;
  opts.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  // Local BA runs every registration and doesn't need high precision.
  // Match COLMAP's convention: local gradient_tolerance (10.0) is 10x looser
  // than global (1.0). Apply the same ratio to our custom function tolerance.
  opts.custom_function_tolerance *= 10.0;
  return opts;
}

estimators::StructureBundleAdjustmentOptions
StructureIncrementalPipelineOptions::GlobalStructureBA() const {
  auto opts = structure_options.structure_ba;
  // Merge solver settings from COLMAP's global BA config
  const auto colmap_ba = colmap_options.GlobalBundleAdjustment();
  opts.ceres->solver_options.function_tolerance =
      colmap_ba.ceres->solver_options.function_tolerance;
  opts.ceres->solver_options.gradient_tolerance =
      colmap_ba.ceres->solver_options.gradient_tolerance;
  opts.ceres->solver_options.parameter_tolerance =
      colmap_ba.ceres->solver_options.parameter_tolerance;
  opts.ceres->solver_options.max_num_iterations =
      colmap_ba.ceres->solver_options.max_num_iterations;
  opts.ceres->solver_options.max_linear_solver_iterations =
      colmap_ba.ceres->solver_options.max_linear_solver_iterations;
  opts.ceres->solver_options.logging_type =
      colmap_ba.ceres->solver_options.logging_type;
  opts.ceres->solver_options.num_threads = colmap_options.num_threads;
  opts.print_summary = true;
  opts.refine_focal_length = colmap_options.ba_refine_focal_length;
  opts.refine_principal_point = colmap_options.ba_refine_principal_point;
  opts.refine_extra_params = colmap_options.ba_refine_extra_params;
  opts.refine_sensor_from_rig = colmap_options.ba_refine_sensor_from_rig;
  opts.ceres->min_num_residuals_for_cpu_multi_threading =
      colmap_options.ba_min_num_residuals_for_cpu_multi_threading;
  opts.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  // Looser convergence tolerance for global Structure BA (groups + wireframes).
  // Angle cost (VP + angular constraints) is stripped by
  // AngleCostAwareToleranceCallback, but structure cost (plane 2D) still slows
  // convergence of the combined check. 1e-4 gives ~13 median iterations (vs 4
  // for Lines-only at 1e-5).
  opts.custom_function_tolerance = 1e-4;
  return opts;
}

// === StructureIncrementalPipeline ===

StructureIncrementalPipeline::StructureIncrementalPipeline(
    std::shared_ptr<StructureIncrementalPipelineOptions> options,
    std::shared_ptr<colmap::DatabaseCache> database_cache,
    std::shared_ptr<StructureDatabaseCache> structure_db_cache,
    std::shared_ptr<HolisticReconstructionManager> reconstruction_manager)
    : options_(std::move(THROW_CHECK_NOTNULL(options))),
      database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))),
      structure_db_cache_(std::move(THROW_CHECK_NOTNULL(structure_db_cache))),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {
  THROW_CHECK(options_->Check());
}

void StructureIncrementalPipeline::Run(
    std::function<void()> initial_image_pair_callback,
    std::function<void()> next_image_callback) {
  if (database_cache_->NumImages() == 0) {
    LOG(WARNING) << "No images with matches";
    return;
  }

  const bool continue_reconstruction = reconstruction_manager_->Size() > 0;
  THROW_CHECK_LE(reconstruction_manager_->Size(), 1)
      << "Can only continue from a single reconstruction, "
         "but multiple are given.";

  const size_t num_images = database_cache_->NumImages();

  colmap::IncrementalMapper::Options mapper_options = options_->Mapper();
  StructureIncrementalMapper mapper(database_cache_, structure_db_cache_);

  if (Reconstruct(mapper, mapper_options, continue_reconstruction,
                  initial_image_pair_callback,
                  next_image_callback) == Status::STOP) {
    return;
  }

  auto ShouldStop = [this, &mapper, &num_images]() {
    return mapper.NumTotalRegImages() == num_images;
  };

  // Relaxation loop: relax init_min_num_inliers and init_min_tri_angle
  constexpr size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    if (ShouldStop()) {
      break;
    }

    LOG(INFO) << "=> Relaxing the initialization constraints.";
    mapper_options.init_min_num_inliers /= 2;
    mapper.ResetInitializationStats();
    if (Reconstruct(mapper, mapper_options, false, initial_image_pair_callback,
                    next_image_callback) == Status::STOP) {
      break;
    }

    if (ShouldStop()) {
      break;
    }

    LOG(INFO) << "=> Relaxing the initialization constraints.";
    mapper_options.init_min_tri_angle /= 2;
    mapper.ResetInitializationStats();
    if (Reconstruct(mapper, mapper_options, false, initial_image_pair_callback,
                    next_image_callback) == Status::STOP) {
      break;
    }
  }
}

StructureIncrementalPipeline::Status StructureIncrementalPipeline::Reconstruct(
    StructureIncrementalMapper &mapper,
    const colmap::IncrementalMapper::Options &mapper_options,
    bool continue_reconstruction,
    const std::function<void()> &initial_image_pair_callback,
    const std::function<void()> &next_image_callback) {
  for (int num_trials = 0;
       num_trials < options_->colmap_options.init_num_trials; ++num_trials) {
    const size_t reconstruction_idx =
        (!continue_reconstruction || num_trials > 0)
            ? reconstruction_manager_->Add()
            : 0;
    auto reconstruction = reconstruction_manager_->Get(reconstruction_idx);

    const Status status =
        ReconstructSubModel(mapper, mapper_options, reconstruction,
                            initial_image_pair_callback, next_image_callback);
    switch (status) {
    case Status::INTERRUPTED: {
      reconstruction->PointRecon().UpdatePoint3DErrors();
      LOG(INFO) << "Keeping reconstruction due to interrupt";
      mapper.EndReconstruction(false);
      return Status::STOP;
    }

    case Status::BAD_INITIAL_PAIR: {
      LOG(INFO) << "Discarding reconstruction due to bad initial pair";
      mapper.EndReconstruction(true);
      reconstruction_manager_->Delete(reconstruction_idx);
      break;
    }

    case Status::NO_INITIAL_PAIR: {
      LOG(INFO) << "Discarding reconstruction due to no initial pair";
      mapper.EndReconstruction(true);
      reconstruction_manager_->Delete(reconstruction_idx);
      return Status::CONTINUE;
    }

    case Status::SUCCESS: {
      const size_t num_reg_images = reconstruction->PointRecon().NumRegImages();
      const size_t total_num_reg_images = mapper.NumTotalRegImages();

      // Always keep the first reconstruction, independent of size
      if ((options_->colmap_options.multiple_models &&
           reconstruction_manager_->Size() > 1 &&
           num_reg_images <
               static_cast<size_t>(options_->colmap_options.min_model_size)) ||
          num_reg_images == 0) {
        LOG(WARNING) << "Discarding reconstruction due to insufficient size";
        mapper.EndReconstruction(true);
        reconstruction_manager_->Delete(reconstruction_idx);
      } else {
        reconstruction->PointRecon().UpdatePoint3DErrors();
        LOG(INFO) << "Keeping successful reconstruction";
        mapper.EndReconstruction(false);
      }

      // Check if we should reconstruct another sub-model
      if (!options_->colmap_options.multiple_models ||
          reconstruction_manager_->Size() >=
              static_cast<size_t>(options_->colmap_options.max_num_models) ||
          total_num_reg_images >= database_cache_->NumImages() - 1) {
        return Status::STOP;
      }
      break;
    }

    default:
      LOG(FATAL_THROW) << "Unknown reconstruction status.";
    }
  }

  return Status::CONTINUE;
}

StructureIncrementalPipeline::Status
StructureIncrementalPipeline::ReconstructSubModel(
    StructureIncrementalMapper &mapper,
    const colmap::IncrementalMapper::Options &mapper_options,
    std::shared_ptr<HolisticReconstruction> reconstruction,
    const std::function<void()> &initial_image_pair_callback,
    const std::function<void()> &next_image_callback) {
  mapper.BeginReconstruction(reconstruction);

  auto &point_recon = reconstruction->PointRecon();

  // Phase 1: Initialize
  if (point_recon.NumRegFrames() == 0) {
    const Status init_status =
        InitializeReconstruction(mapper, mapper_options, *reconstruction);
    if (init_status != Status::SUCCESS) {
      return init_status;
    }
  }

  // Callback after initial pair registration (mirrors COLMAP's
  // INITIAL_IMAGE_PAIR_REG_CALLBACK). Throwing from the callback
  // (e.g. py::error_already_set) propagates out for clean interrupt.
  if (initial_image_pair_callback) {
    initial_image_pair_callback();
  }

  // Phase 2: Incremental registration loop
  const auto &opts = options_->colmap_options;

  // Thread num_threads from COLMAP options to line triangulation
  auto structure_opts = options_->structure_options;
  structure_opts.triangulation.line_options.num_threads = opts.num_threads;

  // Build local/global BA options from merged settings
  auto local_opts = structure_opts;
  local_opts.structure_ba = options_->LocalStructureBA();
  auto global_opts = structure_opts;
  global_opts.structure_ba = options_->GlobalStructureBA();

  size_t ba_prev_num_reg_frames = point_recon.NumRegFrames();
  size_t ba_prev_num_points = point_recon.NumPoints3D();

  // Two-pass registration (matching COLMAP's approach):
  // Pass 1: try all candidates via hybrid PnPL (structure-based)
  // Pass 2: only if all PnPL attempts fail, try structure-less candidates
  std::vector<bool> structure_less_flags;
  if (opts.structure_less_registration_fallback) {
    structure_less_flags = {false, true};
  } else {
    structure_less_flags = {false};
  }

  bool reg_next_success = true;
  bool prev_reg_next_success = true;
  do {
    prev_reg_next_success = reg_next_success;
    reg_next_success = false;
    colmap::image_t next_image_id = colmap::kInvalidImageId;

    for (const bool structure_less : structure_less_flags) {
      const std::vector<colmap::image_t> next_images =
          mapper.FindNextImages(mapper_options, structure_opts, structure_less);

      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        next_image_id = next_images[reg_trial];

        LOG(INFO) << colmap::StringPrintf(
            "Registering image #%d (num_reg_frames=%d)", next_image_id,
            point_recon.NumRegFrames());

        if (structure_less) {
          LOG(INFO) << "  => Structure-less registration";
          reg_next_success = mapper.RegisterNextStructureLessImage(
              mapper_options, next_image_id);
          if (reg_next_success) {
            const auto &img = point_recon.Image(next_image_id);
            mapper.RegisterFrameEvent(img.FrameId());
          }
        } else {
          reg_next_success = mapper.RegisterNextImage(
              mapper_options, structure_opts, next_image_id);
        }

        if (reg_next_success) {
          break;
        } else {
          LOG(INFO) << "  => Could not register, trying another image.";
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              point_recon.NumRegImages() <
                  static_cast<size_t>(opts.min_model_size)) {
            break;
          }
        }
      }

      if (reg_next_success) {
        break;
      }
    }

    if (reg_next_success) {
      // Triangulate points + lines + groups
      mapper.TriangulateImage(structure_opts, next_image_id);

      // Local refinement with structure BA
      mapper.IterativeLocalRefinement(
          opts.ba_local_max_refinements, opts.ba_local_max_refinement_change,
          mapper_options, local_opts, next_image_id);

      const auto &srec = reconstruction->StructureRecon();
      LOG(INFO) << "  => Points: " << point_recon.NumPoints3D()
                << ", Lines: " << srec.NumLines3D()
                << ", Groups: " << srec.NumGroups3D();

      // Check if global BA needed
      if (CheckRunGlobalRefinement(*reconstruction, ba_prev_num_reg_frames,
                                   ba_prev_num_points)) {
        LOG(INFO) << "Retriangulation and Global bundle adjustment";
        mapper.IterativeGlobalRefinement(opts.ba_global_max_refinements,
                                         opts.ba_global_max_refinement_change,
                                         mapper_options, global_opts);
        mapper.FilterFrames(mapper_options);
        ba_prev_num_points = point_recon.NumPoints3D();
        ba_prev_num_reg_frames = point_recon.NumRegFrames();
      }

      if (opts.extract_colors) {
        const auto &image = point_recon.Image(next_image_id);
        for (const auto &data_id : image.FramePtr()->ImageIds()) {
          point_recon.ExtractColorsForImage(data_id.id, opts.image_path);
        }
      }

      // Callback after each image registration (mirrors COLMAP's
      // NEXT_IMAGE_REG_CALLBACK). Enables Python interrupt handling.
      if (next_image_callback) {
        next_image_callback();
      }
    }

    // Check model overlap
    const size_t max_model_overlap =
        static_cast<size_t>(opts.max_model_overlap);
    if (mapper.NumSharedRegImages() >= max_model_overlap) {
      break;
    }

    // If stuck, try one global refinement then retry
    if (!reg_next_success && prev_reg_next_success &&
        point_recon.NumRegFrames() >= 2) {
      LOG(INFO) << "Retriangulation and Global bundle adjustment";
      mapper.IterativeGlobalRefinement(opts.ba_global_max_refinements,
                                       opts.ba_global_max_refinement_change,
                                       mapper_options, global_opts);
      mapper.FilterFrames(mapper_options);
    }
  } while (reg_next_success || prev_reg_next_success);

  // Final global refinement
  if (point_recon.NumRegFrames() > 0 &&
      point_recon.NumRegFrames() != ba_prev_num_reg_frames &&
      point_recon.NumPoints3D() != ba_prev_num_points) {
    LOG(INFO) << "Final retriangulation and global bundle adjustment";
    mapper.IterativeGlobalRefinement(opts.ba_global_max_refinements,
                                     opts.ba_global_max_refinement_change,
                                     mapper_options, global_opts);
    mapper.FilterFrames(mapper_options);
  }

  return Status::SUCCESS;
}

StructureIncrementalPipeline::Status
StructureIncrementalPipeline::InitializeReconstruction(
    StructureIncrementalMapper &mapper,
    const colmap::IncrementalMapper::Options &mapper_options,
    HolisticReconstruction &reconstruction) {
  auto &point_recon = reconstruction.PointRecon();
  const auto &opts = options_->colmap_options;
  auto structure_opts = options_->structure_options;
  structure_opts.triangulation.line_options.num_threads = opts.num_threads;

  colmap::image_t image_id1 = static_cast<colmap::image_t>(opts.init_image_id1);
  colmap::image_t image_id2 = static_cast<colmap::image_t>(opts.init_image_id2);

  // Find initial pair (point-only, same as COLMAP)
  colmap::Rigid3d cam2_from_cam1;
  if (!opts.IsInitialPairProvided()) {
    LOG(INFO) << "Finding good initial image pair";
    if (!mapper.FindInitialImagePair(mapper_options, image_id1, image_id2,
                                     cam2_from_cam1)) {
      LOG(INFO) << "=> No good initial image pair found.";
      return Status::NO_INITIAL_PAIR;
    }
  } else {
    if (!point_recon.ExistsImage(image_id1) ||
        !point_recon.ExistsImage(image_id2)) {
      LOG(INFO) << colmap::StringPrintf(
          "=> Initial image pair #%d and #%d does not exist.", image_id1,
          image_id2);
      return Status::NO_INITIAL_PAIR;
    }
    if (!mapper.EstimateInitialTwoViewGeometry(mapper_options, image_id1,
                                               image_id2, cam2_from_cam1)) {
      LOG(INFO) << "=> Provided pair is unsuitable for initialization.";
      return Status::BAD_INITIAL_PAIR;
    }
  }

  LOG(INFO) << colmap::StringPrintf(
      "Registering initial image pair #%d and #%d", image_id1, image_id2);
  mapper.RegisterInitialImagePair(mapper_options, image_id1, image_id2,
                                  cam2_from_cam1);

  // Triangulate points for initial pair (use COLMAP's triangulator)
  colmap::IncrementalTriangulator::Options tri_options = opts.Triangulation();
  tri_options.min_angle = mapper_options.init_min_tri_angle;
  for (const colmap::image_t image_id : {image_id1, image_id2}) {
    const auto &image = point_recon.Image(image_id);
    for (const auto &data_id : image.FramePtr()->ImageIds()) {
      mapper.ColmapMapper().TriangulateImage(tri_options, data_id.id);
    }
  }

  // Global BA (point-only for initialization)
  LOG(INFO) << "Global bundle adjustment";
  auto global_opts = structure_opts;
  global_opts.structure_ba = options_->GlobalStructureBA();
  mapper.AdjustGlobalBundle(mapper_options, global_opts);
  point_recon.Normalize();
  mapper.FilterPoints(mapper_options);
  mapper.FilterFrames(mapper_options);

  // Check initialization quality
  if (point_recon.NumRegFrames() == 0 || point_recon.NumPoints3D() == 0) {
    return Status::BAD_INITIAL_PAIR;
  }
  if (static_cast<int>(point_recon.NumPoints3D()) <
      mapper_options.abs_pose_min_num_inliers) {
    return Status::BAD_INITIAL_PAIR;
  }

  // Now triangulate initial structure (lines + groups)
  mapper.TriangulateImage(structure_opts, image_id1);
  mapper.TriangulateImage(structure_opts, image_id2);

  if (opts.extract_colors) {
    for (const colmap::image_t image_id : {image_id1, image_id2}) {
      const auto &image = point_recon.Image(image_id);
      for (const auto &data_id : image.FramePtr()->ImageIds()) {
        point_recon.ExtractColorsForImage(data_id.id, opts.image_path);
      }
    }
  }

  return Status::SUCCESS;
}

bool StructureIncrementalPipeline::CheckRunGlobalRefinement(
    const HolisticReconstruction &reconstruction, size_t ba_prev_num_reg_frames,
    size_t ba_prev_num_points) const {
  const auto &point_recon = reconstruction.PointRecon();
  const auto &opts = options_->colmap_options;
  return point_recon.NumRegFrames() >=
             opts.ba_global_frames_ratio * ba_prev_num_reg_frames ||
         point_recon.NumRegFrames() >=
             opts.ba_global_frames_freq + ba_prev_num_reg_frames ||
         point_recon.NumPoints3D() >=
             opts.ba_global_points_ratio * ba_prev_num_points ||
         point_recon.NumPoints3D() >=
             opts.ba_global_points_freq + ba_prev_num_points;
}

std::shared_ptr<const StructureIncrementalPipelineOptions>
StructureIncrementalPipeline::Options() const {
  return options_;
}

const std::shared_ptr<HolisticReconstructionManager> &
StructureIncrementalPipeline::ReconstructionManager() const {
  return reconstruction_manager_;
}

} // namespace limap
