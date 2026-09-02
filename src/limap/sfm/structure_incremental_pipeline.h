#pragma once

#include <functional>

#include <colmap/controllers/incremental_pipeline.h>

#include "limap/scene/holistic_reconstruction_manager.h"
#include "limap/sfm/structure_incremental_mapper.h"

namespace limap {

struct StructureIncrementalPipelineOptions {
  // COLMAP pipeline options (all point-only config lives here:
  // init pair, num threads, BA solver settings, snapshot, etc.)
  colmap::IncrementalPipelineOptions colmap_options;

  // Structure mapper options (registration, triangulation, structure BA)
  StructureIncrementalMapper::Options structure_options;

  // Convenience accessors (forward to colmap_options)
  colmap::IncrementalMapper::Options Mapper() const;
  bool IsInitialPairProvided() const;
  bool Check() const;

  // Build local/global BA options by merging COLMAP solver settings
  // into structure_options.structure_ba.
  estimators::StructureBundleAdjustmentOptions LocalStructureBA() const;
  estimators::StructureBundleAdjustmentOptions GlobalStructureBA() const;
};

class StructureIncrementalPipeline {
public:
  enum class Status {
    SUCCESS,
    INTERRUPTED,
    CONTINUE,
    STOP,
    NO_INITIAL_PAIR,
    BAD_INITIAL_PAIR,
  };

  StructureIncrementalPipeline(
      std::shared_ptr<StructureIncrementalPipelineOptions> options,
      std::shared_ptr<colmap::DatabaseCache> database_cache,
      std::shared_ptr<StructureDatabaseCache> structure_db_cache,
      std::shared_ptr<HolisticReconstructionManager> reconstruction_manager);

  // Run the full pipeline.
  // Callbacks are called at safe points; throwing from a callback
  // (e.g. py::error_already_set for Ctrl-C) interrupts the pipeline.
  void Run(std::function<void()> initial_image_pair_callback = nullptr,
           std::function<void()> next_image_callback = nullptr);

  // Public for Python custom pipelines
  Status Reconstruct(
      StructureIncrementalMapper &mapper,
      const colmap::IncrementalMapper::Options &mapper_options,
      bool continue_reconstruction,
      const std::function<void()> &initial_image_pair_callback = nullptr,
      const std::function<void()> &next_image_callback = nullptr);

  Status ReconstructSubModel(
      StructureIncrementalMapper &mapper,
      const colmap::IncrementalMapper::Options &mapper_options,
      std::shared_ptr<HolisticReconstruction> reconstruction,
      const std::function<void()> &initial_image_pair_callback = nullptr,
      const std::function<void()> &next_image_callback = nullptr);

  Status InitializeReconstruction(
      StructureIncrementalMapper &mapper,
      const colmap::IncrementalMapper::Options &mapper_options,
      HolisticReconstruction &reconstruction);

  bool CheckRunGlobalRefinement(const HolisticReconstruction &reconstruction,
                                size_t ba_prev_num_reg_frames,
                                size_t ba_prev_num_points) const;

  // Accessors
  std::shared_ptr<const StructureIncrementalPipelineOptions> Options() const;
  const std::shared_ptr<HolisticReconstructionManager> &
  ReconstructionManager() const;

private:
  std::shared_ptr<StructureIncrementalPipelineOptions> options_;
  std::shared_ptr<colmap::DatabaseCache> database_cache_;
  std::shared_ptr<StructureDatabaseCache> structure_db_cache_;
  std::shared_ptr<HolisticReconstructionManager> reconstruction_manager_;
};

} // namespace limap
