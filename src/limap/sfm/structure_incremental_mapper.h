#pragma once

#include <colmap/sfm/incremental_mapper.h>

#include "limap/estimators/absolute_pose/point_line_absolute_pose.h"
#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/sfm/group_verification.h"
#include "limap/sfm/incremental_structure_triangulator.h"

namespace limap {

class StructureIncrementalMapper {
public:
  struct Options {
    // Weight for line visibility in next-image ranking.
    // Combined score = num_visible_points + line_visibility_weight *
    // num_visible_lines
    double line_visibility_weight = 1.0;

    // Registration options (new: hybrid point+line)
    bool use_lines_for_registration = true;
    estimators::absolute_pose::PointLineAbsolutePoseOptions registration;

    // Structure triangulation options
    IncrementalStructureTriangulator::Options triangulation;

    // Structure BA options
    estimators::StructureBundleAdjustmentOptions structure_ba;

    // Post-BA filtering
    double filter_max_line_angular_error = 8.0; // degrees
    double filter_max_line_perp_error = 5.0;    // pixels
    GroupVerificationOptions group_verification;

    // 2-step BA: Step 1 optimizes reliable lines with full poses,
    // Step 2 refines unreliable lines with fixed poses.
    bool use_two_step_ba = true;
    size_t min_active_line_observations = 4;    // reliability threshold
    double min_active_ratio_for_deletion = 0.2; // hard-delete threshold
    double pixel_uncertainty_threshold =
        30.0; // covariance-based filter (0=disable)

    bool Check() const;
  };

  struct LocalBundleAdjustmentReport {
    // Point-related (from COLMAP)
    size_t num_merged_observations = 0;
    size_t num_completed_observations = 0;
    size_t num_filtered_observations = 0;
    size_t num_adjusted_observations = 0;
    // Structure-related
    size_t num_merged_line_observations = 0;
    size_t num_completed_line_observations = 0;
    size_t num_merged_group_observations = 0;
    size_t num_completed_group_observations = 0;
  };

  StructureIncrementalMapper(
      std::shared_ptr<const colmap::DatabaseCache> database_cache,
      std::shared_ptr<const StructureDatabaseCache> structure_db_cache);

  // === Lifecycle (delegates to COLMAP mapper) ===
  void
  BeginReconstruction(std::shared_ptr<HolisticReconstruction> reconstruction);
  void EndReconstruction(bool discard);

  // === Initialization (delegates to COLMAP mapper) ===
  bool FindInitialImagePair(const colmap::IncrementalMapper::Options &options,
                            colmap::image_t &image_id1,
                            colmap::image_t &image_id2,
                            colmap::Rigid3d &cam2_from_cam1);
  void
  RegisterInitialImagePair(const colmap::IncrementalMapper::Options &options,
                           colmap::image_t image_id1, colmap::image_t image_id2,
                           const colmap::Rigid3d &cam2_from_cam1);
  bool EstimateInitialTwoViewGeometry(
      const colmap::IncrementalMapper::Options &options,
      colmap::image_t image_id1, colmap::image_t image_id2,
      colmap::Rigid3d &cam2_from_cam1);

  // === Image selection (extends COLMAP with line visibility) ===
  std::vector<colmap::image_t>
  FindNextImages(const colmap::IncrementalMapper::Options &options,
                 const Options &structure_options, bool structure_less = false);

  // === Registration (NEW: hybrid PnPL) ===
  bool
  RegisterNextImage(const colmap::IncrementalMapper::Options &mapper_options,
                    const Options &options, colmap::image_t image_id);

  // Shared tail of RegisterNextImage: sets the pose, registers the frame and
  // continues the point tracks of the inlier correspondences.
  bool FinishRegistration(
      colmap::Image &image,
      const estimators::absolute_pose::PointLineAbsolutePoseResult &result,
      const std::vector<std::pair<colmap::point2D_t, colmap::point3D_t>>
          &tri_corrs);

  // Structure-less registration (delegates to COLMAP, tracks our counter).
  bool RegisterNextStructureLessImage(
      const colmap::IncrementalMapper::Options &mapper_options,
      colmap::image_t image_id);

  // === Triangulation (NEW: structure triangulation) ===
  size_t TriangulateImage(const Options &options, colmap::image_t image_id);

  // === Bundle Adjustment (NEW: structure BA) ===
  LocalBundleAdjustmentReport
  AdjustLocalBundle(const colmap::IncrementalMapper::Options &mapper_options,
                    const Options &options, colmap::image_t image_id);

  bool
  AdjustGlobalBundle(const colmap::IncrementalMapper::Options &mapper_options,
                     const Options &options);

  void IterativeLocalRefinement(
      int max_num_refinements, double max_refinement_change,
      const colmap::IncrementalMapper::Options &mapper_options,
      const Options &options, colmap::image_t image_id);

  void IterativeGlobalRefinement(
      int max_num_refinements, double max_refinement_change,
      const colmap::IncrementalMapper::Options &mapper_options,
      const Options &options, bool normalize_reconstruction = true);

  // === Filtering (delegates to COLMAP mapper / StructureObservationManager)
  // ===
  size_t FilterFrames(const colmap::IncrementalMapper::Options &options);
  size_t FilterPoints(const colmap::IncrementalMapper::Options &options);
  size_t FilterLines(const Options &options);

  // Soft filter: tag observations as inactive instead of deleting.
  // Returns number of newly inactivated observations.
  size_t SoftFilterLines(const Options &options);

  // === Track management ===
  size_t CompleteAndMergeTracks(const Options &options);
  size_t Retriangulate(const Options &options);

  // === Accessors ===
  colmap::IncrementalMapper &ColmapMapper();
  const colmap::IncrementalMapper &ColmapMapper() const;
  IncrementalStructureTriangulator &StructureTriangulator();
  std::shared_ptr<HolisticReconstruction> Reconstruction() const;
  size_t NumTotalRegImages() const;
  size_t NumSharedRegImages() const;
  void ResetInitializationStats();

  // Mirror of COLMAP's RegisterFrameEvent/DeRegisterFrameEvent.
  // We maintain our own stats because COLMAP's are private.
  void RegisterFrameEvent(colmap::frame_t frame_id);
  void DeRegisterFrameEvent(colmap::frame_t frame_id);

  // 2-step global BA: Step 1 with reliable lines, Step 2 with unreliable lines
  void AdjustGlobalBundleTwoStep(
      const colmap::IncrementalMapper::Options &mapper_options,
      const Options &options, const FlatHashSet<line3D_t> &reliable_ids,
      const FlatHashSet<line3D_t> &unreliable_ids);

private:
  // Gather 2D-3D line correspondences for registration
  void FindLineCorrespondences(colmap::image_t image_id,
                               std::vector<Line3d> &lines3d,
                               std::vector<Line2d> &lines2d) const;

  // Count visible 3D lines for an unregistered image
  size_t NumVisibleLines3D(colmap::image_t image_id) const;

  std::shared_ptr<const colmap::DatabaseCache> database_cache_;
  std::shared_ptr<const StructureDatabaseCache> structure_db_cache_;

  colmap::IncrementalMapper colmap_mapper_;

  std::unique_ptr<IncrementalStructureTriangulator> structure_triangulator_;
  std::shared_ptr<HolisticReconstruction> reconstruction_;
  std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph_;

  // Registration statistics (mirrors COLMAP's RegistrationStatistics).
  size_t num_total_reg_images_ = 0;
  size_t num_shared_reg_images_ = 0;
  FlatHashMap<colmap::image_t, size_t> num_registrations_;
  FlatHashSet<colmap::frame_t> filtered_frames_;

  // Per-image registration trial counters (mirrors COLMAP's separated
  // counters). Regular and structure-less registration have independent budgets
  // so that structure-less attempts don't eat into the regular trial limit.
  FlatHashMap<colmap::image_t, size_t> num_reg_trials_;
  FlatHashMap<colmap::image_t, size_t> num_structure_less_reg_trials_;
};

} // namespace limap
