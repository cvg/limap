#pragma once

#include <memory>
#include <optional>

#include <colmap/scene/correspondence_graph.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/geometry/line_linker.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_correspondence_graph.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/sfm/line_triangulation_utils.h"
#include "limap/sfm/structure_observation_manager.h"
#include "limap/util/types.h"

namespace limap {

// Class that triangulates lines during incremental reconstruction.
// Follows COLMAP's IncrementalTriangulator pattern.
class IncrementalLineTriangulator {
public:
  struct Options {
    // Transitivity for correspondence search
    int max_transitivity = 1;

    // Images each image is exhaustively paired against. Supplying this
    // selects exhaustive line matching: candidates come from these images
    // instead of the line correspondence graph.
    ExhaustiveMatchNeighbors exhaustive_match_neighbors;

    // Triangulation variants
    bool use_point_assisted_triangulation = true;
    bool use_vp_assisted_triangulation = true;

    // Hyperparameters (from GlobalLineTriangulationOptions)
    double min_length_2d = 20.0;
    double IoU_threshold = 0.1;
    double sensitivity_threshold = 70.0; // degrees
    double var2d = 2.0;

    // Scoring
    double fullscore_th = 1.0;
    double scale_inv_th = 0.01;

    // Linker options
    LineLinker2dOptions linker2d_options;
    LineLinker3dOptions linker3d_options;

    // Error thresholds (from COLMAP)
    double create_max_angle_error = 2.0;    // degrees
    double continue_max_angle_error = 2.0;  // degrees
    double merge_max_reproj_error = 4.0;    // pixels
    double merge_max_angle_error_2d = 5.0;  // degrees
    double merge_max_angle_error_3d = 20.0; // degrees, early 3D rejection
    double complete_max_reproj_error = 4.0;

    // Maximum transitivity for track completion
    int complete_max_transitivity = 5;

    // Re-triangulation thresholds
    double re_max_angle_error = 5.0;
    double re_min_ratio = 0.2;
    int re_max_trials = 1;

    // Minimum triangulation angle for a stable triangulation
    double min_angle = 1.5; // degrees

    // Whether to ignore two-view tracks
    bool ignore_two_view_tracks = true;

    // Number of threads for parallel triangulation (-1 = all available)
    int num_threads = -1;

    bool Check() const;

    // Get triangulation parameters for helper functions
    LineTriangulationParams GetTriangulationParams() const {
      LineTriangulationParams params;
      params.var2d = var2d;
      params.IoU_threshold = IoU_threshold;
      params.sensitivity_threshold = sensitivity_threshold;
      return params;
    }
  };

  // Create incremental line triangulator. Note that both the correspondence
  // graph and the reconstruction objects must live as long as the triangulator.
  // If obs_manager is not provided, one will be created internally.
  IncrementalLineTriangulator(
      std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph,
      HolisticReconstruction &reconstruction,
      std::shared_ptr<StructureObservationManager> obs_manager = nullptr);

  // Triangulate observations of image.
  //
  // Triangulation includes creation of new lines, continuation of existing
  // lines, and merging of separate lines if given image bridges tracks.
  //
  // Note that the given image must be registered and its pose must be set
  // in the associated reconstruction.
  size_t TriangulateImage(const Options &options, colmap::image_t image_id);

  // Complete triangulations for image. Tries to create new tracks for not
  // yet triangulated observations and tries to complete existing tracks.
  // Returns the number of completed observations.
  size_t CompleteImage(const Options &options, colmap::image_t image_id);

  // Complete tracks for specific 3D lines.
  //
  // Completion tries to recursively add observations to a track that might
  // have failed to triangulate before due to inaccurate poses, etc.
  // Returns the number of completed observations.
  size_t CompleteTracks(const Options &options,
                        const FlatHashSet<line3D_t> &line3D_ids);

  // Complete tracks of all 3D lines.
  // Returns the number of completed observations.
  size_t CompleteAllTracks(const Options &options);

  // Merge tracks for specific 3D lines.
  // Returns the number of merged observations.
  size_t MergeTracks(const Options &options,
                     const FlatHashSet<line3D_t> &line3D_ids);

  // Merge tracks of all 3D lines.
  // Returns the number of merged observations.
  size_t MergeAllTracks(const Options &options);

  // Perform retriangulation for under-reconstructed image pairs. Under-
  // reconstruction usually occurs in the case of a drifting reconstruction.
  size_t Retriangulate(const Options &options);

  // Indicate that a 3D line has been modified.
  void AddModifiedLine3D(line3D_t line3D_id);

  // Get changed 3D lines, since the last call to `ClearModifiedLines3D`.
  const FlatHashSet<line3D_t> &GetModifiedLines3D();

  // Clear the collection of changed 3D lines.
  void ClearModifiedLines3D();

  // Data for a correspondence / element of a track, used to store all
  // relevant data for triangulation, in order to avoid duplicate lookup
  // in the underlying unordered_map's in the Reconstruction
  struct LineCorrData {
    colmap::image_t image_id;
    line2D_t line2D_idx;
    const colmap::Image *image;
    const Line2d *line2D;
  };

  // Result of evaluating a single 2D line for triangulation (Phase 1).
  // Contains all read-only results; mutations happen in Phase 2.
  struct TriLineResult {
    line2D_t line2D_idx;
    // For Create: best triangulation proposal
    std::optional<Line3d> best_proposal;
    // Correspondences for building a new track (only those without 3D lines)
    std::vector<LineCorrData> new_corrs;
    // For Continue: best existing 3D line to add this observation to
    line3D_t best_continue_id = kInvalidLine3dId;
    double best_continue_score = 0.0;
    // How many correspondences already had 3D lines
    size_t num_triangulated = 0;
  };

private:
  // Clear cache of merge trials.
  void ClearCaches();

  // Candidates for one 2D line: from the correspondence graph, or from
  // options.exhaustive_match_neighbors when those are supplied.
  void GetCorrespondences(
      const Options &options, colmap::image_t image_id, line2D_t line2D_idx,
      std::vector<colmap::CorrespondenceGraph::Correspondence> *corrs) const;

  // Find (transitive) correspondences to other images.
  size_t Find(const Options &options, colmap::image_t image_id,
              line2D_t line2D_idx, size_t transitivity,
              std::vector<LineCorrData> *corrs_data);

  // Try to create a new 3D line from the given correspondences.
  size_t Create(const Options &options,
                const std::vector<LineCorrData> &corrs_data);

  // Try to continue the 3D line with the given correspondences.
  size_t Continue(const Options &options, const LineCorrData &ref_corr_data,
                  const std::vector<LineCorrData> &corrs_data);

  // Try to merge 3D line with any of its corresponding 3D lines.
  // Linkers are pre-built by callers (MergeTracks/MergeAllTracks) to avoid
  // reconstructing std::function pipelines per call.
  size_t Merge(const Options &options, line3D_t line3D_id,
               const LineLinker2d &merge_linker2d,
               const LineLinker3d &merge_linker3d);

  // Evaluate a single 2D line for triangulation (thread-safe, read-only).
  // Returns proposals and scores without modifying any state.
  TriLineResult EvaluateLine(const Options &options, colmap::image_t image_id,
                             line2D_t line2D_idx) const;

  // Try to transitively complete the track of a 3D line.
  // Linker is pre-built by callers (CompleteTracks/CompleteImage) to avoid
  // reconstructing std::function pipelines per call.
  size_t Complete(const Options &options, line3D_t line3D_id,
                  const LineLinker &linker);

  // Structure correspondence graph for lines.
  std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph_;

  // Reconstruction of the model. Modified when triangulating new lines.
  HolisticReconstruction &reconstruction_;

  // Observation manager for CRUD operations on 3D lines.
  std::shared_ptr<StructureObservationManager> obs_manager_;

  // Cache for tried track merges to avoid duplicate merge trials.
  FlatHashMap<line3D_t, FlatHashSet<line3D_t>> merge_trials_;

  // Cache for found correspondences in the graph.
  std::vector<colmap::CorrespondenceGraph::Correspondence> found_corrs_;

  // Number of trials to retriangulate image pair.
  FlatHashMap<colmap::image_pair_t, int> re_num_trials_;

  // Changed 3D lines, i.e. if a 3D line is modified (created, continued,
  // deleted, merged, etc.). Cleared once `GetModifiedLines3D` is called.
  FlatHashSet<line3D_t> modified_line3D_ids_;
};

// Free function that runs the full incremental line triangulation pipeline.
// Loads structure data, initializes wireframes, creates triangulator, processes
// all registered images, completes/merges tracks, and optionally runs BA.
void IncrementalLineTriangulationPipeline(
    const StructureDatabaseCache &structure_db_cache,
    HolisticReconstruction &reconstruction,
    const IncrementalLineTriangulator::Options &options,
    const estimators::PointLineBundleAdjustmentOptions *ba_options = nullptr);

} // namespace limap
