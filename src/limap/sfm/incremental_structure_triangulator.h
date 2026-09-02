#pragma once

#include <memory>

#include <colmap/scene/correspondence_graph.h>
#include <colmap/sfm/incremental_triangulator.h>
#include <colmap/sfm/observation_manager.h>

#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_correspondence_graph.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/sfm/group_verification.h"
#include "limap/sfm/incremental_group_triangulator.h"
#include "limap/sfm/incremental_line_triangulator.h"
#include "limap/sfm/structure_observation_manager.h"
#include "limap/util/types.h"

namespace limap {

// Unified triangulator that orchestrates points, lines, and groups.
// Follows COLMAP's IncrementalTriangulator pattern.
class IncrementalStructureTriangulator {
public:
  struct Options {
    // Which components to triangulate
    bool triangulate_points = true;
    bool triangulate_lines = true;
    bool triangulate_groups = true;

    // Post-triangulation operations
    bool complete_tracks = true;
    bool merge_tracks = true;

    // Component-specific options
    colmap::IncrementalTriangulator::Options point_options;
    IncrementalLineTriangulator::Options line_options;
    IncrementalGroupTriangulator::Options group_options;

    bool Check() const;
  };

  // Create incremental structure triangulator.
  // Note: All graph and reconstruction objects must live as long as the
  // triangulator.
  // point_corr_graph: COLMAP's correspondence graph for points (shared_ptr)
  // structure_corr_graph: Correspondence graph for lines and groups
  // reconstruction: HolisticReconstruction to modify
  // point_obs_manager: Optional COLMAP ObservationManager for point
  //   triangulation. When provided, the point triangulator will properly
  //   update pair statistics (num_tri_corrs) when creating/modifying points.
  //   This is required when the pipeline also filters points through the
  //   ObservationManager (e.g., incremental SfM), otherwise the counters
  //   become unbalanced and cause "duplicate matches" errors.
  IncrementalStructureTriangulator(
      std::shared_ptr<const colmap::CorrespondenceGraph> point_corr_graph,
      std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph,
      HolisticReconstruction &reconstruction,
      std::shared_ptr<colmap::ObservationManager> point_obs_manager = nullptr);

  // Triangulate observations of image (points, lines, groups in order).
  // Returns total number of triangulated observations.
  size_t TriangulateImage(const Options &options, colmap::image_t image_id);

  // Complete triangulations for image.
  // Returns total number of completed observations.
  size_t CompleteImage(const Options &options, colmap::image_t image_id);

  // Complete all tracks.
  // Returns total number of completed observations.
  size_t CompleteAllTracks(const Options &options);

  // Merge all tracks.
  // Returns total number of merged observations.
  size_t MergeAllTracks(const Options &options);

  // Access individual triangulators
  colmap::IncrementalTriangulator &PointTriangulator();
  IncrementalLineTriangulator &LineTriangulator();
  IncrementalGroupTriangulator &GroupTriangulator();

  const colmap::IncrementalTriangulator &PointTriangulator() const;
  const IncrementalLineTriangulator &LineTriangulator() const;
  const IncrementalGroupTriangulator &GroupTriangulator() const;

  // Modified tracking (union of all modified)
  FlatHashSet<colmap::point3D_t> GetModifiedPoints3D();
  FlatHashSet<line3D_t> GetModifiedLines3D();
  FlatHashSet<group3D_t> GetModifiedGroups3D();
  void ClearModified();

private:
  // Individual triangulators
  std::unique_ptr<colmap::IncrementalTriangulator> point_triangulator_;
  std::unique_ptr<IncrementalLineTriangulator> line_triangulator_;
  std::unique_ptr<IncrementalGroupTriangulator> group_triangulator_;

  // Shared observation manager
  std::shared_ptr<StructureObservationManager> obs_manager_;

  // Reference to reconstruction
  HolisticReconstruction &reconstruction_;

  // Point correspondence graph (shared_ptr for COLMAP's
  // IncrementalTriangulator)
  std::shared_ptr<const colmap::CorrespondenceGraph> point_corr_graph_;
};

// Free function that runs the full incremental structure triangulation
// pipeline. Loads structure data, creates triangulator, processes all
// registered images, and optionally completes/merges tracks.
// Optionally runs bundle adjustment and post-BA group filtering.
void IncrementalTriangulateStructure(
    std::shared_ptr<const colmap::CorrespondenceGraph> point_corr_graph,
    const StructureDatabaseCache &structure_db_cache,
    HolisticReconstruction &reconstruction,
    const IncrementalStructureTriangulator::Options &options,
    const estimators::StructureBundleAdjustmentOptions *ba_options = nullptr,
    const GroupVerificationOptions *filter_options = nullptr);

} // namespace limap
