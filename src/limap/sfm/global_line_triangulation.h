#pragma once

#include <map>
#include <memory>
#include <mutex>

#include <colmap/math/union_find.h>
#include <colmap/scene/correspondence_graph.h>
#include <colmap/util/base_controller.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/geometry/line_linker.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/sfm/line_triangulation_utils.h"
#include "limap/util/eigen_types.h"
#include "limap/util/types.h"

namespace limap {

struct GlobalLineTriangulationOptions {
  // Wireframe
  double wireframe2d_th = 2.0; // pixels

  // Triangulation variants
  bool use_point_assisted_triangulation = true;
  bool use_vp_assisted_triangulation = true;

  // Hyperparameters
  double min_length_2d = 20.0;
  double IoU_threshold = 0.1;
  double sensitivity_threshold = 70.0; // degrees
  double var2d = 2.0;

  // Scoring
  double fullscore_th = 1.0;
  double scale_inv_th = 0.01;

  // Aggregation
  int num_outliers_aggregator = 2;

  // Linker options
  LineLinker2dOptions linker2d_options;
  LineLinker3dOptions linker3d_options;

  // Enable spatial remerging of 3D lines after initial triangulation
  bool enable_remerge = true;

  // === Filtering options ===
  // Reprojection
  double filtering2d_th_angular_2d = 8.0; // degrees
  double filtering2d_th_perp_2d = 5.0;    // pixels

  // Sensitivity
  double filtering2d_th_sv_angular_3d = 75.0; // degrees
  int filtering2d_th_sv_num_supports = 3;

  // Overlap
  double filtering2d_th_overlap = 0.5;
  int filtering2d_th_overlap_num_supports = 3;

  // Minimum visible views
  int min_visible_views = 2;

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

class GlobalLineTriangulationController : public colmap::BaseController {
public:
  GlobalLineTriangulationController(
      const GlobalLineTriangulationOptions &options,
      const std::shared_ptr<HolisticReconstruction> &recon,
      const colmap::CorrespondenceGraph &line_corr_graph,
      const ExhaustiveMatchNeighbors &exhaustive_match_neighbors = {});
  // Supplying them selects exhaustive matching instead of corr_graph.

  void Run() override;

private:
  void TriangulateAndScoreNode(const Node2d &node);
  void MergeTriangulationsIntoTracks();
  void RemergeLines3D();

  GlobalLineTriangulationOptions options_;

  std::shared_ptr<HolisticReconstruction> recon_;
  const colmap::CorrespondenceGraph &corr_graph_;
  ExhaustiveMatchNeighbors exhaustive_match_neighbors_;

  Node2dMap<LineTriangulationProposal> best_tris_;
  std::mutex best_tris_mutex_;
};

// Free function that runs the full global line triangulation pipeline.
// Loads structure data, initializes wireframes, runs global line
// triangulation, and optionally runs bundle adjustment.
void GlobalLineTriangulationPipeline(
    const StructureDatabaseCache &structure_db_cache,
    const std::shared_ptr<HolisticReconstruction> &reconstruction,
    const GlobalLineTriangulationOptions &options,
    const estimators::PointLineBundleAdjustmentOptions *ba_options = nullptr,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors = {});

} // namespace limap
