#pragma once

#include <map>

#include "limap/geometry/line_linker.h"
#include "limap/scene/holistic_reconstruction.h"

namespace limap {

// Options for merging pre-fitted Line3d (geometry-guided reconstruction).
struct LineMergingOptions {
  LineLinker2dOptions linker2d_options;
  LineLinker3dOptions linker3d_options;         // For initial merge
  LineLinker3dOptions linker3d_remerge_options; // For spatial remerge
  int num_outliers_aggregator = 2;
  bool enable_remerge = true;
  double filtering2d_th_angular_2d = 8.0; // degrees
  double filtering2d_th_perp_2d = 5.0;    // pixels
  int min_visible_views = 2;

  // Number of threads for parallel edge detection (-1 = all available)
  int num_threads = -1;

  LineMergingOptions() {
    // Initial merge thresholds (from reference config)
    linker3d_options.th_angle = 8.0;
    linker3d_options.th_overlap = 0.01;
    linker3d_options.th_perp = 0.75;
    linker3d_options.th_innerseg = 0.75;
    // Remerge thresholds (tighter, from reference config)
    linker3d_remerge_options.th_angle = 5.0;
    linker3d_remerge_options.th_overlap = 0.001;
    linker3d_remerge_options.th_perp = 0.5;
    linker3d_remerge_options.th_innerseg = 0.5;
  }

  bool Check() const;
};

// Merge pre-fitted Line3d based on 2D+3D similarity. Modifies recon in-place.
void MergeFittedLines3D(
    HolisticReconstruction &recon,
    const std::map<colmap::image_t, std::vector<colmap::image_t>> &neighbors,
    const LineMergingOptions &options);

} // namespace limap
