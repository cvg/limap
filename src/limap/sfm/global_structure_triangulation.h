#pragma once

#include <memory>

#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/sfm/global_group_triangulation.h"
#include "limap/sfm/global_line_triangulation.h"
#include "limap/sfm/group_verification.h"

namespace limap {

// Free function that runs the full global structure triangulation pipeline.
// Loads structure data, initializes wireframes, then runs global line
// triangulation followed by global group triangulation.
// Optionally runs bundle adjustment and post-BA group filtering.
void GlobalTriangulateStructure(
    const StructureDatabaseCache &structure_db_cache,
    const std::shared_ptr<HolisticReconstruction> &reconstruction,
    const GlobalLineTriangulationOptions &line_options,
    const GlobalGroupTriangulationOptions &group_options,
    const estimators::StructureBundleAdjustmentOptions *ba_options = nullptr,
    const GroupVerificationOptions *filter_options = nullptr,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors = {});

} // namespace limap
