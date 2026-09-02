#pragma once

#include <cstddef>
#include <vector>

#include <colmap/scene/reconstruction.h>

#include "limap/geometry/line3d.h"
#include "limap/scene/group3d.h"
#include "limap/scene/group3d_with_active_labels.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_database.h"
#include "limap/util/eigen_types.h"

namespace limap {

// Statistics from group verification (always returned, with passed flag)
struct GroupVerificationStats {
  // Whether the group passed verification thresholds
  bool passed = false;

  // Line verification
  size_t num_line_inliers = 0;
  size_t total_lines = 0;
  double line_inlier_ratio = 0.0;

  // Point verification
  size_t num_point_inliers = 0;
  size_t total_points = 0;
  double point_inlier_ratio = 0.0;

  // Threshold used (in group-specific units)
  double threshold = 0.0;
};

// Options for group verification
struct GroupVerificationOptions {
  // Threshold for inlier determination (-1 = use default below)
  double inlier_threshold = -1.0;

  // Default thresholds when inlier_threshold is -1
  double default_vp_threshold = 10.0;    // degrees (angular error)
  double default_reproj_threshold = 3.0; // pixels (2D reprojection error)

  // Minimum inlier count to validate (for primary feature type)
  size_t min_num_inliers = 3;

  // Minimum inlier ratio to validate
  double min_inlier_ratio = 0.5;

  // For multi-view features: ratio of observations that must pass
  // Feature is inlier if >= obs_inlier_ratio of its views pass
  double obs_inlier_ratio = 0.8;

  // Whether to filter outlier associations from validated groups
  bool filter_outliers = true;
};

// Verify VP group using 3D angular error (lines only)
// Threshold is in degrees (default: 5 deg)
// No camera access needed - pure 3D geometry
// Marks outlier line associations inactive. Always returns stats with
// passed flag indicating whether thresholds were met.
GroupVerificationStats
VerifyVPGroup(Group3dWithActiveLabels &group, const std::vector<Line3d> &lines,
              const GroupVerificationOptions &options = {});

// General group verification using 2D reprojection error
// Threshold is in pixels (default: 3 px)
// For each associated feature:
//   - Get its track (multi-view observations)
//   - For each view: project 3D feature → group surface → camera → 2D
//   - Compare to 2D observation, count passing views
//   - Feature is inlier if >= obs_inlier_ratio views pass
// Marks outlier associations inactive. Always returns stats with
// passed flag indicating whether thresholds were met.
GroupVerificationStats
VerifyGroupByReprojError(Group3dWithActiveLabels &group,
                         const HolisticReconstruction &recon,
                         const GroupVerificationOptions &options = {});

// Dispatch to appropriate verifier based on group type
// VP: uses VerifyVPGroup (3D angular error, special case)
// All others: uses VerifyGroupByReprojError (2D reprojection error)
// Always returns stats with passed flag.
GroupVerificationStats
VerifyGroup(Group3dWithActiveLabels &group, const HolisticReconstruction &recon,
            const std::vector<V3D> &points, const std::vector<Line3d> &lines,
            const GroupVerificationOptions &options = {});

// ============================================================================
// Batch Group Filtering (Post-BA)
// ============================================================================

// Statistics from filtering all groups
struct FilterGroupsStats {
  size_t num_groups_processed = 0;    // Total groups processed
  size_t num_groups_passed = 0;       // Groups that passed verification
  size_t num_groups_failed = 0;       // Groups that failed verification
  size_t num_associations_marked = 0; // Associations marked inactive
  size_t num_associations_purged =
      0; // Inactive associations permanently removed
};

// Soft-filter all Group3D associations in the reconstruction by error.
// For each group:
//   - Removes stale references to deleted points/lines
//   - Verifies using appropriate metric (marks bad associations inactive)
//   - Groups with >= min_active_for_purge active associations have their
//     inactive associations permanently removed (PurgeInactiveAssociations)
//   - Groups below the purge threshold keep all associations (even inactive)
// Never deletes groups. Returns statistics.
FilterGroupsStats
FilterGroupAssociations(HolisticReconstruction &recon,
                        const GroupVerificationOptions &options = {},
                        size_t min_active_for_purge = 10);

// Delete groups with zero active associations or track length < 2.
// Cleans up 2D references for deleted groups.
// Returns number of groups deleted.
size_t DeleteSupportlessGroups(HolisticReconstruction &recon);

// ============================================================================
// VP Match Geometric Verification
// ============================================================================

// Verify all VP matches in the database and update in place.
// For each image pair, filters out VP matches that fail geometric verification.
// Returns total number of removed matches.
size_t VerifyVPMatches(const colmap::Reconstruction &recon,
                       StructureDatabase &structure_db,
                       double threshold_degrees = 10.0);

// ============================================================================
// Plane Match Geometric Verification
// ============================================================================

// Verify all plane matches in the database and update in place.
// For each image pair, filters out plane matches where the camera-frame normals
// (stored as Group2d params) project to inconsistent world-frame directions.
// Returns total number of removed matches.
size_t VerifyPlaneMatches(const colmap::Reconstruction &recon,
                          StructureDatabase &structure_db,
                          double threshold_degrees = 30.0);

} // namespace limap
