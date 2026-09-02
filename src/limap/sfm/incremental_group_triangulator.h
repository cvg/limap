#pragma once

#include <map>
#include <memory>
#include <vector>

#include <colmap/scene/correspondence_graph.h>

#include "limap/geometry/groups.h"
#include "limap/scene/group3d.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/scene/structure_correspondence_graph.h"
#include "limap/sfm/structure_observation_manager.h"
#include "limap/util/types.h"

namespace limap {

// Class that triangulates groups during incremental reconstruction.
// Follows COLMAP's IncrementalTriangulator pattern.
class IncrementalGroupTriangulator {
public:
  struct Options {
    // Transitivity for correspondence search
    int max_transitivity = 1;

    // Minimum correspondences to create Group3D
    int min_num_observations = 2;

    // Minimum correspondences containing a feature to associate it
    int min_num_supports_for_association = 2;

    // RANSAC inlier threshold for group parameter initialization, per group
    // type. Units depend on group type: radians for VP, world units for
    // Plane/Sphere. If a group type is not in the map or its value is <= 0,
    // uses the group type's default threshold.
    std::map<GroupType, double> init_ransac_thresh = {};

    // Verification threshold per group type.
    // Units depend on group type: degrees for VP, pixels for Plane/Sphere/etc.
    // If a group type is not in the map or its value is <= 0, uses default.
    std::map<GroupType, double> verification_threshold = {};

    // Default thresholds when verification_threshold is not set for a group
    // type.
    double default_vp_threshold = 10.0;    // degrees
    double default_reproj_threshold = 3.0; // pixels

    // Verification criteria
    size_t verification_min_num_inliers = 3;
    double verification_min_inlier_ratio = 0.5;
    double verification_obs_inlier_ratio = 0.8;
    bool verification_filter_outliers = true;

    // Continue/Complete/Merge error thresholds
    double continue_max_reproj_error = 2.0; // pixels for Plane 2D verification
    double continue_max_angle_error_vp = 5.0; // degrees for VP 3D angular check
    double complete_max_reproj_error = 2.0;
    double complete_max_angle_error_vp = 5.0;
    int complete_max_transitivity = 5;
    double merge_max_reproj_error = 2.0;   // pixels for Plane 2D verification
    double merge_max_angle_error_vp = 5.0; // degrees for VP 3D angular check

    // Maximum ratio of merged 3D fitting error to pre-merge error.
    // Rejects merge if the merged group's median distance-to-surface exceeds
    // this factor times the worse of the two individual groups' errors.
    // Applies to all non-VP group types (Plane, Sphere, etc.).
    double merge_max_error_ratio = 2.0;

    // Maximum angular deviation (degrees) between each individual plane's
    // normal and the merged plane's normal. Rejects merge if either normal
    // deviates by more than this. Only applies to PLANE groups.
    double merge_max_plane_normal_angle = 15.0;

    // Minimum number of associated features (points + lines) before running
    // 2D refinement and filtering in UpdateGroupAssociations.
    // Below this threshold, associations are kept without optimization.
    int association_filter_min_num_features = 10;

    // Reprojection error threshold (pixels) for filtering outlier associations
    // in UpdateGroupAssociations. Looser than post-BA default_reproj_threshold
    // because point positions and group params are not yet jointly refined.
    double association_filter_reproj_threshold = 3.0;

    // Minimum active associations for a group's params to be considered
    // BA-optimized. In UpdateGroupAssociations, groups at or above this
    // threshold keep their existing params (verified only); groups below
    // get full RANSAC re-initialization + refinement.
    // Should match GroupBundleAdjustmentOptions::min_active_group_associations.
    size_t min_active_to_keep_params = 10;

    // MAD-based adaptive 3D distance threshold for Continue/Complete.
    // For non-VP groups, computes distance-to-surface for all existing
    // associated points, then sets threshold = median + k * MAD where
    // k = surface_dist_mad_factor. Rejects observations whose points
    // have median distance exceeding this threshold.
    // Set to 0 to disable (only use 2D reprojection check).
    double surface_dist_mad_factor = 3.0;

    // Whether to ignore two-view tracks
    bool ignore_two_view_tracks = true;

    bool Check() const;

    // Returns the RANSAC threshold for a specific group type.
    // Returns -1.0 if not set (meaning use default).
    double GetInitRansacThresh(GroupType type) const {
      auto it = init_ransac_thresh.find(type);
      if (it != init_ransac_thresh.end()) {
        return it->second;
      }
      return -1.0;
    }

    // Returns the verification threshold for a specific group type.
    // Returns -1.0 if not set (meaning use default).
    double GetVerificationThresh(GroupType type) const {
      auto it = verification_threshold.find(type);
      if (it != verification_threshold.end()) {
        return it->second;
      }
      return -1.0;
    }
  };

  // Create incremental group triangulator. Note that both the correspondence
  // graph and the reconstruction objects must live as long as the triangulator.
  // If obs_manager is not provided, one will be created internally.
  IncrementalGroupTriangulator(
      std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph,
      HolisticReconstruction &reconstruction,
      std::shared_ptr<StructureObservationManager> obs_manager = nullptr);

  // Triangulate observations of image.
  //
  // Triangulation includes creation of new groups, continuation of existing
  // groups, and merging of separate groups if given image bridges tracks.
  //
  // Note that the given image must be registered and its pose must be set
  // in the associated reconstruction. Also, points and lines should be
  // triangulated first as groups depend on them.
  size_t TriangulateImage(const Options &options, colmap::image_t image_id);

  // Complete triangulations for image. Tries to create new tracks for not
  // yet triangulated observations and tries to complete existing tracks.
  // Returns the number of completed observations.
  size_t CompleteImage(const Options &options, colmap::image_t image_id);

  // Complete tracks for specific 3D groups.
  //
  // Completion tries to recursively add observations to a track that might
  // have failed to triangulate before due to inaccurate poses, etc.
  // Returns the number of completed observations.
  size_t CompleteTracks(const Options &options,
                        const FlatHashSet<group3D_t> &group3D_ids);

  // Complete tracks of all 3D groups.
  // Returns the number of completed observations.
  size_t CompleteAllTracks(const Options &options);

  // Merge tracks for specific 3D groups.
  // Returns the number of merged observations.
  size_t MergeTracks(const Options &options,
                     const FlatHashSet<group3D_t> &group3D_ids);

  // Merge tracks of all 3D groups.
  // Returns the number of merged observations.
  size_t MergeAllTracks(const Options &options);

  // Update feature associations and re-estimate parameters for given groups.
  // For each group, rebuilds point/line associations from all track
  // observations and re-fits parameters via RANSAC.
  // Should be called after TriangulateImage to pick up newly triangulated
  // points/lines.
  // Returns the number of groups whose params were updated.
  size_t UpdateGroupAssociations(const Options &options,
                                 const FlatHashSet<group3D_t> &group3D_ids);

  // Indicate that a 3D group has been modified.
  void AddModifiedGroup3D(group3D_t group3D_id);

  // Get changed 3D groups, since the last call to `ClearModifiedGroups3D`.
  const FlatHashSet<group3D_t> &GetModifiedGroups3D();

  // Clear the collection of changed 3D groups.
  void ClearModifiedGroups3D();

  // Data for a correspondence / element of a track, used to store all
  // relevant data for triangulation, in order to avoid duplicate lookup
  // in the underlying unordered_map's in the Reconstruction
  struct GroupCorrData {
    colmap::image_t image_id;
    group2D_t group2D_idx;
    const colmap::Image *image;
    const Group2d *group2D;
  };

private:
  // Clear cache of merge trials.
  void ClearCaches();

  // Find (transitive) correspondences to other images.
  // Returns the count of correspondences that already have Group3D.
  size_t Find(const Options &options, colmap::image_t image_id,
              group2D_t group2D_idx, size_t transitivity,
              std::vector<GroupCorrData> *corrs_data);

  // Try to create a new 3D group from the given correspondences.
  size_t Create(const Options &options,
                const std::vector<GroupCorrData> &corrs_data);

  // Try to continue the 3D group with the given correspondences.
  size_t Continue(const Options &options, const GroupCorrData &ref_corr_data,
                  const std::vector<GroupCorrData> &corrs_data);

  // Try to merge 3D group with any of its corresponding 3D groups.
  size_t Merge(const Options &options, group3D_t group3D_id);

  // Try to transitively complete the track of a 3D group.
  size_t Complete(const Options &options, group3D_t group3D_id);

  // Helper: Extract 3D points from a 2D group observation.
  void Extract3DPoints(const GroupCorrData &g,
                       std::vector<colmap::point3D_t> &pts) const;

  // Helper: Extract 3D lines from a 2D group observation.
  void Extract3DLines(const GroupCorrData &g,
                      std::vector<line3D_t> &lines) const;

  // Helper: Aggregate shared 3D features from all correspondences.
  // Returns point/line IDs with support >= min_support.
  void
  AggregateSharedFeatures(const std::vector<GroupCorrData> &corrs_data,
                          int min_support,
                          NodeHashMap<colmap::point3D_t, int> &point_support,
                          NodeHashMap<line3D_t, int> &line_support) const;

  // Helper: Check if observation can be added to existing group
  // (Continue/Complete). For VP: checks angular error between line directions.
  // For others: checks 2D reprojection error and adaptive 3D distance.
  bool CheckObservationError(const Group3d &group3d, const GroupCorrData &corr,
                             double max_angle_error_vp, double max_reproj_error,
                             double surface_dist_mad_factor = 3.0) const;

  // Helper: Check if two groups can be merged (same type, similar params).
  // Returns true if groups are compatible for merging.
  bool CheckMergeCompatibility(const Options &options, const Group3d &group1,
                               const Group3d &group2,
                               const std::vector<V3D> &merged_points,
                               const std::vector<Line3d> &merged_lines) const;

  // Helper: Add new 3D feature associations from an observation to a group.
  // Extracts triangulated points/lines from the observation and adds any
  // that are not already associated with the group.
  void UpdateGroupFeatures(group3D_t group3D_id, const GroupCorrData &new_obs);

  // Structure correspondence graph for groups.
  std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph_;

  // Reconstruction of the model. Modified when triangulating new groups.
  HolisticReconstruction &reconstruction_;

  // Observation manager for CRUD operations on 3D groups.
  std::shared_ptr<StructureObservationManager> obs_manager_;

  // Cache for tried track merges to avoid duplicate merge trials.
  FlatHashMap<group3D_t, FlatHashSet<group3D_t>> merge_trials_;

  // Cache for found correspondences in the graph.
  std::vector<colmap::CorrespondenceGraph::Correspondence> found_corrs_;

  // Changed 3D groups, i.e. if a 3D group is modified (created, continued,
  // deleted, merged, etc.). Cleared once `GetModifiedGroups3D` is called.
  FlatHashSet<group3D_t> modified_group3D_ids_;
};

} // namespace limap
