#pragma once

#include <ceres/loss_function.h>
#include <colmap/scene/correspondence_graph.h>
#include <colmap/scene/track.h>
#include <memory>
#include <vector>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/scene/group3d.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/util/types.h"

namespace limap {

// Manages observations for 3D lines and groups stored in
// StructureReconstruction. Provides CRUD operations (aligned with COLMAP's
// ObservationManager pattern) and filtering methods. All methods update
// Lines3D() in-place.
class StructureObservationManager {
public:
  explicit StructureObservationManager(StructureReconstruction &srec)
      : srec_(srec) {}

  // Access to underlying reconstruction
  const StructureReconstruction &StructureRecon() const { return srec_; }
  StructureReconstruction &StructureRecon() { return srec_; }

  // ============ Line CRUD Operations (aligned with COLMAP ObservationManager)
  // =

  // Add a new 3D line with its track. Returns the new line3D_id.
  // Updates 2D->3D assignments for all track elements.
  line3D_t AddLine3D(const Line3d &line3d);

  // Add a line observation to an existing 3D line.
  // Updates 2D->3D assignment for the observation.
  void AddLineObservation(line3D_t line3D_id,
                          const colmap::TrackElement &track_el);

  // Delete a 3D line and clear all its 2D->3D assignments.
  void DeleteLine3D(line3D_t line3D_id);

  // Delete a line observation from its associated 3D line.
  // If the track becomes too short (< 2 elements), the 3D line is deleted.
  void DeleteLineObservation(colmap::image_t image_id, line2D_t line2D_idx);

  // Merge two 3D lines into one. Returns the ID of the merged line.
  // The merged line geometry is a weighted average by track length.
  // Updates all 2D->3D assignments.
  line3D_t MergeLines3D(line3D_t line3D_id1, line3D_t line3D_id2);

  // ============ Group CRUD Operations (aligned with COLMAP ObservationManager)
  // =

  // Add a new 3D group with its track. Returns the new group3D_id.
  // Updates 2D->3D assignments for all track elements.
  group3D_t AddGroup3D(const Group3d &group3d);

  // Add a group observation to an existing 3D group.
  // Updates 2D->3D assignment for the observation.
  void AddGroupObservation(group3D_t group3D_id,
                           const colmap::TrackElement &track_el);

  // Delete a 3D group and clear all its 2D->3D assignments.
  void DeleteGroup3D(group3D_t group3D_id);

  // Delete a group observation from its associated 3D group.
  // If the track becomes too short (< 2 elements), the 3D group is deleted.
  void DeleteGroupObservation(colmap::image_t image_id, group2D_t group2D_idx);

  // Merge two 3D groups into one. Returns the ID of the merged group.
  // Merges tracks and feature associations (union with max weights).
  // Note: Params are NOT merged - caller must re-initialize params after merge.
  // Updates all 2D->3D assignments.
  group3D_t MergeGroups3D(group3D_t group3D_id1, group3D_t group3D_id2);

  // ============ Frame De-registration =======================================

  // Remove all structure observations (lines + groups) for a frame's images.
  // Mirrors COLMAP's ObservationManager::DeRegisterFrame for structure data.
  // Does NOT touch the point reconstruction — that's COLMAP's responsibility.
  void DeRegisterFrame(colmap::frame_t frame_id);

  // ============ Filtering Operations ========================================

  // Remove 3D lines whose 2D observations fail reprojection checks
  // (angular + perpendicular distance). Also prunes bad supports from tracks.
  void FilterLines3dByReprojection(double th_angular_2d, double th_perp_2d,
                                   int num_outliers = 2);

  // Remove 3D lines whose camera-frame angular sensitivity is too high.
  void FilterLines3dBySensitivity(double th_sensitivity_3d, int min_support_ns);

  // Remove 3D lines whose projected/observed 2D overlap is too low.
  void FilterLines3dByOverlap(double th_overlap, int min_support_ns);

  // Remove 3D lines that appear in fewer than min_visible_views images.
  void FilterLines3dByMinVisibleViews(int min_visible_views);

  // Filter line observations by reprojection error, using proper CRUD
  // operations to maintain 2D->3D consistency. Unlike
  // FilterLines3dByReprojection which replaces the lines3d map wholesale, this
  // method uses DeleteLineObservation for each bad observation and returns the
  // number of filtered observations (analogous to COLMAP's FilterAllPoints3D).
  size_t FilterAllLines3D(double max_angular_error, double max_perp_error);

  // ============ Active/Inactive Operations (soft filtering) ================

  // Check if a line track is reliable (enough active observations).
  bool IsReliableTrack(line3D_t line3D_id, size_t min_active = 1) const;

  // Classify all line tracks into reliable and unreliable sets.
  // reliable: >= min_active_observations active observations
  // unreliable: < min_active_observations active observations
  void ClassifyLineTracks(size_t min_active_observations,
                          FlatHashSet<line3D_t> &reliable_ids,
                          FlatHashSet<line3D_t> &unreliable_ids) const;

  // Enhanced classification: first by observation count, then by pixel
  // uncertainty. Lines with pixel uncertainty (std dev) above threshold are
  // demoted from reliable to unreliable. Set pixel_uncertainty_threshold=0 to
  // disable.
  void ClassifyLineTracks(size_t min_active_observations,
                          double pixel_uncertainty_threshold,
                          const ceres::LossFunction *loss_function,
                          FlatHashSet<line3D_t> &reliable_ids,
                          FlatHashSet<line3D_t> &unreliable_ids) const;

  // Soft filtering: mark observations as inactive instead of deleting.
  // Same logic as FilterAllLines3D, but uses SetObservationInactive instead
  // of DeleteLineObservation. Returns number of newly inactivated observations.
  size_t UpdateLineObservationActivity(double max_angular_error,
                                       double max_perp_error);

  // Re-evaluate all observations against refined 3D lines.
  // Marks observations active/inactive based on current error thresholds.
  // Can reactivate previously inactive observations that now pass.
  // Returns number of observations that changed status.
  size_t UpdateActiveSupports(double max_angular_error, double max_perp_error);

  // Filter line tracks:
  // 1. Marks all observations active/inactive based on reprojection error
  //    (can both activate and deactivate, like UpdateActiveSupports)
  // 2. For tracks with >10 active images: hard-deletes individual inactive
  //    observations from the track (keeps the track clean)
  // 3. Deletes entire tracks with <2 observations, or (if
  //    min_active_ratio_for_deletion > 0) with active ratio below threshold
  // If local_image_ids is non-null, only filters tracks visible in those
  // images. Returns number of deleted tracks.
  size_t FilterLineTracks(
      double max_angular_error, double max_perp_error,
      double min_active_ratio_for_deletion = 0,
      const FlatHashSet<colmap::image_t> *local_image_ids = nullptr);

  // Hard-delete tracks with track.Length() < 2 or (if min_active_ratio > 0)
  // active ratio below threshold. Returns number of deleted tracks.
  // When min_active_ratio=0, only deletes tracks with <2 observations.
  size_t DeleteSupportlessLineTracks(double min_active_ratio);

private:
  StructureReconstruction &srec_;
};

} // namespace limap
