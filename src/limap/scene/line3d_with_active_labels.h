#pragma once

#include <colmap/scene/image.h>

#include "limap/geometry/line3d.h"

namespace limap {

// Extends Line3d with per-observation active/inactive tagging.
// During incremental SfM, observations can be marked inactive (soft filter)
// instead of permanently deleted. This allows recovery after BA improves poses.
//
// The inactive state is transient (not serialized). When reading from disk,
// all observations default to active.
class Line3dWithActiveLabels : public Line3d {
public:
  using Line3d::Line3d; // inherit constructors
  Line3dWithActiveLabels() = default;
  Line3dWithActiveLabels(const Line3d &line) : Line3d(line) {} // NOLINT

  // Assignment from Line3d (clears inactive state)
  inline Line3dWithActiveLabels &operator=(const Line3d &line);
  inline Line3dWithActiveLabels &operator=(Line3d &&line);

  // Per-observation activity (transient, not serialized)
  inline void SetObservationActive(colmap::image_t image_id);
  inline void SetObservationInactive(colmap::image_t image_id);
  inline bool IsObservationActive(colmap::image_t image_id) const;

  // Track-level reliability metrics
  inline size_t CountActiveObservations() const;
  inline double ActiveRatio() const;
  inline bool IsReliable(size_t min_active = 1) const;

  // Remove inactive entries for observations no longer in track.
  // Called after track.DeleteElement() to stay in sync.
  inline void CleanupInactiveSet();

  // Merge inactive sets from another Line3dWithActiveLabels.
  // Used when merging two lines via MergeLines3D.
  inline void MergeInactiveFrom(const Line3dWithActiveLabels &other);

  // Direct access to the inactive set (for testing/debugging)
  inline const NodeHashSet<colmap::image_t> &InactiveImages() const;

  // Clear all inactive labels (mark all observations as active)
  inline void ClearInactiveLabels();

private:
  NodeHashSet<colmap::image_t> inactive_images_;
};

Line3dWithActiveLabels &Line3dWithActiveLabels::operator=(const Line3d &line) {
  Line3d::operator=(line);
  inactive_images_.clear();
  return *this;
}

Line3dWithActiveLabels &Line3dWithActiveLabels::operator=(Line3d &&line) {
  Line3d::operator=(std::move(line));
  inactive_images_.clear();
  return *this;
}

void Line3dWithActiveLabels::SetObservationActive(colmap::image_t image_id) {
  inactive_images_.erase(image_id);
}

void Line3dWithActiveLabels::SetObservationInactive(colmap::image_t image_id) {
  inactive_images_.insert(image_id);
}

bool Line3dWithActiveLabels::IsObservationActive(
    colmap::image_t image_id) const {
  return inactive_images_.count(image_id) == 0;
}

size_t Line3dWithActiveLabels::CountActiveObservations() const {
  size_t count = 0;
  for (const auto &elem : track.Elements()) {
    if (inactive_images_.count(elem.image_id) == 0) {
      ++count;
    }
  }
  return count;
}

double Line3dWithActiveLabels::ActiveRatio() const {
  if (track.Length() == 0) {
    return 0.0;
  }
  return static_cast<double>(CountActiveObservations()) /
         static_cast<double>(track.Length());
}

bool Line3dWithActiveLabels::IsReliable(size_t min_active) const {
  return CountActiveObservations() >= min_active;
}

void Line3dWithActiveLabels::CleanupInactiveSet() {
  if (inactive_images_.empty()) {
    return;
  }
  FlatHashSet<colmap::image_t> track_images;
  for (const auto &elem : track.Elements()) {
    track_images.insert(elem.image_id);
  }
  for (auto it = inactive_images_.begin(); it != inactive_images_.end();) {
    if (track_images.count(*it) == 0) {
      it = inactive_images_.erase(it);
    } else {
      ++it;
    }
  }
}

void Line3dWithActiveLabels::MergeInactiveFrom(
    const Line3dWithActiveLabels &other) {
  inactive_images_.insert(other.inactive_images_.begin(),
                          other.inactive_images_.end());
}

const NodeHashSet<colmap::image_t> &
Line3dWithActiveLabels::InactiveImages() const {
  return inactive_images_;
}

void Line3dWithActiveLabels::ClearInactiveLabels() { inactive_images_.clear(); }

} // namespace limap
