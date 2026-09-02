#pragma once

#include "limap/scene/group3d.h"
#include "limap/util/types.h"

namespace limap {

// Extends Group3d with per-association active/inactive tagging.
// During incremental SfM, associations can be marked inactive (soft filter)
// instead of permanently deleted. This allows recovery after BA improves
// geometry.
//
// The inactive state is transient (not serialized). When reading from disk,
// all associations default to active.
class Group3dWithActiveLabels : public Group3d {
public:
  using Group3d::Group3d; // inherit constructors
  Group3dWithActiveLabels() = default;
  Group3dWithActiveLabels(const Group3d &group) : Group3d(group) {} // NOLINT

  // Assignment from Group3d (clears inactive state)
  inline Group3dWithActiveLabels &operator=(const Group3d &group);
  inline Group3dWithActiveLabels &operator=(Group3d &&group);

  // Per-association activity (transient, not serialized)
  inline void SetPointActive(feature3D_t id);
  inline void SetPointInactive(feature3D_t id);
  inline bool IsPointActive(feature3D_t id) const;

  inline void SetLineActive(feature3D_t id);
  inline void SetLineInactive(feature3D_t id);
  inline bool IsLineActive(feature3D_t id) const;

  // Association-level metrics
  inline size_t CountActivePoints() const;
  inline size_t CountActiveLines() const;
  inline size_t CountActiveAssociations() const;

  // Remove inactive entries for associations no longer in points/lines vectors.
  // Called after association vectors are modified externally.
  inline void CleanupInactiveSets();

  // Clear all inactive labels (mark all associations as active)
  inline void ClearInactiveLabels();

  // Hard-delete inactive associations from points/lines vectors.
  // Returns count of removed associations.
  inline size_t PurgeInactiveAssociations();

  // Direct access to inactive sets (for testing/debugging)
  inline const NodeHashSet<feature3D_t> &InactivePointIds() const;
  inline const NodeHashSet<feature3D_t> &InactiveLineIds() const;

private:
  NodeHashSet<feature3D_t> inactive_point_ids_;
  NodeHashSet<feature3D_t> inactive_line_ids_;
};

Group3dWithActiveLabels &
Group3dWithActiveLabels::operator=(const Group3d &group) {
  Group3d::operator=(group);
  inactive_point_ids_.clear();
  inactive_line_ids_.clear();
  return *this;
}

Group3dWithActiveLabels &Group3dWithActiveLabels::operator=(Group3d &&group) {
  Group3d::operator=(std::move(group));
  inactive_point_ids_.clear();
  inactive_line_ids_.clear();
  return *this;
}

void Group3dWithActiveLabels::SetPointActive(feature3D_t id) {
  inactive_point_ids_.erase(id);
}

void Group3dWithActiveLabels::SetPointInactive(feature3D_t id) {
  inactive_point_ids_.insert(id);
}

bool Group3dWithActiveLabels::IsPointActive(feature3D_t id) const {
  return inactive_point_ids_.count(id) == 0;
}

void Group3dWithActiveLabels::SetLineActive(feature3D_t id) {
  inactive_line_ids_.erase(id);
}

void Group3dWithActiveLabels::SetLineInactive(feature3D_t id) {
  inactive_line_ids_.insert(id);
}

bool Group3dWithActiveLabels::IsLineActive(feature3D_t id) const {
  return inactive_line_ids_.count(id) == 0;
}

size_t Group3dWithActiveLabels::CountActivePoints() const {
  size_t count = 0;
  for (const auto &af : points) {
    if (inactive_point_ids_.count(af.idx) == 0) {
      ++count;
    }
  }
  return count;
}

size_t Group3dWithActiveLabels::CountActiveLines() const {
  size_t count = 0;
  for (const auto &af : lines) {
    if (inactive_line_ids_.count(af.idx) == 0) {
      ++count;
    }
  }
  return count;
}

size_t Group3dWithActiveLabels::CountActiveAssociations() const {
  return CountActivePoints() + CountActiveLines();
}

void Group3dWithActiveLabels::CleanupInactiveSets() {
  if (inactive_point_ids_.empty() && inactive_line_ids_.empty()) {
    return;
  }
  // Build sets of current association IDs
  FlatHashSet<feature3D_t> current_point_ids;
  for (const auto &af : points) {
    current_point_ids.insert(af.idx);
  }
  for (auto it = inactive_point_ids_.begin();
       it != inactive_point_ids_.end();) {
    if (current_point_ids.count(*it) == 0) {
      it = inactive_point_ids_.erase(it);
    } else {
      ++it;
    }
  }

  FlatHashSet<feature3D_t> current_line_ids;
  for (const auto &af : lines) {
    current_line_ids.insert(af.idx);
  }
  for (auto it = inactive_line_ids_.begin(); it != inactive_line_ids_.end();) {
    if (current_line_ids.count(*it) == 0) {
      it = inactive_line_ids_.erase(it);
    } else {
      ++it;
    }
  }
}

void Group3dWithActiveLabels::ClearInactiveLabels() {
  inactive_point_ids_.clear();
  inactive_line_ids_.clear();
}

size_t Group3dWithActiveLabels::PurgeInactiveAssociations() {
  size_t removed = 0;

  if (!inactive_point_ids_.empty()) {
    size_t before = points.size();
    points.erase(std::remove_if(points.begin(), points.end(),
                                [this](const AssociatedFeature3d &af) {
                                  return inactive_point_ids_.count(af.idx) > 0;
                                }),
                 points.end());
    removed += before - points.size();
    inactive_point_ids_.clear();
  }

  if (!inactive_line_ids_.empty()) {
    size_t before = lines.size();
    lines.erase(std::remove_if(lines.begin(), lines.end(),
                               [this](const AssociatedFeature3d &af) {
                                 return inactive_line_ids_.count(af.idx) > 0;
                               }),
                lines.end());
    removed += before - lines.size();
    inactive_line_ids_.clear();
  }

  return removed;
}

const NodeHashSet<feature3D_t> &
Group3dWithActiveLabels::InactivePointIds() const {
  return inactive_point_ids_;
}

const NodeHashSet<feature3D_t> &
Group3dWithActiveLabels::InactiveLineIds() const {
  return inactive_line_ids_;
}

} // namespace limap
