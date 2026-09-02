#include "limap/scene/structure2d.h"

#include <colmap/util/logging.h>

namespace limap {

Structure2d::Structure2d() = default;

Structure2d::Structure2d(const std::vector<Line2d> &lines,
                         const std::vector<Group2d> &groups,
                         const Wireframe2d &wireframe, point2D_t num_points)
    : lines_(lines), groups_(groups), wireframe_(wireframe),
      num_points_(num_points) {
  for (const auto &group : groups) {
    THROW_CHECK(CheckValidGroup(group))
        << "Group is not valid. Index out of bound";
  }
}

std::vector<Line2d> &Structure2d::Lines() { return lines_; }

std::vector<Group2d> &Structure2d::Groups() { return groups_; }

Wireframe2d &Structure2d::Wireframe() { return wireframe_; }

const std::vector<Line2d> &Structure2d::Lines() const { return lines_; }

const std::vector<Group2d> &Structure2d::Groups() const { return groups_; }

const Wireframe2d &Structure2d::Wireframe() const { return wireframe_; }

size_t Structure2d::NumPoints() const { return num_points_; }

void Structure2d::SetLines(const std::vector<Line2d> &newLines) {
  lines_ = newLines;
}

void Structure2d::AddGroup(const Group2d &group) {
  THROW_CHECK(CheckValidGroup(group))
      << "Group is not valid. Index out of bound";
  groups_.push_back(group);
}

void Structure2d::AddGroups(const std::vector<Group2d> &groups) {
  for (const auto &group : groups) {
    AddGroup(group);
  }
}

void Structure2d::SetGroups(const std::vector<Group2d> &newGroups) {
  for (const auto &group : newGroups) {
    THROW_CHECK(CheckValidGroup(group))
        << "Group is not valid. Index out of bound";
  }
  groups_ = newGroups;
}

void Structure2d::SetWireframe(const Wireframe2d &newWireframe) {
  wireframe_ = newWireframe;
}

void Structure2d::SetNumPoints(point2D_t num_points) {
  num_points_ = num_points;
}

Line2d &Structure2d::Line(size_t index) {
  THROW_CHECK_LE(index, lines_.size()) << "Line index out of range";
  return lines_[index];
}

const Line2d &Structure2d::Line(size_t index) const {
  THROW_CHECK_LE(index, lines_.size()) << "Line index out of range";
  return lines_[index];
}

size_t Structure2d::NumLines() const { return lines_.size(); }

Group2d &Structure2d::Group(size_t index) {
  THROW_CHECK_LE(index, groups_.size()) << "Group index out of range";
  return groups_[index];
}

const Group2d &Structure2d::Group(size_t index) const {
  THROW_CHECK_LE(index, groups_.size()) << "Group index out of range";
  return groups_[index];
}

size_t Structure2d::NumGroups() const { return groups_.size(); }

bool Structure2d::CheckValidGroup(const Group2d &group) const {
  auto max_point_id = group.GetMaxPointId();
  if (max_point_id.has_value()) {
    if (*max_point_id >= NumPoints()) {
      return false;
    }
  }
  auto max_line_id = group.GetMaxLineId();
  if (max_line_id.has_value()) {
    if (*max_line_id >= NumLines()) {
      return false;
    }
  }
  return true;
}

} // namespace limap
