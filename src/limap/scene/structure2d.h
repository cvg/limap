#pragma once

#include "limap/geometry/line2d.h"
#include "limap/scene/group2d.h"
#include "limap/scene/wireframe.h"
#include "limap/util/types.h"

namespace limap {

// structure2d class for the structure abstraction of one single image
class Structure2d {
public:
  Structure2d();
  Structure2d(const std::vector<Line2d> &lines,
              const std::vector<Group2d> &groups, const Wireframe2d &wireframe,
              point2D_t num_points = 0);

  std::vector<Line2d> &Lines();
  std::vector<Group2d> &Groups();
  Wireframe2d &Wireframe();
  const std::vector<Line2d> &Lines() const;
  const std::vector<Group2d> &Groups() const;
  const Wireframe2d &Wireframe() const;
  size_t NumPoints() const;

  void AddGroup(const Group2d &group);
  void AddGroups(const std::vector<Group2d> &groups);
  void SetLines(const std::vector<Line2d> &newLines);
  void SetGroups(const std::vector<Group2d> &newGroups);
  void SetWireframe(const Wireframe2d &newWireframe);
  void SetNumPoints(point2D_t num_points);

  Line2d &Line(size_t index);
  const Line2d &Line(size_t index) const;
  size_t NumLines() const;

  Group2d &Group(size_t index);
  const Group2d &Group(size_t index) const;
  size_t NumGroups() const;

private:
  // check if the group is valid at construction
  bool CheckValidGroup(const Group2d &group) const;

  std::vector<Line2d> lines_;
  std::vector<Group2d> groups_;
  Wireframe2d wireframe_;
  point2D_t num_points_ = 0;
};

} // namespace limap
