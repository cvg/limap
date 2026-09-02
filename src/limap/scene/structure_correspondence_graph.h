#pragma once

#include <colmap/feature/types.h>
#include <colmap/scene/correspondence_graph.h>
#include <vector>

#include "limap/util/types.h"

namespace limap {

class StructureCorrespondenceGraph {
public:
  colmap::CorrespondenceGraph G_line;
  colmap::CorrespondenceGraph G_group;

  StructureCorrespondenceGraph() = default;
  void AddLineMatches(colmap::image_t image_id1, colmap::image_t image_id2,
                      const LineMatches &matches);
  void AddGroupMatches(colmap::image_t image_id1, colmap::image_t image_id2,
                       const GroupMatches &matches);

  colmap::CorrespondenceGraph &LineGraph();
  const colmap::CorrespondenceGraph &LineGraph() const;
  colmap::CorrespondenceGraph &GroupGraph();
  const colmap::CorrespondenceGraph &GroupGraph() const;
};

} // namespace limap
