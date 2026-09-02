#include "limap/scene/structure_correspondence_graph.h"

#include <colmap/scene/two_view_geometry.h>
#include <colmap/util/logging.h>

namespace limap {

void StructureCorrespondenceGraph::AddLineMatches(colmap::image_t image_id1,
                                                  colmap::image_t image_id2,
                                                  const LineMatches &matches) {
  colmap::FeatureMatches colmap_matches;
  for (const auto &match : matches) {
    colmap_matches.emplace_back(match.line2D_idx1, match.line2D_idx2);
  }
  THROW_CHECK_EQ(G_line.NumMatchesBetweenImages(image_id1, image_id2), 0);
  colmap::TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = std::move(colmap_matches);
  G_line.AddTwoViewGeometry(image_id1, image_id2, std::move(two_view_geometry));
}

void StructureCorrespondenceGraph::AddGroupMatches(
    colmap::image_t image_id1, colmap::image_t image_id2,
    const GroupMatches &matches) {
  colmap::FeatureMatches colmap_matches;
  for (const auto &match : matches) {
    colmap_matches.emplace_back(match.group2D_idx1, match.group2D_idx2);
  }
  THROW_CHECK_EQ(G_group.NumMatchesBetweenImages(image_id1, image_id2), 0);
  colmap::TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = std::move(colmap_matches);
  G_group.AddTwoViewGeometry(image_id1, image_id2,
                             std::move(two_view_geometry));
}

colmap::CorrespondenceGraph &StructureCorrespondenceGraph::LineGraph() {
  return G_line;
}

const colmap::CorrespondenceGraph &
StructureCorrespondenceGraph::LineGraph() const {
  return G_line;
}

colmap::CorrespondenceGraph &StructureCorrespondenceGraph::GroupGraph() {
  return G_group;
}

const colmap::CorrespondenceGraph &
StructureCorrespondenceGraph::GroupGraph() const {
  return G_group;
}

} // namespace limap
