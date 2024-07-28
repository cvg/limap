#include "vplib/base_vp_detector.h"

#include "base/graph.h"
#include "base/infinite_line.h"

#include <third-party/progressbar.hpp>

namespace limap {

namespace vplib {

py::dict BaseVPDetectorConfig::as_dict() const {
  py::dict output;
  output["min_length"] = min_length;
  output["inlier_threshold"] = inlier_threshold;
  output["min_num_supports"] = min_num_supports;
  output["th_perp_supports"] = th_perp_supports;
  return output;
}

std::map<int, VPResult> BaseVPDetector::AssociateVPsParallel(
    const std::map<int, std::vector<Line2d>> &all_lines) const {
  std::vector<int> image_ids;
  for (std::map<int, std::vector<Line2d>>::const_iterator it =
           all_lines.begin();
       it != all_lines.end(); ++it) {
    image_ids.push_back(it->first);
  }

  std::map<int, VPResult> vpresults;
  progressbar bar(image_ids.size());
#pragma omp parallel for
  for (const int &img_id : image_ids) {
    bar.update();
    vpresults.insert(
        std::make_pair(img_id, AssociateVPs(all_lines.at(img_id))));
  }
  return vpresults;
}

int BaseVPDetector::count_valid_supports_2d(
    const std::vector<Line2d> &lines) const {
  // count 2d supports that do that lie on the same infinite line
  size_t n_lines = lines.size();
  std::vector<int> parent_nodes(n_lines, -1);
  for (size_t i = 0; i < n_lines - 1; ++i) {
    size_t root_i = union_find_get_root(i, parent_nodes);
    for (size_t j = i + 1; j < n_lines; ++j) {
      size_t root_j = union_find_get_root(j, parent_nodes);
      if (root_j == root_i)
        continue;
      // test connection: project the shorter line to the longer one
      int k1 = i;
      int k2 = j;
      if (lines[i].length() > lines[j].length()) {
        k1 = j;
        k2 = i;
      }
      double ds = InfiniteLine2d(lines[k2]).point_distance(lines[k1].start);
      double de = InfiniteLine2d(lines[k2]).point_distance(lines[k1].end);
      double dist = std::max(ds, de);
      if (dist > config_.th_perp_supports)
        continue;
      parent_nodes[root_j] = root_i;
    }
  }
  int n_supports = 0;
  for (size_t i = 0; i < n_lines; ++i) {
    if (parent_nodes[i] == -1)
      n_supports++;
  }
  return n_supports;
}

} // namespace vplib

} // namespace limap
