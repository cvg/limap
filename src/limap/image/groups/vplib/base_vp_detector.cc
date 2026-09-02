#include "limap/image/groups/vplib/base_vp_detector.h"

#include "limap/geometry/inf_line2d.h"

#include <colmap/math/union_find.h>
#include <thirdparty/progressbar.hpp>

namespace limap {
namespace image {
namespace groups {

namespace vplib {

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
  int n_jobs = base_options_.n_jobs;
  if (n_jobs == -1) {
    n_jobs = omp_get_max_threads();
  }

  if (n_jobs == 1) {
    for (const int &img_id : image_ids) {
      bar.update();
      vpresults.insert(
          std::make_pair(img_id, AssociateVPs(all_lines.at(img_id))));
    }
  } else {
#pragma omp parallel for num_threads(n_jobs) schedule(dynamic)
    for (int i = 0; i < (int)image_ids.size(); ++i) {
      int img_id = image_ids[i];
      auto result = AssociateVPs(all_lines.at(img_id));

// merge results safely
#pragma omp critical
      {
        vpresults.insert(std::make_pair(img_id, std::move(result)));
        bar.update(); // also needs critical section
      }
    }
  }
  return vpresults;
}

int BaseVPDetector::CountValidSupports2D(const std::vector<Line2d> &lines,
                                         const double th_perp_supports) const {
  const size_t n_lines = lines.size();

  // Initialize Union-Find structure
  colmap::UnionFind<size_t> uf;
  uf.Reserve(n_lines);
  for (size_t i = 0; i < n_lines; ++i) {
    uf.Find(i); // Ensure each element is initialized
  }

  // Connect lines that lie on the same infinite 2D line
  for (size_t i = 0; i < n_lines - 1; ++i) {
    const size_t root_i = uf.Find(i);

    for (size_t j = i + 1; j < n_lines; ++j) {
      const size_t root_j = uf.Find(j);

      if (root_i == root_j) {
        continue;
      }

      // project the shorter line to the longer one
      int k1 = i;
      int k2 = j;

      if (lines[i].Length() > lines[j].Length()) {
        k1 = j;
        k2 = i;
      }

      double ds = InfiniteLine2d(lines[k2]).PointDistance(lines[k1].start);
      double de = InfiniteLine2d(lines[k2]).PointDistance(lines[k1].end);
      double dist = std::max(ds, de);

      if (dist > th_perp_supports) {
        continue;
      }

      // Merge sets
      uf.Union(root_i, root_j);
    }
  }

  // Count distinct root components
  int n_supports = 0;
  for (size_t i = 0; i < n_lines; ++i) {
    if (uf.Find(i) == i) { // i is its own root
      n_supports++;
    }
  }

  return n_supports;
}

} // namespace vplib

} // namespace groups
} // namespace image
} // namespace limap
