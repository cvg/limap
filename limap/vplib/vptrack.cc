#include "vplib/vptrack.h"
#include "base/graph.h"

namespace limap {

namespace vplib {

py::dict VPTrack::as_dict() const {
  py::dict output;
  output["direction"] = direction;
  output["supports"] = supports;
  return output;
}

VPTrack::VPTrack(py::dict dict){
    ASSIGN_PYDICT_ITEM(dict, direction, V3D)
        ASSIGN_PYDICT_ITEM(dict, supports, std::vector<Node2d>)}

std::vector<VPTrack> MergeVPTracksByDirection(
    const std::vector<VPTrack> &tracks, const double th_angle_merge) {
  std::vector<int> parent_nodes(tracks.size(), -1);
  std::vector<std::set<int>> images_in_track(tracks.size());
  for (size_t i = 0; i < tracks.size(); ++i) {
    for (auto it = tracks[i].supports.begin(); it != tracks[i].supports.end();
         ++it) {
      images_in_track[i].insert(it->first);
    }
  }
  for (size_t i = 0; i < tracks.size() - 1; ++i) {
    int root_i = union_find_get_root(i, parent_nodes);
    for (size_t j = i + 1; j < tracks.size(); ++j) {
      int root_j = union_find_get_root(j, parent_nodes);
      if (root_i == root_j)
        continue;

      // test angle
      double cosine = std::abs(tracks[i].direction.dot(tracks[j].direction));
      if (cosine > 1.0)
        cosine = 1.0;
      double angle = acos(cosine) * 180.0 / M_PI;
      if (angle > th_angle_merge)
        continue;

      // test image id intersection
      std::set<int> intersection;
      std::set_intersection(
          images_in_track[root_i].begin(), images_in_track[root_i].end(),
          images_in_track[root_j].begin(), images_in_track[root_j].end(),
          std::inserter(intersection, intersection.begin()));
      ;
      if (intersection.size() != 0)
        continue;

      // link two nodes
      if (images_in_track[root_i].size() < images_in_track[root_j].size()) {
        parent_nodes[root_i] = root_j;
        images_in_track[root_j].insert(images_in_track[root_i].begin(),
                                       images_in_track[root_i].end());
        images_in_track[root_i].clear();
      } else {
        parent_nodes[root_j] = root_i;
        images_in_track[root_i].insert(images_in_track[root_j].begin(),
                                       images_in_track[root_j].end());
        images_in_track[root_j].clear();
      }
    }
  }

  std::vector<int> labels(tracks.size(), -1);
  int n_tracks = 0;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (parent_nodes[i] == -1)
      labels[i] = n_tracks++;
  }
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (labels[i] != -1)
      continue;
    labels[i] = labels[union_find_get_root(i, parent_nodes)];
  }
  std::vector<VPTrack> vptracks(n_tracks);
  for (size_t i = 0; i < tracks.size(); ++i) {
    int label = labels[i];
    if (vptracks[label].supports.empty()) {
      vptracks[label].direction = tracks[i].direction;
    } else {
      V3D direction = V3D::Zero();
      direction += vptracks[label].direction * vptracks[label].length();
      direction += tracks[i].direction * tracks[i].length();
      direction /= (vptracks[label].length() + tracks[i].length());
      vptracks[label].direction = direction;
    }
    std::copy(tracks[i].supports.begin(), tracks[i].supports.end(),
              std::back_inserter(vptracks[label].supports));
  }
  for (size_t label = 0; label < n_tracks; ++label) {
    vptracks[label].direction = vptracks[label].direction.normalized();
  }
  std::sort(
      vptracks.begin(), vptracks.end(),
      [](const auto &d1, const auto &d2) { return d1.length() > d2.length(); });
  return vptracks;
}

} // namespace vplib

} // namespace limap
