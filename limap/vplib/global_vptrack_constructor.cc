#include "vplib/global_vptrack_constructor.h"
#include "base/graph.h"

namespace limap {

namespace vplib {

std::vector<VPTrack> GlobalVPTrackConstructor::ClusterLineTracks(
    const std::vector<LineTrack> &linetracks,
    const ImageCollection &imagecols) const {
  // count edges
  std::map<std::pair<Node2d, Node2d>, int> edge_counters;
  for (size_t track_id = 0; track_id < linetracks.size(); ++track_id) {
    const auto &track = linetracks[track_id];
    std::set<std::pair<Node2d, Node2d>> edges; // edge for the current track

    for (size_t i = 0; i < track.count_lines() - 1; ++i) {
      int img_id1 = track.image_id_list[i];
      int line_id1 = track.line_id_list[i];
      if (!vpresults_.at(img_id1).HasVP(line_id1))
        continue;
      int vp_id1 = vpresults_.at(img_id1).GetVPLabel(line_id1);
      Node2d node1 = std::make_pair(img_id1, vp_id1);
      V3D vp1 = vpresults_.at(img_id1).GetVP(line_id1);
      V3D direc1 = imagecols.camview(img_id1).get_direction_from_vp(vp1);

      for (size_t j = i + 1; j < track.count_lines(); ++j) {
        int img_id2 = track.image_id_list[j];
        int line_id2 = track.line_id_list[j];
        if (img_id1 == img_id2) // on the same image
          continue;
        if (!vpresults_.at(img_id2).HasVP(line_id2))
          continue;

        // test if the edge is already added for the current track
        int vp_id2 = vpresults_.at(img_id2).GetVPLabel(line_id2);
        Node2d node2 = std::make_pair(img_id2, vp_id2);
        std::pair<Node2d, Node2d> edge;
        if (img_id1 < img_id2)
          edge = std::make_pair(node1, node2);
        else
          edge = std::make_pair(node2, node1);
        if (edges.find(edge) != edges.end())
          continue;

        // verify edge with poses
        V3D vp2 = vpresults_.at(img_id2).GetVP(line_id2);
        V3D direc2 = imagecols.camview(img_id2).get_direction_from_vp(vp2);
        double cosine = std::abs(direc1.dot(direc2));
        if (cosine > 1.0)
          cosine = 1.0;
        double angle = acos(cosine) * 180.0 / M_PI;
        if (angle > config_.th_angle_verify)
          continue;

        // add the edge to the edge set for the current track
        edges.insert(edge);
      }
    }

    // update edge counters
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      if (edge_counters.find(*it) == edge_counters.end())
        edge_counters.insert(std::make_pair(*it, 0));
      edge_counters.at(*it) += 1;
    }
  }

  // build graph
  Graph graph;
  for (auto it = edge_counters.begin(); it != edge_counters.end(); ++it) {
    if (it->second < config_.min_common_lines)
      continue;
    auto node = it->first;
    int img_id1 = node.first.first;
    int vp_id1 = node.first.second;
    int img_id2 = node.second.first;
    int vp_id2 = node.second.second;

    PatchNode *node1 = graph.FindOrCreateNode(img_id1, vp_id1);
    PatchNode *node2 = graph.FindOrCreateNode(img_id2, vp_id2);
    graph.AddEdge(node1, node2, it->second);
  }

  // track construction
  std::vector<size_t> track_labels = ComputeTrackLabels(graph); // MSF
  const size_t n_nodes = graph.nodes.size();
  std::map<int, VPTrack> m_vptrack;
  for (size_t i = 0; i < n_nodes; ++i) {
    int label = track_labels[i];
    if (m_vptrack.find(label) == m_vptrack.end())
      m_vptrack.insert(std::make_pair(label, VPTrack()));

    int img_id = graph.nodes[i]->image_idx;
    int vp_id = graph.nodes[i]->line_idx;
    m_vptrack.at(label).supports.push_back(std::make_pair(img_id, vp_id));
  }
  std::vector<VPTrack> vptracks;
  for (auto it = m_vptrack.begin(); it != m_vptrack.end(); ++it) {
    if (it->second.length() < config_.min_track_length)
      continue;
    VPTrack track = it->second;
    // compute average direction
    V3D avg_direction = V3D::Zero();
    for (size_t i = 0; i < track.length(); ++i) {
      int img_id = track.supports[i].first;
      int vp_id = track.supports[i].second;
      V3D vp = vpresults_.at(img_id).GetVPbyCluster(vp_id);
      V3D direction = imagecols.camview(img_id).get_direction_from_vp(vp);
      avg_direction += direction;
    }
    V3D direction = avg_direction /= double(track.length());
    track.direction = direction.normalized();
    vptracks.push_back(track);
  }
  std::sort(
      vptracks.begin(), vptracks.end(),
      [](const auto &d1, const auto &d2) { return d1.length() > d2.length(); });
  return vptracks;
}

} // namespace vplib

} // namespace limap
