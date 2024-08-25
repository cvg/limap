#include "merging/merging.h"
#include "merging/aggregator.h"

#include "base/graph.h"
#include "base/line_dists.h"
#include "base/linebase.h"
#include "util/types.h"

#include <algorithm>
#include <colmap/util/logging.h>
#include <set>
#include <third-party/progressbar.hpp>

namespace limap {

namespace merging {

std::vector<int>
ComputeLineTrackLabelsGreedy(const Graph &graph,
                             const std::vector<Line3d> &line3d_list_nodes) {
  const size_t n_nodes = graph.nodes.size();
  std::vector<edge_tuple> edges;
  for (Edge *edge : graph.undirected_edges) {
    edges.push_back(
        std::make_tuple(edge->sim, edge->node_idx1, edge->node_idx2));
  }
  STDLOG(INFO) << "# graph nodes:"
               << " " << n_nodes << std::endl;
  STDLOG(INFO) << "# graph edges:"
               << " " << edges.size() * 2 << std::endl;

  // Build the MSF.
  std::sort(edges.begin(), edges.end());
  std::reverse(edges.begin(), edges.end());

  std::vector<int> parent_nodes(n_nodes, -1);
  std::vector<std::set<int>> images_in_track(n_nodes);
  std::vector<std::set<int>> nodes_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_idx);
    nodes_in_track[node_idx].insert(node_idx);
  }

  size_t n_edges = edges.size();
  for (size_t edge_id = 0; edge_id < n_edges; ++edge_id) {
    const auto &e = edges[edge_id];
    size_t node_idx1 = std::get<1>(e);
    size_t node_idx2 = std::get<2>(e);

    size_t root1 = union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      // Union-find merging heuristic.
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        // update images_in_track for root2
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
        // update nodes_in_track for root2
        nodes_in_track[root2].insert(nodes_in_track[root1].begin(),
                                     nodes_in_track[root1].end());
        nodes_in_track[root1].clear();
      } else {
        parent_nodes[root2] = root1;
        // update images_in_track for root1
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
        // update nodes_in_track for root1
        nodes_in_track[root1].insert(nodes_in_track[root2].begin(),
                                     nodes_in_track[root2].end());
        nodes_in_track[root2].clear();
      }
    }
  }

  // Compute the tracks.
  std::vector<int> track_labels(n_nodes, -1);

  // only save tracks with at least two nodes
  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    size_t parent_idx = parent_nodes[node_idx];
    if (parent_nodes[parent_idx] == -1 && track_labels[parent_idx] == -1) {
      track_labels[parent_idx] = n_tracks++;
    }
  }
  STDLOG(INFO) << "# tracks:"
               << " " << n_tracks << std::endl;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    track_labels[node_idx] =
        track_labels[union_find_get_root(node_idx, parent_nodes)];
  }
  return track_labels;
}

std::vector<int>
ComputeLineTrackLabelsExhaustive(const Graph &graph,
                                 const std::vector<Line3d> &line3d_list_nodes,
                                 LineLinker3d linker3d) {
  // set the mode to avgtest merging
  linker3d.config.set_to_avgtest_merging();
  const size_t n_nodes = graph.nodes.size();
  std::vector<edge_tuple> edges;
  for (Edge *edge : graph.undirected_edges) {
    edges.push_back(
        std::make_tuple(edge->sim, edge->node_idx1, edge->node_idx2));
  }
  STDLOG(INFO) << "# graph nodes:"
               << " " << n_nodes << std::endl;
  STDLOG(INFO) << "# graph edges:"
               << " " << edges.size() * 2 << std::endl;

  // Build the MSF.
  std::sort(edges.begin(), edges.end());
  std::reverse(edges.begin(), edges.end());

  std::vector<int> parent_nodes(n_nodes, -1);
  std::vector<std::set<int>> images_in_track(n_nodes);
  std::vector<std::set<int>> nodes_in_track(n_nodes);
  std::vector<std::vector<Line3d>> lines_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_idx);
    nodes_in_track[node_idx].insert(node_idx);
    lines_in_track[node_idx].push_back(line3d_list_nodes[node_idx]);
  }

  size_t n_edges = edges.size();
  progressbar bar(n_edges);
  for (size_t edge_id = 0; edge_id < n_edges; ++edge_id) {
    bar.update();
    const auto &e = edges[edge_id];
    size_t node_idx1 = std::get<1>(e);
    size_t node_idx2 = std::get<2>(e);

    size_t root1 = union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      // test the similarity between the two unions
      bool flag = true;
      for (auto it1 = lines_in_track[root1].begin();
           it1 != lines_in_track[root1].end(); ++it1) {
        for (auto it2 = lines_in_track[root2].begin();
             it2 != lines_in_track[root2].end(); ++it2) {
          if (compute_overlap<Line3d>(*it1, *it2) <= 0)
            continue;
          if (!linker3d.check_connection(*it1, *it2)) {
            flag = false;
            break;
          }
        }
        if (!flag)
          break;
      }
      if (!flag)
        continue;

      // Union-find merging heuristic.
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        // update images_in_track for root2
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
        // update nodes_in_track for root2
        nodes_in_track[root2].insert(nodes_in_track[root1].begin(),
                                     nodes_in_track[root1].end());
        nodes_in_track[root1].clear();
        // update lines_in_track for root2
        lines_in_track[root2].insert(lines_in_track[root2].end(),
                                     lines_in_track[root1].begin(),
                                     lines_in_track[root1].end());
        lines_in_track[root1].clear();
      } else {
        parent_nodes[root2] = root1;
        // update images_in_track for root1
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
        // update nodes_in_track for root1
        nodes_in_track[root1].insert(nodes_in_track[root2].begin(),
                                     nodes_in_track[root2].end());
        nodes_in_track[root2].clear();
        // update lines_in_track for root1
        lines_in_track[root1].insert(lines_in_track[root1].end(),
                                     lines_in_track[root2].begin(),
                                     lines_in_track[root2].end());
        lines_in_track[root2].clear();
      }
    }
  }

  // Compute the tracks.
  std::vector<int> track_labels(n_nodes, -1);

  // only save tracks with at least two nodes
  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    size_t parent_idx = parent_nodes[node_idx];
    if (parent_nodes[parent_idx] == -1 && track_labels[parent_idx] == -1) {
      track_labels[parent_idx] = n_tracks++;
    }
  }
  STDLOG(INFO) << "# tracks:"
               << " " << n_tracks << std::endl;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    track_labels[node_idx] =
        track_labels[union_find_get_root(node_idx, parent_nodes)];
  }
  return track_labels;
}

std::vector<int>
ComputeLineTrackLabelsAvg(const Graph &graph,
                          const std::vector<Line3d> &line3d_list_nodes,
                          LineLinker3d linker3d) {
  // set the mode to avgtest merging
  linker3d.config.set_to_avgtest_merging();
  const size_t n_nodes = graph.nodes.size();
  std::vector<edge_tuple> edges;
  for (Edge *edge : graph.undirected_edges) {
    edges.push_back(
        std::make_tuple(edge->sim, edge->node_idx1, edge->node_idx2));
  }
  STDLOG(INFO) << "# graph nodes:"
               << " " << n_nodes << std::endl;
  STDLOG(INFO) << "# graph edges:"
               << " " << edges.size() * 2 << std::endl;

  // Build the MSF.
  std::sort(edges.begin(), edges.end());
  std::reverse(edges.begin(), edges.end());

  std::vector<int> parent_nodes(n_nodes, -1);
  std::vector<std::set<int>> images_in_track(n_nodes);
  std::vector<std::set<int>> nodes_in_track(n_nodes);
  std::vector<std::pair<Line3d, int>> avgline_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_idx);
    nodes_in_track[node_idx].insert(node_idx);
    avgline_in_track[node_idx] = std::make_pair(line3d_list_nodes[node_idx], 1);
  }

  size_t num_edges = edges.size();
  for (size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
    const auto &e = edges[edge_id];
    size_t node_idx1 = std::get<1>(e);
    size_t node_idx2 = std::get<2>(e);

    size_t root1 = union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      // test the similarity between the two unions
      if (!linker3d.check_connection(avgline_in_track[root1].first,
                                     avgline_in_track[root2].first))
        continue;

      // Union-find merging heuristic.
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        // update images_in_track for root2
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
        // update nodes_in_track for root2
        nodes_in_track[root2].insert(nodes_in_track[root1].begin(),
                                     nodes_in_track[root1].end());
        nodes_in_track[root1].clear();
        // update avgline_in_track for root2
        auto d1 = avgline_in_track[root2];
        auto d2 = avgline_in_track[root1];
        Line3d newline;
        newline.start =
            (d1.first.start * d1.second + d2.first.start * d2.second) /
            (d1.second + d2.second);
        newline.end = (d1.first.end * d1.second + d2.first.end * d2.second) /
                      (d1.second + d2.second);
        avgline_in_track[root2] =
            std::make_pair(newline, d1.second + d2.second);
      } else {
        parent_nodes[root2] = root1;
        // update images_in_track for root1
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
        // update nodes_in_track for root1
        nodes_in_track[root1].insert(nodes_in_track[root2].begin(),
                                     nodes_in_track[root2].end());
        nodes_in_track[root2].clear();
        // update avgline_in_track for root1
        auto d1 = avgline_in_track[root1];
        auto d2 = avgline_in_track[root2];
        Line3d newline;
        newline.start =
            (d1.first.start * d1.second + d2.first.start * d2.second) /
            (d1.second + d2.second);
        newline.end = (d1.first.end * d1.second + d2.first.end * d2.second) /
                      (d1.second + d2.second);
        avgline_in_track[root1] =
            std::make_pair(newline, d1.second + d2.second);
      }
    }
  }

  // Compute the tracks.
  std::vector<int> track_labels(n_nodes, -1);

  // only save tracks with at least two nodes
  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    size_t parent_idx = parent_nodes[node_idx];
    if (parent_nodes[parent_idx] == -1 && track_labels[parent_idx] == -1) {
      track_labels[parent_idx] = n_tracks++;
    }
  }
  STDLOG(INFO) << "# tracks:"
               << " " << n_tracks << std::endl;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1)
      continue;
    track_labels[node_idx] =
        track_labels[union_find_get_root(node_idx, parent_nodes)];
  }
  return track_labels;
}

void MergeToLineTracks(Graph &graph, std::vector<LineTrack> &linetracks,
                       const std::map<int, std::vector<Line2d>> &all_lines_2d,
                       const ImageCollection &imagecols,
                       const std::map<int, std::vector<Line3d>> &all_lines_3d,
                       const std::map<int, std::vector<int>> &neighbors,
                       LineLinker linker) {
  // set the mode to spatial merging
  linker.linker_3d.config.set_to_spatial_merging();
  THROW_CHECK_EQ(all_lines_2d.size(), all_lines_3d.size());
  THROW_CHECK_EQ(all_lines_2d.size(), neighbors.size());

  // get image ids
  std::vector<int> image_ids;
  for (auto it = all_lines_2d.begin(); it != all_lines_2d.end(); ++it) {
    image_ids.push_back(it->first);
  }

  // insert nodes
  std::vector<Line3d> line3d_list_nodes;
  std::map<int, std::vector<double>> all_lengths_3d;
  for (const int &image_id : image_ids) {
    all_lengths_3d.insert(std::make_pair(image_id, std::vector<double>()));
  }
  for (const int &image_id : image_ids) {
    THROW_CHECK_EQ(all_lines_2d.at(image_id).size(),
                   all_lines_3d.at(image_id).size());
    size_t n_lines = all_lines_2d.at(image_id).size();
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      const Line3d &line = all_lines_3d.at(image_id)[line_id];
      all_lengths_3d[image_id].push_back(line.length());
      if (line.length() == 0)
        continue;
      graph.FindOrCreateNode(image_id, line_id);
      line3d_list_nodes.push_back(line);
    }
  }

  // compute similarity and collect potential edges
  typedef std::pair<Node2d, Node2d> Node2dPair;
  std::map<int, std::vector<Node2dPair>> pairs;
  for (const int &image_id : image_ids) {
    pairs.insert(std::make_pair(image_id, std::vector<Node2dPair>()));
  }

  // self similarity
#pragma omp parallel for
  for (const int &image_id : image_ids) {
    THROW_CHECK_EQ(all_lines_2d.at(image_id).size(),
                   all_lengths_3d[image_id].size());
    size_t n_lines = all_lines_2d.at(image_id).size();
    for (size_t i = 0; i < n_lines; ++i) {
      if (all_lengths_3d[image_id][i] == 0)
        continue;
      const Line3d &l1 = all_lines_3d.at(image_id)[i];
      for (size_t j = i + 1; j < n_lines; ++j) {
        if (all_lengths_3d[image_id][j] == 0)
          continue;
        const Line3d &l2 = all_lines_3d.at(image_id)[j];
        // check 3d
        if (!linker.check_connection_3d(l1, l2))
          continue;
        // check 2d
        if (!linker.check_connection_2d(all_lines_2d.at(image_id)[i],
                                        all_lines_2d.at(image_id)[j]))
          continue;
        pairs[image_id].push_back(std::make_pair(std::make_pair(image_id, i),
                                                 std::make_pair(image_id, j)));
      }
    }
  }
  // cross-image similarity
  progressbar bar(image_ids.size());
#pragma omp parallel for
  for (const int &image_id : image_ids) {
    bar.update();
    size_t n_neighbors = neighbors.at(image_id).size();
    THROW_CHECK_EQ(all_lines_2d.at(image_id).size(),
                   all_lengths_3d[image_id].size());
    size_t n_lines = all_lines_2d.at(image_id).size();

    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      // get l1
      if (all_lengths_3d[image_id][line_id] == 0)
        continue;
      const Line3d &l1 = all_lines_3d.at(image_id)[line_id];
      for (size_t ng_id = 0; ng_id < n_neighbors; ++ng_id) {
        size_t ng_image_id = neighbors.at(image_id)[ng_id];
        size_t ng_n_lines = all_lines_2d.at(ng_image_id).size();
        for (size_t ng_line_id = 0; ng_line_id < ng_n_lines; ++ng_line_id) {
          // for openmp speed up
          int key = image_id + line_id + ng_image_id + ng_line_id;
          if (key % 2 == 0 && image_id < ng_image_id)
            continue;
          if (key % 2 == 1 && image_id > ng_image_id)
            continue;
          // get l2
          if (all_lengths_3d[ng_image_id][ng_line_id] == 0)
            continue;
          const Line3d &l2 = all_lines_3d.at(ng_image_id)[ng_line_id];
          // check 3d
          if (!linker.check_connection_3d(l1, l2))
            continue;
          // check 2d
          if (!linker.check_connection_2d(
                  l1.projection(imagecols.camview(ng_image_id)),
                  all_lines_2d.at(ng_image_id)[ng_line_id]))
            continue;
          if (!linker.check_connection_2d(
                  l2.projection(imagecols.camview(image_id)),
                  all_lines_2d.at(image_id)[line_id]))
            continue;
          pairs[image_id].push_back(
              std::make_pair(std::make_pair(image_id, line_id),
                             std::make_pair(ng_image_id, ng_line_id)));
        }
      }
    }
  }

  // insert edges
  for (const int &image_id : image_ids) {
    for (auto it = pairs[image_id].begin(); it != pairs[image_id].end(); ++it) {
      PatchNode *node1 =
          graph.FindOrCreateNode(it->first.first, it->first.second);
      PatchNode *node2 =
          graph.FindOrCreateNode(it->second.first, it->second.second);
      double score = all_lengths_3d[node1->image_idx][node1->line_idx] +
                     all_lengths_3d[node2->image_idx][node2->line_idx];
      graph.AddEdge(node1, node2, score);
    }
  }

  // compute tracks
  int n_nodes = graph.nodes.size();
  std::vector<int> track_labels =
      ComputeLineTrackLabelsGreedy(graph, line3d_list_nodes);

  int n_tracks =
      *std::max_element(track_labels.begin(), track_labels.end()) + 1;
  linetracks.resize(n_tracks);

  // set all lines into tracks
  for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
    PatchNode *node = graph.nodes[node_id];
    Line2d line2d = all_lines_2d.at(node->image_idx)[node->line_idx];
    Line3d line3d = all_lines_3d.at(node->image_idx)[node->line_idx];
    double score = line3d.length();

    size_t track_id = track_labels[node_id];
    if (track_id == -1)
      continue;
    linetracks[track_id].node_id_list.push_back(node_id);
    linetracks[track_id].image_id_list.push_back(node->image_idx);
    linetracks[track_id].line_id_list.push_back(node->line_idx);
    linetracks[track_id].line2d_list.push_back(line2d);
    linetracks[track_id].line3d_list.push_back(line3d);
    linetracks[track_id].score_list.push_back(score);
  }

  // aggregate 3d lines to get final line proposal
  for (auto it = linetracks.begin(); it != linetracks.end(); ++it) {
    it->line =
        Aggregator::aggregate_line3d_list(it->line3d_list, it->score_list, 0);
  }
}

std::vector<LineTrack>
RemergeLineTracks(const std::vector<LineTrack> &linetracks,
                  LineLinker3d linker3d, const int num_outliers) {
  linker3d.config.set_to_spatial_merging();
  size_t n_tracks = linetracks.size();

  // compute edges to remerge
  std::set<std::pair<size_t, size_t>> edges;
  std::vector<std::set<std::pair<size_t, size_t>>> edges_per_track(n_tracks);
  std::vector<int> active_ids;
  for (size_t i = 0; i < n_tracks; ++i) {
    if (linetracks[i].active)
      active_ids.push_back(i);
  }
  int n_active_ids = active_ids.size();
  progressbar bar(n_active_ids, n_active_ids >= 10000);
#pragma omp parallel for
  for (size_t k = 0; k < n_active_ids; ++k) {
    int i = active_ids[k];
    bar.update();
    const Line3d &l1 = linetracks[i].line;
    for (size_t j = 0; j < n_tracks; ++j) {
      if (i == j)
        continue;
      if (n_active_ids == n_tracks) {
        if (i < j && (i + j) % 2 == 0)
          continue;
        if (i > j && (i + j) % 2 == 1)
          continue;
      }
      const Line3d &l2 = linetracks[j].line;
      bool valid = linker3d.check_connection(l1, l2);
      if (!valid)
        continue;
      if (i < j)
        edges_per_track[i].insert(std::make_pair(i, j));
      else
        edges_per_track[i].insert(std::make_pair(j, i));
    }
  }
  for (size_t i = 0; i < n_tracks; ++i) {
    edges.insert(edges_per_track[i].begin(), edges_per_track[i].end());
  }

  // group connected components
  std::vector<int> parent_tracks(n_tracks, -1);
  std::vector<std::set<int>> tracks_in_group(n_tracks);
  for (size_t i = 0; i < n_tracks; ++i) {
    tracks_in_group[i].insert(i);
  }
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    size_t track_id1 = it->first;
    size_t track_id2 = it->second;

    size_t root1 = union_find_get_root(track_id1, parent_tracks);
    size_t root2 = union_find_get_root(track_id2, parent_tracks);
    if (root1 != root2) {
      // Union-find merging heuristic.
      if (tracks_in_group[root1].size() < tracks_in_group[root2].size()) {
        parent_tracks[root1] = root2;
        // update tracks_in_group for root2
        tracks_in_group[root2].insert(tracks_in_group[root1].begin(),
                                      tracks_in_group[root1].end());
        tracks_in_group[root1].clear();
      } else {
        parent_tracks[root2] = root1;
        // update tracks_in_group for root1
        tracks_in_group[root1].insert(tracks_in_group[root2].begin(),
                                      tracks_in_group[root2].end());
        tracks_in_group[root2].clear();
      }
    }
  }

  // compute groups
  std::vector<size_t> group_labels(n_tracks, -1);
  size_t n_groups = 0;
  for (size_t track_idx = 0; track_idx < n_tracks; ++track_idx) {
    if (parent_tracks[track_idx] == -1) {
      group_labels[track_idx] = n_groups++;
    }
  }
  // STDLOG(INFO) << "# groups after remerging:" << " " << n_groups <<
  // std::endl;
  for (size_t track_idx = 0; track_idx < n_tracks; ++track_idx) {
    if (group_labels[track_idx] != -1) {
      continue;
    }
    group_labels[track_idx] =
        group_labels[union_find_get_root(track_idx, parent_tracks)];
  }

  // recompute track information for each group
  std::vector<LineTrack> new_linetracks(n_groups);
  std::vector<int> counter_groups(n_groups, 0);
  for (size_t track_id = 0; track_id < n_tracks; ++track_id) {
    const LineTrack &track = linetracks[track_id];
    size_t group_id = group_labels[track_id];
    counter_groups[group_id]++;

    new_linetracks[group_id].node_id_list.insert(
        new_linetracks[group_id].node_id_list.end(), track.node_id_list.begin(),
        track.node_id_list.end());
    new_linetracks[group_id].image_id_list.insert(
        new_linetracks[group_id].image_id_list.end(),
        track.image_id_list.begin(), track.image_id_list.end());
    new_linetracks[group_id].line_id_list.insert(
        new_linetracks[group_id].line_id_list.end(), track.line_id_list.begin(),
        track.line_id_list.end());
    new_linetracks[group_id].line2d_list.insert(
        new_linetracks[group_id].line2d_list.end(), track.line2d_list.begin(),
        track.line2d_list.end());
    new_linetracks[group_id].line3d_list.insert(
        new_linetracks[group_id].line3d_list.end(), track.line3d_list.begin(),
        track.line3d_list.end());
    new_linetracks[group_id].score_list.insert(
        new_linetracks[group_id].score_list.end(), track.score_list.begin(),
        track.score_list.end());
  }

  // aggregate 3d lines to get final line proposal
  for (size_t group_id = 0; group_id < n_groups; ++group_id) {
    new_linetracks[group_id].line = Aggregator::aggregate_line3d_list(
        new_linetracks[group_id].line3d_list,
        new_linetracks[group_id].score_list, num_outliers);
    if (counter_groups[group_id] == 1) {
      new_linetracks[group_id].active = false;
    }
  }
  return new_linetracks;
}

} // namespace merging

} // namespace limap
