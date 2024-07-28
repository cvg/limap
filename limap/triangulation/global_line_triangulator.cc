#include "triangulation/global_line_triangulator.h"
#include "merging/aggregator.h"
#include "merging/merging.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <third-party/progressbar.hpp>

namespace limap {

namespace triangulation {

void GlobalLineTriangulator::Init(
    const std::map<int, std::vector<Line2d>> &all_2d_segs,
    const ImageCollection &imagecols) {
  BaseLineTriangulator::Init(all_2d_segs, imagecols);

  // initialize additional empty containers
  for (const int &img_id : imagecols.get_img_ids()) {
    size_t n_lines = all_2d_segs.at(img_id).size();
    valid_edges_.insert(
        std::make_pair(img_id, std::vector<std::vector<NeighborLineNode>>()));
    valid_edges_.at(img_id).resize(n_lines);
    valid_tris_.insert(
        std::make_pair(img_id, std::vector<std::vector<TriTuple>>()));
    valid_tris_.at(img_id).resize(n_lines);
    tris_best_.insert(std::make_pair(img_id, std::vector<TriTuple>()));
    tris_best_.at(img_id).resize(n_lines);
  }

  // flags for monitoring the scoring process
  for (const int &img_id : imagecols.get_img_ids()) {
    size_t n_lines = all_2d_segs.at(img_id).size();
    already_scored_.insert(std::make_pair(img_id, std::vector<bool>()));
    already_scored_.at(img_id).resize(n_lines);
    std::fill(already_scored_[img_id].begin(), already_scored_[img_id].end(),
              false);
  }
}

void GlobalLineTriangulator::ScoringCallback(const int img_id) {
  LineLinker linker_scoring = linker_;
  linker_scoring.linker_3d.config.set_to_shared_parent_scoring();
  for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
    scoreOneNode(img_id, line_id, linker_scoring);
  }
  if (!config_.debug_mode) {
    valid_tris_[img_id].clear();
    valid_tris_[img_id].resize(CountLines(img_id));
  }
}

void GlobalLineTriangulator::scoreOneNode(const int img_id, const int line_id,
                                          const LineLinker &linker) {
  if (already_scored_[img_id][line_id])
    return;
  auto &tris = tris_[img_id][line_id];
  size_t n_tris = tris.size();

  // score all the pairs
  std::vector<double> scores(n_tris, 0);
#pragma omp parallel for
  for (size_t i = 0; i < n_tris; ++i) {
    // each image contributes only once
    std::map<int, std::vector<double>> score_table;
    const Line3d &l1 = std::get<0>(tris[i]);
    int img_id = std::get<2>(tris[i]).first;
    int line_id = std::get<2>(tris[i]).second;
    const CameraView &view1 = imagecols_->camview(img_id);
    for (size_t j = 0; j < n_tris; ++j) {
      if (i == j)
        continue;
      const Line3d &l2 = std::get<0>(tris[j]);
      int ng_img_id = std::get<2>(tris[j]).first;
      int ng_line_id = std::get<2>(tris[j]).second;
      if (ng_img_id == img_id)
        continue;
      const CameraView &view2 = imagecols_->camview(ng_img_id);
      double score3d = linker.compute_score_3d(l1, l2);
      if (score3d == 0)
        continue;
      double score2d = linker.compute_score_2d(
          l1.projection(view2), all_lines_2d_[ng_img_id][ng_line_id]);
      if (score2d == 0)
        continue;
      double score = std::min(score3d, score2d);
      if (score_table.find(ng_img_id) == score_table.end())
        score_table.insert(std::make_pair(ng_img_id, std::vector<double>()));
      score_table[ng_img_id].push_back(score);
    }
    // one image contributes at most one support
    for (auto it = score_table.begin(); it != score_table.end(); ++it) {
      scores[i] += *std::max_element(it->second.begin(), it->second.end());
    }
  }
  for (size_t i = 0; i < n_tris; ++i) {
    std::get<1>(tris[i]) = scores[i];
  }

  // get valid tris and connections
  std::map<int, int> reverse_mapper;
  int n_neighbors = neighbors_[img_id].size();
  for (int i = 0; i < n_neighbors; ++i) {
    reverse_mapper.insert(std::make_pair(neighbors_[img_id][i], i));
  }
  std::vector<std::pair<double, int>> scores_to_sort;
  for (size_t tri_id = 0; tri_id < tris.size(); ++tri_id) {
    scores_to_sort.push_back(std::make_pair(std::get<1>(tris[tri_id]), tri_id));
  }
  std::sort(scores_to_sort.begin(), scores_to_sort.end(),
            std::greater<std::pair<double, int>>());
  int n_valid_conns =
      std::min(int(scores_to_sort.size()), config_.max_valid_conns);
  for (size_t i = 0; i < n_valid_conns; ++i) {
    int tri_id = scores_to_sort[i].second;
    auto &tri = tris[tri_id];
    double score = std::get<1>(tri);
    if (score < config_.fullscore_th)
      continue;
    valid_tris_[img_id][line_id].push_back(tri);
    auto &node = std::get<2>(tri);
    valid_edges_[img_id][line_id].push_back(
        std::make_pair(reverse_mapper.at(node.first), node.second));
  }

  // get best tris
  double max_score = -1;
  for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
    auto trituple = tris[tri_id];
    double score = std::get<1>(trituple);
    if (score > max_score) {
      tris_best_[img_id][line_id] = tris[tri_id];
      max_score = score;
    }
  }

  // clear intermediate triangulations
  if (!config_.debug_mode) {
    tris_[img_id][line_id].clear();
    valid_tris_[img_id][line_id].clear();
  }
  already_scored_[img_id][line_id] = true;
}

const TriTuple &GlobalLineTriangulator::getBestTri(const int img_id,
                                                   const int line_id) const {
  return tris_best_.at(img_id)[line_id];
}

void GlobalLineTriangulator::filterNodeByNumOuterEdges(
    const std::map<int, std::vector<std::vector<NeighborLineNode>>>
        &valid_edges,
    std::map<int, std::vector<bool>> &flags) {
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    flags.insert(std::make_pair(img_id, std::vector<bool>()));
    flags[img_id].resize(n_lines);
    std::fill(flags[img_id].begin(), flags[img_id].end(), true);
  }

  // build checktable with all edges pointing to the node
  std::map<int, std::vector<std::vector<LineNode>>> parent_neighbors;
  std::map<int, std::vector<int>> counters;
  for (const int &img_id : imagecols_->get_img_ids()) {
    parent_neighbors.insert(
        std::make_pair(img_id, std::vector<std::vector<LineNode>>()));
    counters.insert(std::make_pair(img_id, std::vector<int>()));
    size_t n_lines = CountLines(img_id);
    parent_neighbors[img_id].resize(n_lines);
    counters[img_id].resize(n_lines);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      counters[img_id][line_id] = valid_edges.at(img_id)[line_id].size();
    }
  }
  for (const int &img_id : imagecols_->get_img_ids()) {
    for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
      auto &nodes = valid_edges.at(img_id)[line_id];
      for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        int ng_img_id = neighbors_[img_id][it->first];
        int ng_line_id = it->second;
        parent_neighbors[ng_img_id][ng_line_id].push_back(
            std::make_pair(img_id, line_id));
      }
      if (counters[img_id][line_id] < config_.min_num_outer_edges) {
        flags[img_id][line_id] = false;
      }
    }
  }

  // iteratively filter node
  std::queue<LineNode> q;
  for (const int &img_id : imagecols_->get_img_ids()) {
    for (size_t line_id = 0; line_id < CountLines(img_id); line_id++) {
      if (flags[img_id][line_id])
        continue;
      q.push(std::make_pair(img_id, line_id));
    }
  }

  while (!q.empty()) {
    LineNode node = q.front();
    q.pop();
    auto &parents = parent_neighbors[node.first][node.second];
    for (auto it = parents.begin(); it != parents.end(); ++it) {
      if (!flags[it->first][it->second])
        continue;
      counters[it->first][it->second]--;
      if (counters[it->first][it->second] < config_.min_num_outer_edges) {
        flags[it->first][it->second] = false;
        q.push(std::make_pair(it->first, it->second));
      }
    }
  }
}

void GlobalLineTriangulator::run_clustering(Graph *graph) {
  std::cout << "Start building line graph for clustering..." << std::endl;
  LineLinker linker_clustering = linker_;
  linker_clustering.linker_3d.config.set_to_spatial_merging();

  // get valid flags
  filterNodeByNumOuterEdges(valid_edges_, valid_flags_);

  // collect undirected edges
  std::set<std::pair<LineNode, LineNode>> edges;
  for (const int &img_id : imagecols_->get_img_ids()) {
    for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
      auto &nodes = valid_edges_[img_id][line_id];
      for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        LineNode node1 = std::make_pair(img_id, line_id);
        if (!valid_flags_[node1.first][node1.second])
          continue;
        LineNode node2 =
            std::make_pair(neighbors_[img_id][it->first], it->second);
        if (!valid_flags_[node2.first][node2.second])
          continue;
        if (node1.first > node2.first ||
            (node1.first == node2.first && node1.second > node2.second))
          std::swap(node1, node2);
        edges.insert(std::make_pair(node1, node2));
      }
    }
  }

  // insert edges one by one to build the graph
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    int img_id1 = it->first.first;
    int line_id1 = it->first.second;
    int img_id2 = it->second.first;
    int line_id2 = it->second.second;
    const CameraView &view1 = imagecols_->camview(img_id1);
    const CameraView &view2 = imagecols_->camview(img_id2);
    const Line3d &line1 = std::get<0>(getBestTri(img_id1, line_id1));
    const Line3d &line2 = std::get<0>(getBestTri(img_id2, line_id2));
    Line2d &line2d1 = all_lines_2d_[img_id1][line_id1];
    Line2d &line2d2 = all_lines_2d_[img_id2][line_id2];

    double score_3d = linker_clustering.compute_score_3d(line1, line2);
    double score_2d_1to2 =
        linker_clustering.compute_score_2d(line1.projection(view2), line2d2);
    double score_2d_2to1 =
        linker_clustering.compute_score_2d(line2.projection(view1), line2d1);
    double score_2d = std::min(score_2d_1to2, score_2d_2to1);
    double score = std::min(score_3d, score_2d);
    score = score_3d;
    if (score == 0)
      continue;

    PatchNode *node1 = graph->FindOrCreateNode(img_id1, line_id1);
    PatchNode *node2 = graph->FindOrCreateNode(img_id2, line_id2);
    graph->AddEdge(node1, node2, score);
  }
}

void GlobalLineTriangulator::build_tracks_from_clusters(Graph *graph) {
  std::cout << "Start computing line tracks..." << std::endl;
  LineLinker3d linker3d = linker_.linker_3d;
  linker3d.config.set_to_avgtest_merging();

  // collect lines for each node
  std::vector<Line3d> lines_nodes;
  for (auto it = graph->nodes.begin(); it != graph->nodes.end(); ++it) {
    int img_id = (*it)->image_idx;
    int line_id = (*it)->line_idx;
    lines_nodes.push_back(std::get<0>(getBestTri(img_id, line_id)));
  }
  std::vector<int> track_labels;
  if (config_.merging_strategy == "greedy")
    track_labels = merging::ComputeLineTrackLabelsGreedy(*graph, lines_nodes);
  else if (config_.merging_strategy == "exhaustive")
    track_labels = merging::ComputeLineTrackLabelsExhaustive(
        *graph, lines_nodes, linker3d);
  else if (config_.merging_strategy == "avg")
    track_labels =
        merging::ComputeLineTrackLabelsAvg(*graph, lines_nodes, linker3d);
  else
    throw std::runtime_error(
        "Error!The given merging strategy is not implemented");
  if (track_labels.empty())
    return;
  int n_tracks =
      *std::max_element(track_labels.begin(), track_labels.end()) + 1;
  tracks_.clear();
  tracks_.resize(n_tracks);

  // set all lines into tracks
  size_t n_nodes = graph->nodes.size();
  for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
    PatchNode *node = graph->nodes[node_id];
    int img_id = node->image_idx;
    int line_id = node->line_idx;

    Line2d line2d = all_lines_2d_[img_id][line_id];
    Line3d line3d = std::get<0>(getBestTri(img_id, line_id));
    double score = std::get<1>(getBestTri(img_id, line_id));

    size_t track_id = track_labels[node_id];
    if (track_id == -1)
      continue;
    tracks_[track_id].node_id_list.push_back(node_id);
    tracks_[track_id].image_id_list.push_back(img_id);
    tracks_[track_id].line_id_list.push_back(line_id);
    tracks_[track_id].line2d_list.push_back(line2d);
    tracks_[track_id].line3d_list.push_back(line3d);
    tracks_[track_id].score_list.push_back(score);
  }

  // aggregate 3d lines to get final line proposal
  for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
    it->line = merging::Aggregator::aggregate_line3d_list(
        it->line3d_list, it->score_list, config_.num_outliers_aggregator);
  }
}

std::vector<LineTrack> GlobalLineTriangulator::ComputeLineTracks() {
  Graph finalgraph;
  run_clustering(&finalgraph);
  build_tracks_from_clusters(&finalgraph);
  finalgraph.Clear();
  return GetTracks();
}

// visualization
int GlobalLineTriangulator::CountAllTris() const {
  int counter = 0;
  size_t n_images = CountImages();
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      int n_tris = tris_.at(img_id)[line_id].size();
      counter += n_tris;
    }
  }
  return counter;
}

std::vector<TriTuple>
GlobalLineTriangulator::GetScoredTrisNode(const int &image_id,
                                          const int &line_id) const {
  return tris_.at(image_id)[line_id];
}

std::vector<TriTuple>
GlobalLineTriangulator::GetValidScoredTrisNode(const int &image_id,
                                               const int &line_id) const {
  return valid_tris_.at(image_id)[line_id];
}

std::vector<TriTuple>
GlobalLineTriangulator::GetValidScoredTrisNodeSet(const int &img_id,
                                                  const int &line_id) const {
  std::vector<TriTuple> res;
  auto &tris = valid_tris_.at(img_id)[line_id];
  std::map<int, std::pair<int, double>> table; // (ng_img_id, (tri_id, score))
  int n_tris = tris.size();
  for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
    auto &tri = tris[tri_id];
    double score = std::get<1>(tri);
    int ng_img_id = std::get<2>(tri).first;
    if (table.find(ng_img_id) == table.end()) {
      table.insert(std::make_pair(ng_img_id, std::make_pair(tri_id, score)));
    } else {
      if (score > table.at(ng_img_id).second) {
        table.at(ng_img_id) = std::make_pair(tri_id, score);
      }
    }
  }
  for (auto it = table.begin(); it != table.end(); ++it) {
    int tri_id = it->second.first;
    res.push_back(valid_tris_.at(img_id)[line_id][tri_id]);
  }
  return res;
}

int GlobalLineTriangulator::CountAllValidTris() const {
  int counter = 0;
  size_t n_images = CountImages();
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      int n_tris = valid_tris_.at(img_id)[line_id].size();
      counter += n_tris;
    }
  }
  return counter;
}

std::vector<Line3d> GlobalLineTriangulator::GetAllValidTris() const {
  std::vector<Line3d> res;
  size_t n_images = CountImages();
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      auto &tris = valid_tris_.at(img_id)[line_id];
      for (auto it = tris.begin(); it != tris.end(); ++it) {
        const auto &line = std::get<0>(*it);
        res.push_back(line);
      }
    }
  }
  return res;
}

std::vector<Line3d>
GlobalLineTriangulator::GetValidTrisImage(const int &img_id) const {
  std::vector<Line3d> res;
  size_t n_lines = CountLines(img_id);
  for (size_t line_id = 0; line_id < n_lines; ++line_id) {
    auto &tris = valid_tris_.at(img_id)[line_id];
    for (auto it = tris.begin(); it != tris.end(); ++it) {
      const auto &line = std::get<0>(*it);
      res.push_back(line);
    }
  }
  return res;
}

std::vector<Line3d>
GlobalLineTriangulator::GetValidTrisNode(const int &img_id,
                                         const int &line_id) const {
  std::vector<Line3d> res;
  auto &tris = valid_tris_.at(img_id)[line_id];
  for (auto it = tris.begin(); it != tris.end(); ++it) {
    const auto &line = std::get<0>(*it);
    res.push_back(line);
  }
  return res;
}

std::vector<Line3d>
GlobalLineTriangulator::GetValidTrisNodeSet(const int &img_id,
                                            const int &line_id) const {
  std::vector<Line3d> res;
  auto &tris = valid_tris_.at(img_id)[line_id];
  std::map<int, std::pair<int, double>> table; // (ng_img_id, (tri_id, score))
  int n_tris = tris.size();
  for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
    auto &tri = tris[tri_id];
    double score = std::get<1>(tri);
    int ng_img_id = std::get<2>(tri).first;
    if (table.find(ng_img_id) == table.end()) {
      table.insert(std::make_pair(ng_img_id, std::make_pair(tri_id, score)));
    } else {
      if (score > table.at(ng_img_id).second) {
        table.at(ng_img_id) = std::make_pair(tri_id, score);
      }
    }
  }
  for (auto it = table.begin(); it != table.end(); ++it) {
    int tri_id = it->second.first;
    auto &tri = tris[tri_id];
    const auto &line = std::get<0>(tri);
    res.push_back(line);
  }
  return res;
}

std::vector<Line3d> GlobalLineTriangulator::GetAllBestTris() const {
  std::vector<Line3d> res;
  size_t n_images = CountImages();
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      res.push_back(std::get<0>(getBestTri(img_id, line_id)));
    }
  }
  return res;
}

std::vector<Line3d> GlobalLineTriangulator::GetAllValidBestTris() const {
  std::vector<Line3d> res;
  size_t n_images = CountImages();
  for (const int &img_id : imagecols_->get_img_ids()) {
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      if (!valid_flags_.at(img_id)[line_id])
        continue;
      res.push_back(std::get<0>(getBestTri(img_id, line_id)));
    }
  }
  return res;
}

std::vector<Line3d>
GlobalLineTriangulator::GetBestTrisImage(const int &img_id) const {
  std::vector<Line3d> res;
  size_t n_lines = CountLines(img_id);
  for (size_t line_id = 0; line_id < n_lines; ++line_id) {
    res.push_back(std::get<0>(getBestTri(img_id, line_id)));
  }
  return res;
}

Line3d GlobalLineTriangulator::GetBestTriNode(const int &img_id,
                                              const int &line_id) const {
  return std::get<0>(getBestTri(img_id, line_id));
}

TriTuple
GlobalLineTriangulator::GetBestScoredTriNode(const int &img_id,
                                             const int &line_id) const {
  return getBestTri(img_id, line_id);
}

std::vector<int> GlobalLineTriangulator::GetSurvivedLinesImage(
    const int &image_id, const int &n_visible_views) const {
  std::vector<int> survivers;
  for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
    if (it->count_images() < n_visible_views)
      continue;
    size_t n_lines = it->count_lines();
    for (size_t i = 0; i < n_lines; ++i) {
      int img_id = it->image_id_list[i];
      if (img_id != image_id)
        continue;
      survivers.push_back(it->line_id_list[i]);
    }
  }
  return survivers;
}

} // namespace triangulation

} // namespace limap
