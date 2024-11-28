#ifndef LIMAP_BASE_GRAPH_H_
#define LIMAP_BASE_GRAPH_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "util/simple_logger.h"
#include <colmap/util/logging.h>

// Copyright (c) 2020, ETH Zurich, CVG, Mihai Dusmanu
// (mihai.dusmanu@inf.ethz.ch) Adapted by Philipp Lindenberger for the
// pixel-perfect-sfm project. Adapted again by Shaohui Liu for supporting line
// graphs

namespace limap {

typedef std::tuple<double, size_t, size_t> edge_tuple;

class Edge {
public:
  Edge(size_t, size_t, double);
  size_t OtherIdx(size_t &) const;
  size_t node_idx1 = -1;
  size_t node_idx2 = -1;
  size_t edge_idx = -1;
  double sim;
};

class PatchNode {
public:
  PatchNode(int, size_t);
  void AddEdge(PatchNode *, Edge *);
  int image_idx = -1;
  size_t line_idx = -1;
  size_t node_idx = -1;
  std::vector<Edge *> out_edges; // points to graph's edge
  std::vector<Edge *> in_edges;  // points to graph's edge
};

class Graph {
public:
  void Clear();
  size_t AddNode(PatchNode *);
  PatchNode *FindOrCreateNode(int, size_t);
  size_t GetNodeID(int, size_t);
  std::vector<size_t> GetInputDegrees();
  std::vector<size_t> GetOutputDegrees();
  std::vector<std::pair<double, size_t>> GetScores();
  void AddEdge(PatchNode *node1, PatchNode *node2, double sim);
  void RegisterMatches(int im1_idx, int im2_idx,
                       size_t *matches,      // Nx2
                       double *similarities, // Nx1
                       size_t n_matches);
  ~Graph();

  std::vector<PatchNode *> nodes;
  std::map<std::pair<int, size_t>, size_t> node_map;
  std::vector<Edge *> undirected_edges;
};

class DirectedGraph : public Graph {
public:
  void Clear();
  void AddEdgeDirected(PatchNode *node1, PatchNode *node2, double sim);
  void RegisterMatchesDirected(int im1_idx, int im2_idx,
                               size_t *matches,      // Nx2
                               double *similarities, // Nx1
                               size_t n_matches);
  ~DirectedGraph();

  std::vector<Edge *> directed_edges;
};

size_t union_find_get_root(const size_t node_idx,
                           std::vector<int> &parent_nodes);

std::vector<size_t> ComputeTrackLabels(const Graph &graph);

std::vector<double> ComputeScoreLabels(const Graph &graph,
                                       std::vector<size_t> &track_labels);

std::vector<bool> ComputeRootLabels(const Graph &graph,
                                    std::vector<size_t> track_labels,
                                    std::vector<double> score_labels);

std::vector<std::pair<size_t, size_t>>
CountTrackEdges(const Graph &graph, const std::vector<size_t> &track_labels,
                const std::vector<bool> is_root);

} // namespace limap

#endif
