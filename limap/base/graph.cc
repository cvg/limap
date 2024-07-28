#include "base/graph.h"

namespace limap {

Edge::Edge(size_t source_idx, size_t destination_idx, double similarity) {
  node_idx1 = source_idx;
  node_idx2 = destination_idx;
  sim = similarity;
}

size_t Edge::OtherIdx(size_t &node_idx) const {
  return (this->node_idx1 == node_idx) ? this->node_idx2 : this->node_idx1;
}

PatchNode::PatchNode(int im_idx_, size_t line_idx_)
    : image_idx(im_idx_), line_idx(line_idx_) {}

void PatchNode::AddEdge(PatchNode *neighbor, Edge *edge) {
  out_edges.push_back(edge);
  neighbor->in_edges.push_back(edge);
}

size_t Graph::AddNode(PatchNode *node) {
  nodes.push_back(node);
  node->node_idx = nodes.size() - 1;
  return node->node_idx;
}

std::vector<size_t> Graph::GetInputDegrees() {
  std::vector<size_t> input_degrees;
  for (auto &node : nodes) {
    input_degrees.push_back(node->in_edges.size());
  }
  return input_degrees;
}

std::vector<size_t> Graph::GetOutputDegrees() {
  std::vector<size_t> output_degrees;
  for (auto &node : nodes) {
    output_degrees.push_back(node->out_edges.size());
  }
  return output_degrees;
}

std::vector<std::pair<double, size_t>> Graph::GetScores() {
  std::vector<std::pair<double, size_t>> scores;
  for (auto &node : nodes) {
    double s = 0.;
    for (auto &edge : node->out_edges) {
      s += edge->sim;
    }
    scores.push_back(std::make_pair(s, node->node_idx));
  }
  return scores;
}

PatchNode *Graph::FindOrCreateNode(int image_idx, size_t line_idx) {
  auto it = node_map.find(std::make_pair(image_idx, line_idx));

  if (it != node_map.end()) {
    return nodes[it->second];
  } else {
    PatchNode *node = new PatchNode(image_idx, line_idx);
    size_t node_idx = AddNode(node);
    node->node_idx = node_idx;
    node_map.insert(
        std::make_pair(std::make_pair(image_idx, line_idx), node_idx));
    return node;
  }
}

size_t Graph::GetNodeID(int image_idx, size_t line_idx) {
  auto it = node_map.find(std::make_pair(image_idx, line_idx));
  if (it != node_map.end())
    return it->second;
  else
    return -1;
}

void Graph::AddEdge(PatchNode *node1, PatchNode *node2, double sim) {
  Edge *edge = new Edge(node1->node_idx, node2->node_idx, sim);
  undirected_edges.push_back(edge);
  edge->edge_idx = undirected_edges.size() - 1;
  node1->AddEdge(node2, edge);
  node2->AddEdge(node1, edge);
}

void Graph::RegisterMatches(int im1_idx, int im2_idx,
                            size_t *matches,      // Nx2
                            double *similarities, // Nx1
                            size_t n_matches) {

  for (size_t match_idx = 0; match_idx < n_matches; ++match_idx) {
    size_t line_idx1 = matches[2 * match_idx];
    size_t line_idx2 = matches[2 * match_idx + 1];
    double similarity = similarities[match_idx];

    PatchNode *node1 = FindOrCreateNode(im1_idx, line_idx1);
    PatchNode *node2 = FindOrCreateNode(im2_idx, line_idx2);

    AddEdge(node1, node2, similarities[match_idx]);
  }
}

void Graph::Clear() {
  nodes.clear();
  node_map.clear();
  undirected_edges.clear();
}

Graph::~Graph() {
  for (PatchNode *node_ptr : nodes) {
    delete node_ptr;
  }
  for (Edge *edge_ptr : undirected_edges) {
    delete edge_ptr;
  }
}

void DirectedGraph::AddEdgeDirected(PatchNode *node1, PatchNode *node2,
                                    double sim) {
  Edge *edge = new Edge(node1->node_idx, node2->node_idx, sim);
  directed_edges.push_back(edge);
  edge->edge_idx = directed_edges.size() - 1;
  node1->AddEdge(node2, edge);
}

void DirectedGraph::RegisterMatchesDirected(int im1_idx, int im2_idx,
                                            size_t *matches,      // Nx2
                                            double *similarities, // Nx1
                                            size_t n_matches) {

  for (size_t match_idx = 0; match_idx < n_matches; ++match_idx) {
    size_t line_idx1 = matches[2 * match_idx];
    size_t line_idx2 = matches[2 * match_idx + 1];
    double similarity = similarities[match_idx];

    PatchNode *node1 = FindOrCreateNode(im1_idx, line_idx1);
    PatchNode *node2 = FindOrCreateNode(im2_idx, line_idx2);

    AddEdgeDirected(node1, node2, similarities[match_idx]);
  }
}

void DirectedGraph::Clear() {
  nodes.clear();
  node_map.clear();
  directed_edges.clear();
}

DirectedGraph::~DirectedGraph() {
  for (Edge *edge_ptr : directed_edges) {
    delete edge_ptr;
  }
}

size_t union_find_get_root(const size_t node_idx,
                           std::vector<int> &parent_nodes) {
  if (parent_nodes[node_idx] == -1) {
    return node_idx;
  }
  // Union-find path compression heuristic.
  parent_nodes[node_idx] =
      union_find_get_root(parent_nodes[node_idx], parent_nodes);
  return parent_nodes[node_idx];
}

std::vector<size_t> ComputeTrackLabels(const Graph &graph) {
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

  // auto start = std::chrono::high_resolution_clock::now();
  // Build the MSF.
  std::sort(edges.begin(), edges.end());
  std::reverse(edges.begin(), edges.end());

  std::vector<int> parent_nodes(n_nodes, -1);
  std::vector<std::set<int>> images_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_idx);
  }

  for (auto it : edges) {
    size_t node_idx1 = std::get<1>(it);
    size_t node_idx2 = std::get<2>(it);

    size_t root1 = union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      std::set<int> intersection;
      std::set_intersection(
          images_in_track[root1].begin(), images_in_track[root1].end(),
          images_in_track[root2].begin(), images_in_track[root2].end(),
          std::inserter(intersection, intersection.begin()));
      if (intersection.size() != 0) {
        continue;
      }
      // Union-find merging heuristic.
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
      } else {
        parent_nodes[root2] = root1;
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
      }
    }
  }

  // Compute the tracks.
  std::vector<size_t> track_labels(n_nodes, -1);

  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1) {
      track_labels[node_idx] = n_tracks++;
    }
  }
  STDLOG(INFO) << "# tracks:"
               << " " << n_tracks << std::endl;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (track_labels[node_idx] != -1) {
      continue;
    }
    track_labels[node_idx] =
        track_labels[union_find_get_root(node_idx, parent_nodes)];
  }

  return track_labels;
}

std::vector<double> ComputeScoreLabels(const Graph &graph,
                                       std::vector<size_t> &track_labels) {

  const size_t n_nodes = graph.nodes.size();
  std::vector<double> score_labels(n_nodes, 0.0);
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    PatchNode *node = graph.nodes[node_idx];

    double score = 0.;
    for (auto &edge : node->out_edges) {
      if (track_labels[node_idx] == track_labels[edge->OtherIdx(node_idx)]) {
        score += edge->sim;
      }
    }
    score_labels[node_idx] = score;
  }
  return score_labels;
}

std::vector<bool> ComputeRootLabels(const Graph &graph,
                                    std::vector<size_t> track_labels,
                                    std::vector<double> score_labels) {
  // Find the root nodes.
  const size_t n_nodes = graph.nodes.size();
  const size_t n_tracks =
      (*std::max_element(track_labels.begin(), track_labels.end())) + 1;
  std::vector<std::pair<double, size_t>> scores;

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    scores.push_back(std::make_pair(score_labels[node_idx], node_idx));
  }

  std::sort(scores.begin(), scores.end());
  std::reverse(scores.begin(), scores.end());

  std::vector<bool> is_root(n_nodes, false);
  std::vector<bool> has_root(n_tracks, false);

  for (auto it : scores) {
    size_t node_idx = it.second;

    if (has_root[track_labels[node_idx]]) {
      continue;
    }

    is_root[node_idx] = true;
    has_root[track_labels[node_idx]] = true;
  }

  return is_root;
}

std::vector<std::pair<size_t, size_t>>
CountTrackEdges(const Graph &graph, const std::vector<size_t> &track_labels,
                const std::vector<bool> is_root) {
  // first holds edges A-B, second holds edges b-b
  const size_t n_nodes = graph.nodes.size();
  std::vector<std::pair<size_t, size_t>> track_edge_counts(n_nodes, {0, 0});
  for (Edge *edge : graph.undirected_edges) {
    size_t node_idx1 = edge->node_idx1;
    size_t node_idx2 = edge->node_idx2;
    size_t track_idx1 = track_labels[node_idx1];
    size_t track_idx2 = track_labels[node_idx2];
    if (track_idx1 == track_idx2) {
      if (is_root[node_idx1] || is_root[node_idx2]) {
        track_edge_counts[track_idx1].first += 1;
      } else {
        track_edge_counts[track_idx1].second += 1;
      }
    }
  }

  return track_edge_counts;
}

} // namespace limap
