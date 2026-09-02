#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "limap/geometry/line2d.h"
#include "limap/util/eigen_types.h"
#include "limap/util/types.h"

#include <colmap/util/logging.h>

namespace limap {

template <typename PointId, typename LineId> struct WireframeConnection {
  PointId point_idx;
  LineId line_idx;
  double w;

  WireframeConnection() = default;
  WireframeConnection(PointId pid, LineId lid, double weight = 1.0);
};

template <typename PointId, typename LineId> class Wireframe {
public:
  Wireframe() = default;

  void AddEdge(PointId point_id, LineId line_id, double w);
  void AddEdge(const WireframeConnection<PointId, LineId> &c);

  void RemoveEdge(PointId point_id, LineId line_id);
  void RemoveEdge(const WireframeConnection<PointId, LineId> &c);

  void Clear();
  std::vector<LineId> GetNeighboringLineIds(PointId point_id) const;
  bool CountNeighboringLines(PointId point_id) const;
  std::vector<PointId> GetNeighboringPointIds(LineId line_id) const;
  bool CountNeighboringPoints(LineId line_id) const;
  size_t CountPoints() const;
  size_t CountLines() const;
  size_t CountEdges() const;
  double GetEdgeWeight(PointId point_id, LineId line_id) const;
  WireframeConnection<PointId, LineId> GetEdge(PointId point_id,
                                               LineId line_id) const;
  std::vector<WireframeConnection<PointId, LineId>> GetAllEdges() const;

private:
  FlatHashMap<PointId, FlatHashSet<LineId>> point_to_lines_;
  FlatHashMap<LineId, FlatHashSet<PointId>> line_to_points_;
  FlatHashMap<std::pair<PointId, LineId>, double, colmap::PairHash>
      edge_weights_;
};

using WireframeConnection2d = WireframeConnection<point2D_t, line2D_t>;
using Wireframe2d = Wireframe<point2D_t, line2D_t>;
using WireframeConnection3d = WireframeConnection<point3D_t, line3D_t>;
using Wireframe3d = Wireframe<point3D_t, line3D_t>;

std::shared_ptr<Wireframe2d> CreateWireframe2d(const std::vector<V2D> &points,
                                               const std::vector<Line2d> &lines,
                                               double threshold = 2.0);

// Implementation
template <typename PointId, typename LineId>
WireframeConnection<PointId, LineId>::WireframeConnection(PointId pid,
                                                          LineId lid,
                                                          double weight)
    : point_idx(pid), line_idx(lid), w(weight) {}

template <typename PointId, typename LineId>
void Wireframe<PointId, LineId>::AddEdge(PointId point_id, LineId line_id,
                                         double w) {
  point_to_lines_[point_id].insert(line_id);
  line_to_points_[line_id].insert(point_id);
  edge_weights_[{point_id, line_id}] = w;
}

template <typename PointId, typename LineId>
void Wireframe<PointId, LineId>::AddEdge(
    const WireframeConnection<PointId, LineId> &c) {
  AddEdge(c.point_idx, c.line_idx, c.w);
}

template <typename PointId, typename LineId>
void Wireframe<PointId, LineId>::RemoveEdge(PointId point_id, LineId line_id) {
  auto it = edge_weights_.find({point_id, line_id});
  THROW_CHECK(it != edge_weights_.end());
  point_to_lines_[point_id].erase(line_id);
  line_to_points_[line_id].erase(point_id);
  edge_weights_.erase(it);
}

template <typename PointId, typename LineId>
void Wireframe<PointId, LineId>::RemoveEdge(
    const WireframeConnection<PointId, LineId> &c) {
  RemoveEdge(c.point_idx, c.line_idx);
}

template <typename PointId, typename LineId>
void Wireframe<PointId, LineId>::Clear() {
  point_to_lines_.clear();
  line_to_points_.clear();
  edge_weights_.clear();
}

template <typename PointId, typename LineId>
std::vector<LineId>
Wireframe<PointId, LineId>::GetNeighboringLineIds(PointId point_id) const {
  std::vector<LineId> neighbors;
  auto it = point_to_lines_.find(point_id);
  if (it != point_to_lines_.end()) {
    neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
  }
  return neighbors;
}

template <typename PointId, typename LineId>
bool Wireframe<PointId, LineId>::CountNeighboringLines(PointId point_id) const {
  auto it = point_to_lines_.find(point_id);
  return it != point_to_lines_.end() && !it->second.empty();
}

template <typename PointId, typename LineId>
std::vector<PointId>
Wireframe<PointId, LineId>::GetNeighboringPointIds(LineId line_id) const {
  std::vector<PointId> neighbors;
  auto it = line_to_points_.find(line_id);
  if (it != line_to_points_.end()) {
    neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
  }
  return neighbors;
}

template <typename PointId, typename LineId>
bool Wireframe<PointId, LineId>::CountNeighboringPoints(LineId line_id) const {
  auto it = line_to_points_.find(line_id);
  return it != line_to_points_.end() && !it->second.empty();
}

template <typename PointId, typename LineId>
size_t Wireframe<PointId, LineId>::CountPoints() const {
  return point_to_lines_.size();
}

template <typename PointId, typename LineId>
size_t Wireframe<PointId, LineId>::CountLines() const {
  return line_to_points_.size();
}

template <typename PointId, typename LineId>
size_t Wireframe<PointId, LineId>::CountEdges() const {
  return edge_weights_.size();
}

template <typename PointId, typename LineId>
double Wireframe<PointId, LineId>::GetEdgeWeight(PointId point_id,
                                                 LineId line_id) const {
  auto it = edge_weights_.find({point_id, line_id});
  THROW_CHECK(it != edge_weights_.end());
  return it->second;
}

template <typename PointId, typename LineId>
WireframeConnection<PointId, LineId>
Wireframe<PointId, LineId>::GetEdge(PointId point_id, LineId line_id) const {
  auto it = edge_weights_.find({point_id, line_id});
  THROW_CHECK(it != edge_weights_.end());
  return WireframeConnection<PointId, LineId>(point_id, line_id, it->second);
}

template <typename PointId, typename LineId>
std::vector<WireframeConnection<PointId, LineId>>
Wireframe<PointId, LineId>::GetAllEdges() const {
  std::vector<WireframeConnection<PointId, LineId>> edges;
  for (const auto &kv : edge_weights_) {
    edges.emplace_back(kv.first.first, kv.first.second, kv.second);
  }
  return edges;
}

} // namespace limap
