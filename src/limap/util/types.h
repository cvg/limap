#pragma once

#include <colmap/util/hash_containers.h>
#include <colmap/util/types.h>

#include <utility>

namespace limap {

// COLMAP's hash container aliases (backend picked at configure time).
// Flat*: faster, but a rehash invalidates references to stored elements and
// boost's erase(iterator) returns void. Node*: stable references, as
// std::unordered_*. Use Node* for element stores, Ceres parameter blocks, and
// iterator-based erase loops; Flat* elsewhere.
using colmap::FlatHashMap;
using colmap::FlatHashSet;
using colmap::NodeHashMap;
using colmap::NodeHashSet;

using point2D_t = colmap::point2D_t;
using line2D_t = colmap::point2D_t;
using feature2D_t = colmap::point2D_t;
using group2D_t = colmap::point2D_t;

using point3D_t = colmap::point3D_t;
using line3D_t = colmap::point3D_t;
using feature3D_t = colmap::point3D_t;
using group3D_t = colmap::point3D_t;

constexpr point3D_t kInvalidPoint3dId = std::numeric_limits<point3D_t>::max();
constexpr line3D_t kInvalidLine3dId = std::numeric_limits<line3D_t>::max();
constexpr feature3D_t kInvalidFeature3dId =
    std::numeric_limits<feature3D_t>::max();
constexpr group3D_t kInvalidGroup3dId = std::numeric_limits<group3D_t>::max();

// A node in the graph (img_id, feature_id)
using Node2d = std::pair<colmap::image_t, feature2D_t>;

// COLMAP has no std::hash for std::pair, so Node2d containers need PairHash.
using Node2dSet = FlatHashSet<Node2d, colmap::PairHash>;
template <typename T>
using Node2dMap = FlatHashMap<Node2d, T, colmap::PairHash>;

// Represents a line correspondence between two images
struct LineMatch {
  line2D_t line2D_idx1;
  line2D_t line2D_idx2;
};
typedef std::vector<LineMatch> LineMatches;

// Represents a group correspondence between two images
struct GroupMatch {
  size_t group2D_idx1;
  size_t group2D_idx2;
};
typedef std::vector<GroupMatch> GroupMatches;

} // namespace limap
