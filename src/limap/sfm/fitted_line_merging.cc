#include "limap/sfm/fitted_line_merging.h"

#include <mutex>

#include <colmap/math/union_find.h>
#include <colmap/scene/image.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>

#include "limap/scene/structure2d.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/sfm/structure_observation_manager.h"

namespace limap {

bool LineMergingOptions::Check() const {
  CHECK_OPTION_GE(num_outliers_aggregator, 0);
  CHECK_OPTION_GE(filtering2d_th_angular_2d, 0.0);
  CHECK_OPTION_GE(filtering2d_th_perp_2d, 0.0);
  CHECK_OPTION_GT(min_visible_views, 0);
  return true;
}

void MergeFittedLines3D(
    HolisticReconstruction &recon,
    const std::map<colmap::image_t, std::vector<colmap::image_t>> &neighbors,
    const LineMergingOptions &options) {
  CHECK(options.Check());

  StructureReconstruction &srec = recon.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  auto &structures2d = srec.Structures2d();
  auto &lines3d = srec.Lines3D();

  if (lines3d.empty()) {
    LOG(INFO) << "No 3D lines to merge.";
    return;
  }

  LOG(INFO) << "Merging " << lines3d.size() << " fitted 3D lines...";

  // Configure linkers for spatial merging
  LineLinker2dOptions linker2d_options = options.linker2d_options;
  LineLinker3dOptions linker3d_options = options.linker3d_options;
  linker3d_options.SetToSpatialMerging();
  LineLinker linker(linker2d_options, linker3d_options);

  // Build node list: each Line3d with its 2D observation
  // Node = (image_id, line_id)
  struct NodeInfo {
    line3D_t line3d_id;
    colmap::image_t image_id;
    line2D_t line_id;
    Line3d line3d;
    Line2d line2d;
  };
  std::vector<NodeInfo> nodes;
  FlatHashMap<line3D_t, size_t> line3d_to_node;

  for (const auto &[lid, line] : lines3d) {
    // Each Line3d should have at least one observation
    if (line.track.Length() == 0) {
      continue;
    }

    // Take the first observation as the primary reference
    const auto &elem = line.track.Element(0);
    const colmap::image_t img_id = elem.image_id;
    const line2D_t line_id = static_cast<line2D_t>(elem.point2D_idx);

    auto it_s = structures2d.find(img_id);
    if (it_s == structures2d.end()) {
      continue;
    }
    const Structure2d &S = it_s->second;
    if (line_id >= static_cast<line2D_t>(S.NumLines())) {
      continue;
    }

    NodeInfo info;
    info.line3d_id = lid;
    info.image_id = img_id;
    info.line_id = line_id;
    info.line3d = line;
    info.line2d = S.Line(line_id);

    // Compute uncertainty if not set (required for innerseg check)
    if (info.line3d.uncertainty < 0) {
      const colmap::Image &img = point_recon.Image(img_id);
      // var2d = 5.0 is the typical default for 2D variance
      info.line3d.SetUncertainty(info.line3d.ComputeUncertainty(img, 5.0));
    }

    line3d_to_node[lid] = nodes.size();
    nodes.push_back(std::move(info));
  }

  if (nodes.empty()) {
    LOG(INFO) << "No valid nodes for merging.";
    return;
  }

  // Build edges based on 2D+3D similarity
  using Edge = std::pair<size_t, size_t>;
  std::vector<Edge> edges;

  // Helper to check if two nodes should be connected
  auto check_connection = [&](const NodeInfo &n1, const NodeInfo &n2) -> bool {
    // Check 3D similarity
    if (!linker.CheckConnection3d(n1.line3d, n2.line3d)) {
      return false;
    }

    // Check 2D similarity: project n1's 3D line to n2's image and compare
    const colmap::Image &img2 = point_recon.Image(n2.image_id);
    auto proj1_in_2 = n1.line3d.Projection(img2);
    if (!proj1_in_2 || !linker.CheckConnection2d(*proj1_in_2, n2.line2d)) {
      return false;
    }

    // Also check the reverse: project n2's 3D line to n1's image
    const colmap::Image &img1 = point_recon.Image(n1.image_id);
    auto proj2_in_1 = n2.line3d.Projection(img1);
    if (!proj2_in_1 || !linker.CheckConnection2d(*proj2_in_1, n1.line2d)) {
      return false;
    }

    return true;
  };

  // Self-similarity within same image
  std::map<colmap::image_t, std::vector<size_t>> nodes_per_image;
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes_per_image[nodes[i].image_id].push_back(i);
  }

  for (const auto &[img_id, node_indices] : nodes_per_image) {
    const size_t n = node_indices.size();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        const NodeInfo &n1 = nodes[node_indices[i]];
        const NodeInfo &n2 = nodes[node_indices[j]];

        // For same-image nodes, only check 2D and 3D
        if (!linker.CheckConnection3d(n1.line3d, n2.line3d)) {
          continue;
        }
        if (!linker.CheckConnection2d(n1.line2d, n2.line2d)) {
          continue;
        }

        edges.emplace_back(node_indices[i], node_indices[j]);
      }
    }
  }

  // Cross-image similarity with neighbors
  LOG(INFO) << "Computing cross-image similarity edges...";

  {
    std::mutex cross_edges_mutex;
    colmap::ThreadPool pool(
        colmap::GetEffectiveNumThreads(options.num_threads));
    for (size_t i = 0; i < nodes.size(); ++i) {
      pool.AddTask([&, i]() {
        const NodeInfo &n1 = nodes[i];
        auto it_neighbors = neighbors.find(n1.image_id);
        if (it_neighbors == neighbors.end()) {
          return;
        }

        std::vector<Edge> local_edges;

        for (const colmap::image_t ng_img_id : it_neighbors->second) {
          auto it_ng = nodes_per_image.find(ng_img_id);
          if (it_ng == nodes_per_image.end()) {
            continue;
          }

          for (const size_t j : it_ng->second) {
            // Avoid duplicates: only add edge if i < j
            if (i >= j) {
              continue;
            }

            const NodeInfo &n2 = nodes[j];
            if (check_connection(n1, n2)) {
              local_edges.emplace_back(i, j);
            }
          }
        }

        if (!local_edges.empty()) {
          std::lock_guard<std::mutex> lock(cross_edges_mutex);
          edges.insert(edges.end(), local_edges.begin(), local_edges.end());
        }
      });
    }
    pool.Wait();
  }

  LOG(INFO) << "Found " << edges.size() << " similarity edges.";

  // Union-Find clustering
  colmap::UnionFind<size_t> uf;
  uf.Reserve(nodes.size());

  for (const auto &[i, j] : edges) {
    uf.Union(i, j);
  }

  // Compute clusters
  FlatHashMap<size_t, size_t> root_to_cluster;
  FlatHashMap<size_t, std::vector<size_t>> clusters;
  size_t n_clusters = 0;

  for (size_t i = 0; i < nodes.size(); ++i) {
    size_t root = uf.Find(i);
    if (root_to_cluster.find(root) == root_to_cluster.end()) {
      root_to_cluster[root] = n_clusters++;
    }
    clusters[root_to_cluster[root]].push_back(i);
  }

  LOG(INFO) << "Merged into " << n_clusters << " clusters.";

  // Clear old 3D lines and reset 2D assignments
  for (auto &[img_id, structure] : structures2d) {
    for (size_t i = 0; i < structure.NumLines(); ++i) {
      structure.Line(i).line3D_id = kInvalidLine3dId;
    }
  }
  lines3d.clear();

  // Create new Line3d for each cluster
  line3D_t next_lid = 0;

  for (const auto &[cluster_id, member_indices] : clusters) {
    // Collect all Line3d in this cluster
    std::vector<Line3d> lines_in_cluster;
    lines_in_cluster.reserve(member_indices.size());

    colmap::Track merged_track;

    for (const size_t idx : member_indices) {
      const NodeInfo &node = nodes[idx];
      lines_in_cluster.push_back(node.line3d);

      // Add all observations from the original Line3d's track
      for (const auto &elem : node.line3d.track.Elements()) {
        merged_track.AddElement(elem);
      }
    }

    // Aggregate geometry
    Line3d merged_line;
    if (lines_in_cluster.size() <
        static_cast<size_t>(options.num_outliers_aggregator * 2)) {
      // Small cluster: take the longest line
      double max_length = 0.0;
      for (const auto &l : lines_in_cluster) {
        if (l.Length() > max_length) {
          max_length = l.Length();
          merged_line = l;
        }
      }
    } else {
      // Large cluster: aggregate using SVD
      merged_line = AggregateLine3dList(lines_in_cluster,
                                        options.num_outliers_aggregator);
    }

    merged_line.track = std::move(merged_track);

    const line3D_t lid = next_lid++;
    lines3d[lid] = std::move(merged_line);

    // Update 2D → 3D assignments
    for (const auto &elem : lines3d[lid].track.Elements()) {
      auto it_s = structures2d.find(elem.image_id);
      if (it_s != structures2d.end()) {
        const line2D_t lid2d = static_cast<line2D_t>(elem.point2D_idx);
        if (lid2d < static_cast<line2D_t>(it_s->second.NumLines())) {
          it_s->second.Line(lid2d).line3D_id = lid;
        }
      }
    }
  }

  StructureObservationManager obs_mgr(srec);
  obs_mgr.FilterLines3dByReprojection(options.filtering2d_th_angular_2d,
                                      options.filtering2d_th_perp_2d,
                                      options.num_outliers_aggregator);
  LOG(INFO) << "Merging complete. Created " << lines3d.size() << " 3D lines.";

  // Optional: Spatial remerging (same as in triangulation)
  if (options.enable_remerge && lines3d.size() > 0) {
    LineLinker3dOptions linker3d_remerge = options.linker3d_remerge_options;
    linker3d_remerge.SetToSpatialMerging();
    LineLinker3d linker3d(linker3d_remerge);

    colmap::UnionFind<line3D_t> uf_remerge;
    uf_remerge.Reserve(lines3d.size());

    std::vector<line3D_t> line_ids;
    line_ids.reserve(lines3d.size());
    for (const auto &[lid, _] : lines3d) {
      line_ids.push_back(lid);
    }

    for (size_t i = 0; i < line_ids.size(); ++i) {
      const Line3d &l1 = lines3d.at(line_ids[i]);
      for (size_t j = i + 1; j < line_ids.size(); ++j) {
        const Line3d &l2 = lines3d.at(line_ids[j]);
        if (linker3d.CheckConnection(l1, l2)) {
          uf_remerge.Union(line_ids[i], line_ids[j]);
        }
      }
    }

    // Compute groups and rebuild
    FlatHashMap<line3D_t, size_t> root_to_group;
    FlatHashMap<size_t, std::vector<Line3d>> lines_per_group;
    FlatHashMap<size_t, colmap::Track> tracks_per_group;
    size_t n_groups = 0;

    for (const auto &[lid, line] : lines3d) {
      line3D_t root = uf_remerge.Find(lid);
      if (root_to_group.find(root) == root_to_group.end()) {
        root_to_group[root] = n_groups++;
      }
      size_t group_id = root_to_group[root];
      lines_per_group[group_id].push_back(line);
      for (const auto &elem : line.track.Elements()) {
        tracks_per_group[group_id].AddElement(elem);
      }
    }

    // Clear and rebuild
    for (auto &[img_id, structure] : structures2d) {
      for (size_t i = 0; i < structure.NumLines(); ++i) {
        structure.Line(i).line3D_id = kInvalidLine3dId;
      }
    }
    lines3d.clear();
    next_lid = 0;

    for (auto &[group_id, lines_in_group] : lines_per_group) {
      Line3d merged_line =
          AggregateLine3dList(lines_in_group, options.num_outliers_aggregator);
      merged_line.track = std::move(tracks_per_group[group_id]);

      const line3D_t lid = next_lid++;
      lines3d[lid] = std::move(merged_line);

      for (const auto &elem : lines3d[lid].track.Elements()) {
        auto it_s = structures2d.find(elem.image_id);
        if (it_s != structures2d.end()) {
          const line2D_t lid2d = static_cast<line2D_t>(elem.point2D_idx);
          if (lid2d < static_cast<line2D_t>(it_s->second.NumLines())) {
            it_s->second.Line(lid2d).line3D_id = lid;
          }
        }
      }
    }

    LOG(INFO) << "Remerging complete. Final " << lines3d.size() << " 3D lines.";
  }

  // Filter by minimum visible views
  StructureObservationManager obs_mgr_final(srec);
  obs_mgr_final.FilterLines3dByMinVisibleViews(options.min_visible_views);
  LOG(INFO) << "After min_visible_views filter: " << lines3d.size()
            << " 3D lines.";

  // Logging: track length statistics
  std::vector<int> track_len_ths = {2, 3, 5, 10};
  std::vector<int> num_lines_with_len_geq(track_len_ths.size(), 0);
  for (const auto &kv : lines3d) {
    size_t len = kv.second.track.Length();
    for (size_t i = 0; i < track_len_ths.size(); ++i) {
      if (len >= track_len_ths[i]) {
        num_lines_with_len_geq[i]++;
      }
    }
  }

  for (size_t i = 0; i < track_len_ths.size(); ++i) {
    LOG(INFO) << "#lines with track length >= " << track_len_ths[i] << ": "
              << num_lines_with_len_geq[i];
  }
}

} // namespace limap
