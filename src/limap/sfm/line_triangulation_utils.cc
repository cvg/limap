#include "limap/sfm/line_triangulation_utils.h"

#include <mutex>

#include <colmap/math/union_find.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>

#include "limap/estimators/triangulation/functions.h"
#include "limap/geometry/inf_line3d.h"

namespace limap {
namespace internal {
namespace line_triangulation {

namespace {

// l1: "query / support" line
// l2: reference line along which depths are defined (the proposal line)
// depth_start_l2, depth_end_l2: depths at l2.start and l2.end in the ref camera
double EndpointsScaleinvDistanceOneway(const Line3d &l1, const Line3d &l2,
                                       double depth_start_l2,
                                       double depth_end_l2) {
  constexpr double MAX_DIST = std::numeric_limits<double>::max();
  double dist_start = (l1.start - l2.start).norm();
  double dist_end = (l1.end - l2.end).norm();
  if (depth_start_l2 <= 0 || depth_end_l2 <= 0) {
    return MAX_DIST;
  }
  return std::max(dist_start / depth_start_l2, dist_end / depth_end_l2);
}

} // namespace

void CollectShared3DPoints(const Node2d &node1, const Node2d &node2,
                           const StructureReconstruction &srec,
                           std::vector<V3D> &points3d) {
  points3d.clear();

  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const colmap::image_t img1 = node1.first;
  const colmap::image_t img2 = node2.first;

  const Structure2d &S1 = srec.Structure2d(img1);
  const Structure2d &S2 = srec.Structure2d(img2);

  const Wireframe2d &W1 = S1.Wireframe();
  const Wireframe2d &W2 = S2.Wireframe();

  const colmap::Image &I1 = point_recon.Image(img1);
  const colmap::Image &I2 = point_recon.Image(img2);

  FlatHashSet<colmap::point3D_t> ids1;

  const line2D_t line_id1 = static_cast<line2D_t>(node1.second);
  const std::vector<point2D_t> neighbors_pts1 =
      W1.GetNeighboringPointIds(line_id1);
  for (const point2D_t p2d_idx1 : neighbors_pts1) {
    if (p2d_idx1 >= I1.NumPoints2D())
      continue;

    const colmap::Point2D &p2d1 = I1.Point2D(p2d_idx1);
    const colmap::point3D_t p3d_id = p2d1.point3D_id;
    if (p3d_id == colmap::kInvalidPoint3DId)
      continue;

    ids1.insert(p3d_id);
  }

  if (ids1.empty())
    return;

  const line2D_t line_id2 = static_cast<line2D_t>(node2.second);
  const std::vector<point2D_t> neighbors_pts2 =
      W2.GetNeighboringPointIds(line_id2);
  for (const point2D_t p2d_idx2 : neighbors_pts2) {
    if (p2d_idx2 >= I2.NumPoints2D())
      continue;

    const colmap::Point2D &p2d2 = I2.Point2D(p2d_idx2);
    const colmap::point3D_t p3d_id = p2d2.point3D_id;
    if (p3d_id == colmap::kInvalidPoint3DId)
      continue;

    if (ids1.find(p3d_id) == ids1.end())
      continue;

    const colmap::Point3D &P3D = point_recon.Point3D(p3d_id);
    points3d.push_back(P3D.xyz);
  }
}

bool GetVPDirectionForLine(const StructureReconstruction &srec,
                           const Structure2d &S, const Node2d &node,
                           V3D &direction) {
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  const colmap::image_t img_id = node.first;
  const colmap::Image &image = point_recon.Image(img_id);

  const line2D_t line_id = static_cast<line2D_t>(node.second);

  const size_t num_groups = S.NumGroups();
  for (size_t gi = 0; gi < num_groups; ++gi) {
    const Group2d &G = S.Group(gi);
    if (G.type != GroupType::VP)
      continue;

    const std::vector<AssociatedFeature2d> &lines = G.lines;
    bool is_member = false;
    for (const AssociatedFeature2d &a : lines) {
      if (static_cast<line2D_t>(a.idx) == line_id) {
        is_member = true;
        break;
      }
    }
    if (!is_member)
      continue;

    const std::vector<double> &params = G.GetParams();
    if (params.size() >= 3) {
      // Interpret as 3D direction directly.
      direction = V3D(params[0], params[1], params[2]).normalized();
      return true;
    } else if (params.size() == 2) {
      // Interpret as 2D vanishing point in pixels; convert to 3D direction.
      const colmap::Camera &cam = point_recon.Camera(image.CameraId());
      Eigen::Vector3d vp_h(params[0], params[1], 1.0);
      Eigen::Matrix3d Kinv = cam.CalibrationMatrix().inverse();
      Eigen::Vector3d dir_cam = Kinv * vp_h;
      direction = dir_cam.normalized();
      return true;
    }
  }

  return false;
}

// ---------------- Proposal generation ---------------------------------------
void AddPointBasedProposalsForMatch(const Node2d &node, const Line2d &l1,
                                    const colmap::Image &image1,
                                    const Node2d &ng_node, const Line2d &l2,
                                    const colmap::Image &image2,
                                    const StructureReconstruction &srec,
                                    const LineTriangulationParams &params,
                                    ProposalList &proposals) {
  std::vector<V3D> points3d;
  CollectShared3DPoints(node, ng_node, srec, points3d);

  // many-point triangulation if >=2 shared points
  if (points3d.size() >= 2) {
    V3D center = V3D::Zero();
    for (const V3D &p : points3d) {
      center += p;
    }
    center /= static_cast<double>(points3d.size());

    Eigen::MatrixXd X(points3d.size(), 3);
    for (size_t i = 0; i < points3d.size(); ++i) {
      X.row(i) = (points3d[i] - center).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinV);
    const V3D direction = svd.matrixV().col(0).normalized();

    InfiniteLine3d inf_line(center, direction);
    auto res = estimators::triangulation::TriangulateLineWithInfiniteLine3d(
        l1, image1, inf_line);

    if (res.has_value()) {
      Line3d line = res.value();
      line.uncertainty =
          std::min(line.ComputeUncertainty(image1, params.var2d),
                   line.ComputeUncertainty(image2, params.var2d));
      proposals.emplace_back(line, ng_node, image1.CamFromWorld());
    }
  }

  // one-point triangulation if at least 1 shared point
  if (!points3d.empty()) {
    for (const V3D &p : points3d) {
      auto res = estimators::triangulation::TriangulateLineWithOnePoint(
          l1, image1, l2, image2, p);

      if (res.has_value()) {
        Line3d line = res.value();
        line.uncertainty =
            std::min(line.ComputeUncertainty(image1, params.var2d),
                     line.ComputeUncertainty(image2, params.var2d));
        proposals.emplace_back(line, ng_node, image1.CamFromWorld());
      }
    }
  }
}

void AddVPBasedProposalsForMatch(const Node2d &node, const Line2d &l1,
                                 const colmap::Image &image1,
                                 const Node2d &ng_node, const Line2d &l2,
                                 const colmap::Image &image2,
                                 const Structure2d &S1, const Structure2d &S2,
                                 const StructureReconstruction &srec,
                                 const LineTriangulationParams &params,
                                 ProposalList &proposals) {
  V3D vp_dir;

  // VP from reference line (in image1)
  if (GetVPDirectionForLine(srec, S1, node, vp_dir)) {
    auto res = estimators::triangulation::TriangulateLineWithDirection(
        l1, image1, l2, image2, vp_dir);
    if (res.has_value()) {
      Line3d line = res.value();
      line.uncertainty =
          std::min(line.ComputeUncertainty(image1, params.var2d),
                   line.ComputeUncertainty(image2, params.var2d));
      proposals.emplace_back(line, ng_node, image1.CamFromWorld());
    }
  }

  // VP from neighbor line (in image2)
  if (GetVPDirectionForLine(srec, S2, ng_node, vp_dir)) {
    auto res = estimators::triangulation::TriangulateLineWithDirection(
        l1, image1, l2, image2, vp_dir);
    if (res.has_value()) {
      Line3d line = res.value();
      line.uncertainty =
          std::min(line.ComputeUncertainty(image1, params.var2d),
                   line.ComputeUncertainty(image2, params.var2d));
      proposals.emplace_back(line, ng_node, image1.CamFromWorld());
    }
  }
}

void AddEpipolarProposalsForMatch(const Line2d &l1, const colmap::Image &image1,
                                  const Line2d &l2, const colmap::Image &image2,
                                  const Node2d &ng_node,
                                  const LineTriangulationParams &params,
                                  ProposalList &proposals) {
  const double IoU =
      estimators::triangulation::ComputeEpipolarIoU(l1, image1, l2, image2);
  if (IoU < params.IoU_threshold) {
    return;
  }

  auto res = estimators::triangulation::TriangulateLine(l1, image1, l2, image2);
  if (!res.has_value()) {
    return;
  }
  Line3d line = res.value();

  const double sens1 = line.Sensitivity(image1);
  const double sens2 = line.Sensitivity(image2);
  if (sens1 > params.sensitivity_threshold &&
      sens2 > params.sensitivity_threshold) {
    return;
  }

  line.uncertainty = std::min(line.ComputeUncertainty(image1, params.var2d),
                              line.ComputeUncertainty(image2, params.var2d));
  proposals.emplace_back(line, ng_node, image1.CamFromWorld());
}

// ---------------- Scoring ----------------------------------------------------

double ComputeTotalSupportScore(std::size_t target_idx,
                                const ProposalList &proposals,
                                const StructureReconstruction &srec,
                                const LineLinker &linker, double scale_inv_th) {
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  const Line3d &l1_3d = proposals[target_idx].line;

  // Each supporting image contributes at most once (best hypothesis).
  FlatHashMap<colmap::image_t, double> score_table;

  for (std::size_t j = 0; j < proposals.size(); ++j) {
    if (j == target_idx) {
      continue;
    }

    const Line3d &l2_3d = proposals[j].line;
    const Node2d &ng_node = proposals[j].supporting_node;
    const colmap::image_t ng_img_id = ng_node.first;
    const feature2D_t ng_feat_id = ng_node.second;
    const line2D_t ng_line_id = static_cast<line2D_t>(ng_feat_id);

    double score3d = linker.ComputeScore3d(l1_3d, l2_3d);
    if (score3d <= 0.0) {
      continue;
    }

    double scale_inv_err = EndpointsScaleinvDistanceOneway(
        l2_3d, l1_3d, proposals[target_idx].depth_start,
        proposals[target_idx].depth_end);
    double scale_inv_score =
        ExpScore(scale_inv_err, scale_inv_th * GetMultiplierFromThreshold(0.5));
    scale_inv_score = scale_inv_score >= 0.5 ? scale_inv_score : 0.0;
    score3d = std::min(score3d, scale_inv_score);
    if (score3d <= 0.0) {
      continue;
    }

    const Structure2d &S2 = srec.Structure2d(ng_img_id);
    const Line2d &l2_2d = S2.Line(ng_line_id);

    const colmap::Image &image2 = point_recon.Image(ng_img_id);
    auto res = l1_3d.Projection(image2);
    if (!res.has_value()) {
      continue;
    }
    const Line2d proj_1_on_2 = res.value();

    double score2d = linker.ComputeScore2d(proj_1_on_2, l2_2d);
    if (score2d <= 0.0) {
      continue;
    }

    const double score = std::min(score3d, score2d);

    auto map_it = score_table.find(ng_img_id);
    if (map_it == score_table.end()) {
      score_table.insert(std::make_pair(ng_img_id, score));
    } else if (score > map_it->second) {
      map_it->second = score;
    }
  }

  double total_score = 0.0;
  for (const auto &kv : score_table) {
    total_score += kv.second;
  }
  return total_score;
}

std::pair<int, double> SelectBestSupportedProposal(
    const ProposalList &proposals, const StructureReconstruction &srec,
    const LineLinker &linker, double scale_inv_th, double fullscore_th) {
  if (proposals.empty()) {
    return {-1, 0.0};
  }

  int best_idx = -1;
  double best_score = -std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < proposals.size(); ++i) {
    double score =
        ComputeTotalSupportScore(i, proposals, srec, linker, scale_inv_th);
    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }

  if (best_idx == -1 || best_score < fullscore_th) {
    return {-1, best_score};
  }
  return {best_idx, best_score};
}

// ---------------- Edges + Clusters for creation -----------------------------

std::vector<EdgeRecord> BuildLineSimilarityEdges(
    const std::vector<Node2d> &nodes,
    const Node2dMap<LineTriangulationProposal> &best_tris,
    StructureReconstruction &srec, const LineLinker &linker_clustering,
    const colmap::CorrespondenceGraph &corr_graph, int num_threads,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  // Build a set for quick lookup of which nodes have triangulations
  Node2dSet node_set(nodes.begin(), nodes.end());

  // Only nodes that already carry a triangulation can form an edge, so
  // exhaustive candidates are drawn from these rather than from every line.
  std::unordered_map<colmap::image_t, std::vector<feature2D_t>> nodes_by_image;
  if (!exhaustive_match_neighbors.empty()) {
    for (const Node2d &n : nodes) {
      nodes_by_image[n.first].push_back(n.second);
    }
  }

  // Collect edges using correspondence graph constraints
  std::vector<EdgeRecord> edges;
  std::mutex edges_mutex;
  const size_t n_nodes = nodes.size();

  {
    colmap::ThreadPool pool(colmap::GetEffectiveNumThreads(num_threads));
    for (size_t i = 0; i < n_nodes; ++i) {
      pool.AddTask([&, i]() {
        const Node2d &ni = nodes[i];
        const auto it_i = best_tris.find(ni);
        if (it_i == best_tris.end()) {
          return;
        }
        const LineTriangulationProposal &tri_i = it_i->second;

        const Line3d &line_i_3d = tri_i.line;
        const colmap::image_t img_i = ni.first;
        const feature2D_t feat_i = ni.second;
        const line2D_t line_i_id = static_cast<line2D_t>(feat_i);

        const Structure2d &Si = srec.Structure2d(img_i);
        const Line2d &line_i_2d = Si.Line(line_i_id);
        const colmap::Image &Im_i = point_recon.Image(img_i);

        std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
        if (!exhaustive_match_neighbors.empty()) {
          const auto it_ng = exhaustive_match_neighbors.find(img_i);
          if (it_ng != exhaustive_match_neighbors.end()) {
            for (const colmap::image_t ng_img_id : it_ng->second) {
              const auto it_nb = nodes_by_image.find(ng_img_id);
              if (it_nb == nodes_by_image.end()) {
                continue;
              }
              for (const feature2D_t f : it_nb->second) {
                corrs.emplace_back(ng_img_id,
                                   static_cast<colmap::point2D_t>(f));
              }
            }
          }
        } else {
          // Correspondences for this node from the correspondence graph
          corr_graph.ExtractCorrespondences(
              img_i, static_cast<colmap::point2D_t>(feat_i), &corrs);
        }

        std::vector<EdgeRecord> local_edges;

        for (const auto &corr : corrs) {
          const Node2d nj(corr.image_id,
                          static_cast<feature2D_t>(corr.point2D_idx));

          // Only process if nj has a triangulation and we haven't processed
          // this pair yet (use ordering to avoid duplicates)
          if (node_set.find(nj) == node_set.end()) {
            continue;
          }
          // Only process pairs where ni < nj to avoid duplicates
          if (!(ni < nj)) {
            continue;
          }

          const auto it_j = best_tris.find(nj);
          if (it_j == best_tris.end()) {
            continue;
          }
          const LineTriangulationProposal &tri_j = it_j->second;

          const Line3d &line_j_3d = tri_j.line;
          const colmap::image_t img_j = nj.first;
          const feature2D_t feat_j = nj.second;
          const line2D_t line_j_id = static_cast<line2D_t>(feat_j);

          const Structure2d &Sj = srec.Structure2d(img_j);
          const Line2d &line_j_2d = Sj.Line(line_j_id);
          const colmap::Image &Im_j = point_recon.Image(img_j);

          // 3D similarity.
          double score_3d =
              linker_clustering.ComputeScore3d(line_i_3d, line_j_3d);
          if (score_3d <= 0.0) {
            continue;
          }

          // 2D consistency (both directions).
          auto res_i_on_j = line_i_3d.Projection(Im_j);
          if (!res_i_on_j.has_value()) {
            continue;
          }
          const Line2d proj_i_on_j = res_i_on_j.value();
          auto res_j_on_i = line_j_3d.Projection(Im_i);
          if (!res_j_on_i.has_value()) {
            continue;
          }
          const Line2d proj_j_on_i = res_j_on_i.value();

          double score2d_i2j =
              linker_clustering.ComputeScore2d(proj_i_on_j, line_j_2d);
          if (score2d_i2j <= 0.0) {
            continue;
          }
          double score2d_j2i =
              linker_clustering.ComputeScore2d(proj_j_on_i, line_i_2d);
          if (score2d_j2i <= 0.0) {
            continue;
          }

          const double score_2d = std::min(score2d_i2j, score2d_j2i);
          const double score = std::min(score_3d, score_2d);
          if (score <= 0.0) {
            continue;
          }

          EdgeRecord e;
          e.weight = score;
          e.n1 = ni;
          e.n2 = nj;
          local_edges.push_back(e);
        }

        if (!local_edges.empty()) {
          std::lock_guard<std::mutex> lock(edges_mutex);
          edges.insert(edges.end(), local_edges.begin(), local_edges.end());
        }
      });
    }
    pool.Wait();
  }

  // Sort edges by descending weight (greedy merging).
  std::sort(edges.begin(), edges.end(),
            [](const EdgeRecord &a, const EdgeRecord &b) {
              return a.weight > b.weight;
            });

  return edges;
}

Node2dMap<std::vector<Node2d>>
BuildLineClusters(const std::vector<Node2d> &nodes,
                  const std::vector<EdgeRecord> &edges) {
  Node2dMap<std::vector<Node2d>> clusters;
  clusters.reserve(nodes.size());

  colmap::UnionFind<Node2d, colmap::PairHash> uf;
  uf.Reserve(nodes.size());

  for (const EdgeRecord &e : edges) {
    if (e.weight <= 0.0) {
      break;
    }
    uf.Union(e.n1, e.n2);
  }

  for (const Node2d &n : nodes) {
    const std::optional<Node2d> root_opt = uf.FindIfExists(n);
    if (!root_opt.has_value()) {
      // Node never participated in any union => singleton => ignore.
      continue;
    }
    clusters[*root_opt].push_back(n);
  }

  return clusters;
}

} // namespace line_triangulation
} // namespace internal
} // namespace limap
