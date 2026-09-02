#include "limap/sfm/global_line_triangulation.h"

#include <colmap/scene/image.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>

#include <mutex>

#include <colmap/util/threading.h>
#include <thirdparty/progressbar.hpp>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/scene/structure2d.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/sfm/line_triangulation_utils.h"
#include "limap/sfm/structure_observation_manager.h"

namespace limap {

bool GlobalLineTriangulationOptions::Check() const {
  CHECK_OPTION_GT(wireframe2d_th, 0.0);
  CHECK_OPTION_GE(min_length_2d, 0.0);
  CHECK_OPTION_GE(IoU_threshold, 0.0);
  CHECK_OPTION_GE(sensitivity_threshold, 0.0);
  CHECK_OPTION_GT(var2d, 0.0);
  CHECK_OPTION_GE(fullscore_th, 0.0);
  CHECK_OPTION_GE(num_outliers_aggregator, 0);
  CHECK_OPTION_GE(filtering2d_th_angular_2d, 0.0);
  CHECK_OPTION_GE(filtering2d_th_perp_2d, 0.0);
  CHECK_OPTION_GE(filtering2d_th_sv_angular_3d, 0.0);
  CHECK_OPTION_GE(filtering2d_th_overlap, 0.0);
  CHECK_OPTION_GT(min_visible_views, 0);
  return true;
}

namespace {
Line3d TakeBestTriangulatedLine(const std::vector<Line3d> &lines,
                                const std::vector<double> &scores) {
  const int n = static_cast<int>(lines.size());
  THROW_CHECK_GT(n, 0);

  double best_score = -std::numeric_limits<double>::infinity();
  int best_idx = -1;
  double min_unc = std::numeric_limits<double>::max();

  for (int i = 0; i < n; ++i) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
    min_unc = std::min(min_unc, lines[i].uncertainty);
  }

  Line3d L = lines[best_idx];
  L.uncertainty = min_unc;
  return L;
}

} // namespace

GlobalLineTriangulationController::GlobalLineTriangulationController(
    const GlobalLineTriangulationOptions &options,
    const std::shared_ptr<HolisticReconstruction> &recon,
    const colmap::CorrespondenceGraph &line_corr_graph,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors)
    : options_(options), recon_(recon), corr_graph_(line_corr_graph),
      exhaustive_match_neighbors_(exhaustive_match_neighbors) {
  CHECK(options_.Check());
}

void GlobalLineTriangulationController::Run() {
  LOG(INFO) << "Start global line triangulation...";
  best_tris_.clear();
  StructureReconstruction &srec = recon_->StructureRecon();
  const NodeHashMap<colmap::image_t, Structure2d> &structures2d =
      srec.Structures2d();

  // Triangulate & score each 2D line independently.
  // Collect all nodes first for parallel processing
  std::vector<Node2d> all_nodes;
  for (const auto &kv : structures2d) {
    const colmap::image_t img_id = kv.first;
    const Structure2d &S = kv.second;
    const size_t num_lines = S.NumLines();
    for (size_t li = 0; li < num_lines; ++li) {
      all_nodes.emplace_back(img_id, static_cast<feature2D_t>(li));
    }
  }

  LOG(INFO) << "Triangulating " << all_nodes.size() << " 2D lines across "
            << structures2d.size() << " images...";

  // Show progress bar only for large inputs
  const bool show_tri_progress = all_nodes.size() >= 1000;
  progressbar tri_bar(all_nodes.size(), show_tri_progress, show_tri_progress);

  {
    colmap::ThreadPool pool(
        colmap::GetEffectiveNumThreads(options_.num_threads));
    for (size_t i = 0; i < all_nodes.size(); ++i) {
      pool.AddTask([this, &all_nodes, &tri_bar, show_tri_progress, i]() {
        TriangulateAndScoreNode(all_nodes[i]);
        if (show_tri_progress) {
          tri_bar.update();
        }
      });
    }
    pool.Wait();
  }

  // Merge triangulated lines into 3D line tracks.
  if (!CheckIfStopped()) {
    MergeTriangulationsIntoTracks();
  }
  LOG(INFO) << "Merge " << srec.NumLines3D()
            << " 3D lines from triangulation proposals.";

  // Optional remerging step
  if (options_.enable_remerge && !CheckIfStopped()) {
    StructureObservationManager obs_mgr(recon_->StructureRecon());
    obs_mgr.FilterLines3dByReprojection(options_.filtering2d_th_angular_2d,
                                        options_.filtering2d_th_perp_2d,
                                        options_.num_outliers_aggregator);
    RemergeLines3D();
  }

  // Post filtering the 3D line tracks.
  if (!CheckIfStopped()) {
    StructureObservationManager obs_mgr(recon_->StructureRecon());
    obs_mgr.FilterLines3dByReprojection(options_.filtering2d_th_angular_2d,
                                        options_.filtering2d_th_perp_2d,
                                        options_.num_outliers_aggregator);

    obs_mgr.FilterLines3dBySensitivity(options_.filtering2d_th_sv_angular_3d,
                                       options_.filtering2d_th_sv_num_supports);

    obs_mgr.FilterLines3dByOverlap(
        options_.filtering2d_th_overlap,
        options_.filtering2d_th_overlap_num_supports);

    obs_mgr.FilterLines3dByMinVisibleViews(options_.min_visible_views);
  }

  // Logging
  LOG(INFO) << "Triangulate a total of " << srec.NumLines3D() << " 3D lines.";
  std::vector<int> track_len_ths = {2, 3, 5, 10};
  std::vector<int> num_lines_with_len_geq(track_len_ths.size(), 0);
  for (const auto &kv : srec.Lines3D()) {
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

void GlobalLineTriangulationController::TriangulateAndScoreNode(
    const Node2d &node) {
  StructureReconstruction &srec = recon_->StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const colmap::image_t img_id = node.first;
  const feature2D_t feat_id = node.second;
  const line2D_t line_id = static_cast<line2D_t>(feat_id);

  const Structure2d &S = srec.Structure2d(img_id);
  if (line_id >= static_cast<line2D_t>(S.NumLines())) {
    return;
  }

  const Line2d &l1 = S.Line(line_id);
  if (l1.Length() <= options_.min_length_2d) {
    return;
  }

  const colmap::Image &image1 = point_recon.Image(img_id);

  const NodeHashMap<colmap::image_t, Structure2d> &structures2d =
      srec.Structures2d();

  std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
  if (!exhaustive_match_neighbors_.empty()) {
    // Pair this line with every line in each neighbor.
    const auto it_ng = exhaustive_match_neighbors_.find(img_id);
    if (it_ng != exhaustive_match_neighbors_.end()) {
      for (const colmap::image_t ng_img_id : it_ng->second) {
        const auto it_s2 = structures2d.find(ng_img_id);
        if (it_s2 == structures2d.end()) {
          continue;
        }
        const size_t num_ng_lines = it_s2->second.NumLines();
        corrs.reserve(corrs.size() + num_ng_lines);
        for (size_t ng_line_id = 0; ng_line_id < num_ng_lines; ++ng_line_id) {
          corrs.emplace_back(ng_img_id,
                             static_cast<colmap::point2D_t>(ng_line_id));
        }
      }
    }
  } else {
    // Gather all connected nodes via the correspondence graph.
    corr_graph_.ExtractCorrespondences(
        img_id, static_cast<colmap::point2D_t>(feat_id), &corrs);
  }

  if (corrs.empty()) {
    return;
  }

  internal::line_triangulation::ProposalList proposals;
  proposals.reserve(corrs.size());

  for (const auto &corr : corrs) {
    const colmap::image_t ng_img_id = corr.image_id;
    const feature2D_t ng_feat_id = static_cast<feature2D_t>(corr.point2D_idx);
    const line2D_t ng_line_id = static_cast<line2D_t>(ng_feat_id);
    Node2d ng_node(ng_img_id, ng_feat_id);

    auto it_s2 = structures2d.find(ng_img_id);
    if (it_s2 == structures2d.end()) {
      continue;
    }
    const Structure2d &S2 = it_s2->second;
    THROW_CHECK_LT(ng_line_id, static_cast<line2D_t>(S2.NumLines()));

    const Line2d &l2 = S2.Line(ng_line_id);
    if (l2.Length() <= options_.min_length_2d) {
      continue;
    }

    const colmap::Image &image2 = point_recon.Image(ng_img_id);

    const LineTriangulationParams tri_params =
        options_.GetTriangulationParams();

    // Step 1: point-based triangulation (many points / one point)
    if (options_.use_point_assisted_triangulation) {
      internal::line_triangulation::AddPointBasedProposalsForMatch(
          node, l1, image1, ng_node, l2, image2, srec, tri_params, proposals);
    }

    // Step 2: triangulation with vanishing points (optional)
    if (options_.use_vp_assisted_triangulation) {
      internal::line_triangulation::AddVPBasedProposalsForMatch(
          node, l1, image1, ng_node, l2, image2, S, S2, srec, tri_params,
          proposals);
    }

    // Step 3: algebraic line triangulation with epipolar IoU test
    internal::line_triangulation::AddEpipolarProposalsForMatch(
        l1, image1, l2, image2, ng_node, tri_params, proposals);
  }

  if (proposals.empty()) {
    return;
  }

  // Scoring: each candidate gathers support from other views.
  LineLinker3dOptions linker3d_scoring_options = options_.linker3d_options;
  linker3d_scoring_options.SetToSharedParentScoring();
  LineLinker linker_scoring(options_.linker2d_options,
                            linker3d_scoring_options);
  const auto best = internal::line_triangulation::SelectBestSupportedProposal(
      proposals, srec, linker_scoring, options_.scale_inv_th,
      options_.fullscore_th);

  const int best_idx = best.first;
  const double best_score = best.second;

  if (best_idx < 0) {
    return;
  }

  LineTriangulationProposal chosen =
      proposals[static_cast<std::size_t>(best_idx)];
  chosen.score = best_score;

  {
    std::lock_guard<std::mutex> lock(best_tris_mutex_);
    best_tris_[node] = chosen;
  }
}

void GlobalLineTriangulationController::MergeTriangulationsIntoTracks() {
  StructureReconstruction &srec = recon_->StructureRecon();
  auto &structures2d = srec.Structures2d();
  auto &lines3d = srec.Lines3D();

  // (0) Reset 3D lines and detach all 2D → 3D assignments
  lines3d.clear();
  for (auto &kv : structures2d) {
    Structure2d &S = kv.second;
    for (size_t i = 0; i < S.NumLines(); ++i) {
      S.Line(i).line3D_id = kInvalidLine3dId;
    }
  }

  if (best_tris_.empty()) {
    return;
  }

  // (1) Collect nodes that have a triangulation hypothesis
  std::vector<Node2d> nodes;
  nodes.reserve(best_tris_.size());
  for (const auto &kv : best_tris_) {
    nodes.push_back(kv.first);
  }

  // (2) Build clustering edges based on similarity of 3D lines
  LineLinker3dOptions linker3d_clustering_options = options_.linker3d_options;
  linker3d_clustering_options.SetToSpatialMerging();
  LineLinker linker_clustering(options_.linker2d_options,
                               linker3d_clustering_options);
  const auto edges = internal::line_triangulation::BuildLineSimilarityEdges(
      nodes, best_tris_, srec, linker_clustering, corr_graph_,
      options_.num_threads, exhaustive_match_neighbors_);

  // (3) Convert edges → clusters via Union-Find
  const auto clusters =
      internal::line_triangulation::BuildLineClusters(nodes, edges);

  // (4) Build final Line3D objects for each cluster
  line3D_t next_lid = 0;

  for (const auto &kv : clusters) {
    const std::vector<Node2d> &members = kv.second;

    // We only create a 3D line if there is ≥2 observations (matches old
    // behavior)
    if (members.size() < 2) {
      continue;
    }

    // Collect 3D hypotheses & scores
    std::vector<Line3d> lines_cluster;
    std::vector<double> scores_cluster;
    lines_cluster.reserve(members.size());
    scores_cluster.reserve(members.size());

    for (const Node2d &n : members) {
      const LineTriangulationProposal &tri = best_tris_.at(n);
      lines_cluster.push_back(tri.line);
      scores_cluster.push_back(tri.score);
    }

    Line3d line3d;

    // (4a) Small cluster → pick best scoring line
    //     (identical to old AggregateLine3dListTakeBest)
    if (lines_cluster.size() < options_.num_outliers_aggregator * 2) {
      line3d = TakeBestTriangulatedLine(lines_cluster, scores_cluster);
    }

    // (4b) Large cluster → geometric PCA aggregation
    //     (identical to old SVD-based behavior)
    else {
      line3d =
          AggregateLine3dList(lines_cluster, options_.num_outliers_aggregator);
    }

    // (4c) Fill the track (list of 2D references)
    for (const Node2d &n : members) {
      line3d.track.AddElement(colmap::TrackElement(
          n.first, static_cast<colmap::point2D_t>(n.second)));
    }

    const line3D_t lid = next_lid++;
    lines3d[lid] = line3d;

    // (4d) Assign line3D_id back to all 2D lines in the cluster
    for (const Node2d &n : members) {
      Structure2d &S = srec.Structure2d(n.first);
      const line2D_t lid2d = static_cast<line2D_t>(n.second);
      S.Line(lid2d).line3D_id = lid;
    }
  }
}

void GlobalLineTriangulationController::RemergeLines3D() {
  LOG(INFO) << "Start remerging 3D lines...";
  StructureReconstruction &srec = recon_->StructureRecon();
  auto &lines3d = srec.Lines3D();
  auto &structures2d = srec.Structures2d();

  if (lines3d.empty()) {
    LOG(INFO) << "No 3D lines to remerge.";
    return;
  }

  // Configure LineLinker3d for spatial merging
  LineLinker3dOptions linker3d_options = options_.linker3d_options;
  linker3d_options.SetToSpatialMerging();
  LineLinker3d linker3d(linker3d_options);

  // Convert to vector of IDs for iteration
  std::vector<line3D_t> line_ids;
  line_ids.reserve(lines3d.size());
  for (const auto &[lid, _] : lines3d) {
    line_ids.push_back(lid);
  }
  const size_t n_lines = line_ids.size();

  // Phase 1: Parallel edge detection (read-only, thread-safe)
  std::vector<std::pair<line3D_t, line3D_t>> edges;
  std::mutex edges_mutex;

  {
    colmap::ThreadPool pool(
        colmap::GetEffectiveNumThreads(options_.num_threads));
    for (size_t i = 0; i < n_lines; ++i) {
      pool.AddTask([&, i]() {
        std::vector<std::pair<line3D_t, line3D_t>> local_edges;
        const Line3d &l1 = lines3d.at(line_ids[i]);

        for (size_t j = i + 1; j < n_lines; ++j) {
          const Line3d &l2 = lines3d.at(line_ids[j]);
          if (linker3d.CheckConnection(l1, l2)) {
            local_edges.emplace_back(line_ids[i], line_ids[j]);
          }
        }

        if (!local_edges.empty()) {
          std::lock_guard<std::mutex> lock(edges_mutex);
          edges.insert(edges.end(), local_edges.begin(), local_edges.end());
        }
      });
    }
    pool.Wait();
  }

  // Phase 2: Sequential union (fast, O(edges * α(n)))
  colmap::UnionFind<line3D_t> uf;
  uf.Reserve(n_lines);

  for (const auto &[lid1, lid2] : edges) {
    uf.Union(lid1, lid2);
  }

  LOG(INFO) << "Found " << edges.size() << " similarity edges.";

  // Compute group labels
  FlatHashMap<line3D_t, size_t> root_to_group;
  FlatHashMap<line3D_t, size_t> line_to_group;
  size_t n_groups = 0;

  for (const auto &[lid, _] : lines3d) {
    line3D_t root = uf.Find(lid);
    if (root_to_group.find(root) == root_to_group.end()) {
      root_to_group[root] = n_groups++;
    }
    line_to_group[lid] = root_to_group[root];
  }

  // Build new line tracks for each group (only for groups with 2+ members)
  FlatHashMap<size_t, std::vector<Line3d>> lines_per_group;
  FlatHashMap<size_t, colmap::Track> tracks_per_group;

  for (const auto &[lid, line] : lines3d) {
    size_t group_id = line_to_group[lid];
    lines_per_group[group_id].push_back(line);
    for (const auto &elem : line.track.Elements()) {
      tracks_per_group[group_id].AddElement(elem);
    }
  }

  // Clear old 3D lines and reset 2D assignments
  for (auto &[img_id, structure] : structures2d) {
    for (size_t i = 0; i < structure.NumLines(); ++i) {
      structure.Line(i).line3D_id = kInvalidLine3dId;
    }
  }
  lines3d.clear();
  line3D_t next_lid = 0;

  for (auto &[group_id, lines_in_group] : lines_per_group) {
    // Aggregate geometry and create new line (even singletons, as they may have
    // many observations)
    Line3d merged_line =
        AggregateLine3dList(lines_in_group, options_.num_outliers_aggregator);
    merged_line.track = tracks_per_group[group_id];

    const line3D_t lid = next_lid++;
    lines3d[lid] = std::move(merged_line);

    // Update 2D -> 3D assignments
    for (const auto &elem : lines3d[lid].track.Elements()) {
      Structure2d &S = srec.Structure2d(elem.image_id);
      const line2D_t lid2d = static_cast<line2D_t>(elem.point2D_idx);
      S.Line(lid2d).line3D_id = lid;
    }
  }

  LOG(INFO) << "Remerging complete. New number of 3D lines: " << lines3d.size();
}

void GlobalLineTriangulationPipeline(
    const StructureDatabaseCache &structure_db_cache,
    const std::shared_ptr<HolisticReconstruction> &reconstruction,
    const GlobalLineTriangulationOptions &options,
    const estimators::PointLineBundleAdjustmentOptions *ba_options,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
  THROW_CHECK(options.Check());

  reconstruction->StructureRecon().Load(structure_db_cache);

  // Reset points and wireframes from point reconstruction (line-only
  // pipelines may not have correct points/wireframes in the structure DB).
  {
    auto &srec = reconstruction->StructureRecon();
    bool warned = false;
    for (const auto &[img_id, image] : reconstruction->PointRecon().Images()) {
      auto &s2d = srec.Structure2d(img_id);
      if (!warned && s2d.Wireframe().CountEdges() > 0) {
        LOG(INFO) << "Discarding wireframes from structure database; "
                     "recomputing from point reconstruction keypoints.";
        warned = true;
      }
      s2d.Wireframe().Clear();
      s2d.SetNumPoints(image.NumPoints2D());
    }
  }

  reconstruction->StructureRecon().InitializeAllWireframes(
      options.wireframe2d_th);

  auto structure_corr_graph = structure_db_cache.StructureCorrespondenceGraph();

  LOG(INFO) << "Running global line triangulation...";
  GlobalLineTriangulationController ctrl(options, reconstruction,
                                         structure_corr_graph->LineGraph(),
                                         exhaustive_match_neighbors);
  ctrl.Run();

  LOG(INFO) << "Global line triangulation complete: "
            << reconstruction->StructureRecon().NumLines3D() << " 3D lines";

  if (ba_options) {
    LOG(INFO) << "Running bundle adjustment...";
    // During triangulation we only refine structure, never cameras/rigs.
    auto opts = *ba_options;
    opts.refine_focal_length = false;
    opts.refine_principal_point = false;
    opts.refine_extra_params = false;
    opts.refine_sensor_from_rig = false;
    opts.refine_rig_from_world = false;

    estimators::PointLineBundleAdjustmentConfig ba_config;
    for (const auto &[image_id, image] :
         reconstruction->PointRecon().Images()) {
      if (image.HasPose())
        ba_config.AddImage(image_id);
    }
    auto adjuster = estimators::CreatePointLineBundleAdjuster(
        opts, std::move(ba_config), *reconstruction);
    auto ba_summary = adjuster->Solve();
    const auto &summary =
        static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary)
            .ceres_summary;
    LOG(INFO) << "Bundle adjustment completed: initial_cost="
              << summary.initial_cost << ", final_cost=" << summary.final_cost;
  }
}

} // namespace limap
