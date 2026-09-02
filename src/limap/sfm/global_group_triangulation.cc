#include "limap/sfm/global_group_triangulation.h"

#include "limap/estimators/group3d/group3d_estimator.h"
#include "limap/sfm/group_verification.h"

namespace limap {

GlobalGroupTriangulationController::GlobalGroupTriangulationController(
    const GlobalGroupTriangulationOptions &options,
    const std::shared_ptr<HolisticReconstruction> &recon,
    const colmap::CorrespondenceGraph &group_corr_graph)
    : options_(options), recon_(recon), corr_graph_(group_corr_graph) {
  CHECK(options_.Check());
  uf_.Reserve(1024);

  // Initialize valid group types set for fast lookup
  auto types = options_.GetEffectiveGroupTypes();
  valid_group_types_.insert(types.begin(), types.end());
}

bool GlobalGroupTriangulationController::IsValidGroupType(
    GroupType type) const {
  return valid_group_types_.count(type) > 0;
}

void GlobalGroupTriangulationController::Run() {
  if (!CheckIfStopped())
    BuildConnectivity();
  if (!CheckIfStopped())
    CreateGroup3DObjects();
}

void GlobalGroupTriangulationController::Extract3DPoints(
    const Node2d &g, std::vector<colmap::point3D_t> &pts) const {
  pts.clear();

  const auto &S = recon_->StructureRecon().Structure2d(g.first);
  const auto &G = S.Group(g.second);

  const auto &img = recon_->StructureRecon().PointRecon().Image(g.first);

  pts.reserve(G.points.size());

  for (const auto &af : G.points) {
    auto feat = af.idx;
    if (feat >= img.NumPoints2D())
      continue;

    const auto &p2d = img.Point2D(feat);
    auto pid = p2d.point3D_id;
    if (pid != colmap::kInvalidPoint3DId)
      pts.push_back(pid);
  }

  std::sort(pts.begin(), pts.end());
}

void GlobalGroupTriangulationController::Extract3DLines(
    const Node2d &g, std::vector<line3D_t> &lines) const {
  lines.clear();

  const auto &S = recon_->StructureRecon().Structure2d(g.first);
  const auto &G = S.Group(g.second);

  lines.reserve(G.lines.size());

  for (const auto &af : G.lines) {
    auto feat = af.idx;
    if (feat >= S.NumLines())
      continue;

    const auto &l2d = S.Line(feat);
    auto lid = l2d.line3D_id;
    if (lid != kInvalidLine3dId)
      lines.push_back(lid);
  }

  std::sort(lines.begin(), lines.end());
}

void GlobalGroupTriangulationController::BuildConnectivity() {
  const auto &structures2d = recon_->StructureRecon().Structures2d();
  std::vector<colmap::point3D_t> pts1, pts2;
  std::vector<line3D_t> lines1, lines2;

  for (const auto &kv : structures2d) {
    colmap::image_t img1 = kv.first;
    const Structure2d &S1 = kv.second;

    for (size_t g1 = 0; g1 < S1.NumGroups(); g1++) {
      const Group2d &G1 = S1.Group(g1);
      if (!IsValidGroupType(G1.type))
        continue;

      Node2d k1(img1, g1);
      Extract3DPoints(k1, pts1);
      Extract3DLines(k1, lines1);
      if (pts1.empty() && lines1.empty())
        continue;

      std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
      corr_graph_.ExtractCorrespondences(img1, (colmap::point2D_t)g1, &corrs);

      for (const auto &c : corrs) {
        Node2d k2(c.image_id, c.point2D_idx);

        const auto &S2 = structures2d.at(c.image_id);
        if (k2.second >= S2.NumGroups())
          continue;

        const Group2d &G2 = S2.Group(k2.second);
        // Only union groups of the same type
        if (G2.type != G1.type)
          continue;

        Extract3DPoints(k2, pts2);
        Extract3DLines(k2, lines2);
        if (pts2.empty() && lines2.empty())
          continue;

        // Count shared 3D points
        int shared_points = 0;
        size_t i = 0, j = 0;
        while (i < pts1.size() && j < pts2.size()) {
          if (pts1[i] == pts2[j]) {
            shared_points++;
            i++;
            j++;
          } else if (pts1[i] < pts2[j]) {
            i++;
          } else {
            j++;
          }
        }

        // Count shared 3D lines
        int shared_lines = 0;
        i = 0;
        j = 0;
        while (i < lines1.size() && j < lines2.size()) {
          if (lines1[i] == lines2[j]) {
            shared_lines++;
            i++;
            j++;
          } else if (lines1[i] < lines2[j]) {
            i++;
          } else {
            j++;
          }
        }

        // Check if total shared features meet threshold
        int total_shared = shared_points + shared_lines;
        if (total_shared >= options_.min_shared_features_for_group_connection) {
          uf_.Union(k1, k2);
        }
      }
    }
  }
}

void GlobalGroupTriangulationController::CreateGroup3DObjects() {
  auto &srec = recon_->StructureRecon();
  auto &groups3D = srec.Groups3D();
  groups3D.clear();

  const auto &structures2d = srec.Structures2d();

  // root → members
  Node2dMap<std::vector<Node2d>> clusters;

  for (const auto &kv : structures2d) {
    colmap::image_t img = kv.first;
    const Structure2d &S = kv.second;

    for (size_t gi = 0; gi < S.NumGroups(); gi++) {
      Group2d &G = srec.Structure2d(img).Group(gi);

      if (!IsValidGroupType(G.type)) {
        G.group3D_id = kInvalidGroup3dId;
        continue;
      }

      Node2d gk(img, gi);

      auto root_opt = uf_.FindIfExists(gk);
      if (!root_opt) {
        G.group3D_id = kInvalidGroup3dId;
        continue;
      }

      clusters[*root_opt].push_back(gk);
    }
  }

  group3D_t next_gid = 0;
  std::vector<colmap::point3D_t> pts;
  std::vector<line3D_t> lines;

  for (auto &kv : clusters) {
    const auto &members = kv.second;

    if (members.size() < options_.min_cluster_size) {
      for (auto &gk : members) {
        srec.Structure2d(gk.first).Group(gk.second).group3D_id =
            kInvalidGroup3dId;
      }
      continue;
    }

    group3D_t gid = next_gid++;

    // Get group type from the first member (all members in a cluster have the
    // same type)
    const auto &first_member = members.front();
    GroupType cluster_type =
        srec.Structure2d(first_member.first).Group(first_member.second).type;

    Group3dWithActiveLabels group3d{Group3d(cluster_type)};

    // Fill track + assign 3D id
    for (auto &gk : members) {
      srec.Structure2d(gk.first).Group(gk.second).group3D_id = gid;
      group3d.track.AddElement(
          colmap::TrackElement(gk.first, (colmap::point2D_t)gk.second));
    }

    // Build 3D point associations
    FlatHashMap<colmap::point3D_t, int> point_support;

    for (auto &gk : members) {
      Extract3DPoints(gk, pts);
      for (auto pid : pts) {
        point_support[pid]++;
      }
    }

    for (auto &p : point_support) {
      if (p.second >= options_.min_num_supports_for_association) {
        group3d.AddPoint(AssociatedFeature3d(p.first));
      }
    }

    // Build 3D line associations
    FlatHashMap<line3D_t, int> line_support;

    for (auto &gk : members) {
      Extract3DLines(gk, lines);
      for (auto lid : lines) {
        line_support[lid]++;
      }
    }

    for (auto &l : line_support) {
      if (l.second >= options_.min_num_supports_for_association) {
        group3d.AddLine(AssociatedFeature3d(l.first));
      }
    }

    // Extract 3D coordinates and filter out non-existent features.
    // This keeps group3d.points/lines consistent (no stale references)
    // and keeps lines3d_vec aligned with group3d.lines (required by
    // VerifyVPGroup which uses index-based correspondence).
    std::vector<V3D> points3d_vec;
    std::vector<AssociatedFeature3d> valid_points;
    points3d_vec.reserve(group3d.points.size());
    valid_points.reserve(group3d.points.size());
    for (const auto &af : group3d.points) {
      auto pid = af.idx;
      if (!recon_->PointRecon().ExistsPoint3D(pid)) {
        continue;
      }
      const auto &p3d = recon_->PointRecon().Point3D(pid);
      points3d_vec.push_back(p3d.xyz);
      valid_points.push_back(af);
    }
    group3d.points = std::move(valid_points);

    std::vector<Line3d> lines3d_vec;
    std::vector<AssociatedFeature3d> valid_lines;
    lines3d_vec.reserve(group3d.lines.size());
    valid_lines.reserve(group3d.lines.size());
    for (const auto &af : group3d.lines) {
      auto lid = af.idx;
      if (srec.Lines3D().count(lid) == 0) {
        continue;
      }
      lines3d_vec.push_back(srec.Line(lid));
      valid_lines.push_back(af);
    }
    group3d.lines = std::move(valid_lines);

    // Initialize 3D parameters from geometric features
    auto params = estimators::group3d::InitializeGroupParams(
        cluster_type, points3d_vec, lines3d_vec,
        options_.GetInitRansacThresh(cluster_type));
    if (params.has_value()) {
      group3d.SetParams(params.value());

      // Verify group hypothesis
      if (options_.enable_verification) {
        GroupVerificationOptions verify_opts;
        verify_opts.inlier_threshold =
            options_.GetVerificationThresh(cluster_type);
        verify_opts.default_vp_threshold =
            options_.verification_default_vp_threshold;
        verify_opts.default_reproj_threshold =
            options_.verification_default_reproj_threshold;
        verify_opts.min_num_inliers = options_.verification_min_num_inliers;
        verify_opts.min_inlier_ratio = options_.verification_min_inlier_ratio;
        verify_opts.obs_inlier_ratio = options_.verification_obs_inlier_ratio;
        verify_opts.filter_outliers = options_.verification_filter_outliers;

        auto stats = VerifyGroup(group3d, *recon_, points3d_vec, lines3d_vec,
                                 verify_opts);

        if (!stats.passed) {
          // Reset group3D_id for cluster members
          for (auto &gk : members) {
            srec.Structure2d(gk.first).Group(gk.second).group3D_id =
                kInvalidGroup3dId;
          }
          continue; // Skip adding this group
        }
      }

    } else {
      LOG(WARNING) << "Failed to initialize params for Group3D " << gid;
      // Note: we still add the group even if initialization fails
      // The params will be left uninitialized (empty or default)
    }

    groups3D[gid] = group3d;
  }
}

} // namespace limap
