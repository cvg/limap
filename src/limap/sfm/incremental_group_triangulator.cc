#include "limap/sfm/incremental_group_triangulator.h"

#include <colmap/math/math.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>

#include "limap/estimators/bundle_adjustment/group_bundle_adjustment.h"
#include "limap/estimators/group3d/group3d_estimator.h"
#include "limap/scene/structure2d.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/sfm/group_verification.h"

namespace limap {

bool IncrementalGroupTriangulator::Options::Check() const {
  CHECK_OPTION_GE(max_transitivity, 0);
  CHECK_OPTION_GE(min_num_observations, 2);
  CHECK_OPTION_GE(min_num_supports_for_association, 1);
  CHECK_OPTION_GT(default_vp_threshold, 0.0);
  CHECK_OPTION_GT(default_reproj_threshold, 0.0);
  CHECK_OPTION_GE(verification_min_num_inliers, 1);
  CHECK_OPTION_GT(verification_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(verification_min_inlier_ratio, 1.0);
  CHECK_OPTION_GT(verification_obs_inlier_ratio, 0.0);
  CHECK_OPTION_LE(verification_obs_inlier_ratio, 1.0);
  CHECK_OPTION_GT(continue_max_reproj_error, 0.0);
  CHECK_OPTION_GT(continue_max_angle_error_vp, 0.0);
  CHECK_OPTION_GT(complete_max_reproj_error, 0.0);
  CHECK_OPTION_GT(complete_max_angle_error_vp, 0.0);
  CHECK_OPTION_GE(complete_max_transitivity, 0);
  CHECK_OPTION_GT(merge_max_reproj_error, 0.0);
  CHECK_OPTION_GT(merge_max_angle_error_vp, 0.0);
  CHECK_OPTION_GT(merge_max_error_ratio, 0.0);
  CHECK_OPTION_GT(merge_max_plane_normal_angle, 0.0);
  CHECK_OPTION_GE(association_filter_min_num_features, 0);
  CHECK_OPTION_GT(association_filter_reproj_threshold, 0.0);
  CHECK_OPTION_GE(surface_dist_mad_factor, 0.0);
  return true;
}

IncrementalGroupTriangulator::IncrementalGroupTriangulator(
    std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph,
    HolisticReconstruction &reconstruction,
    std::shared_ptr<StructureObservationManager> obs_manager)
    : structure_corr_graph_(std::move(structure_corr_graph)),
      reconstruction_(reconstruction), obs_manager_(std::move(obs_manager)) {
  if (!obs_manager_) {
    obs_manager_ = std::make_shared<StructureObservationManager>(
        reconstruction_.StructureRecon());
  }
}

size_t
IncrementalGroupTriangulator::TriangulateImage(const Options &options,
                                               const colmap::image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const colmap::Image &image = point_recon.Image(image_id);
  if (!image.HasPose()) {
    return num_tris;
  }

  // Check if this image has structure2d data
  if (!srec.ExistsStructure2D(image_id)) {
    return num_tris;
  }

  const Structure2d &S = srec.Structure2d(image_id);
  const size_t num_groups = S.NumGroups();

  // Correspondence data for reference observation in given image.
  GroupCorrData ref_corr_data;
  ref_corr_data.image_id = image_id;
  ref_corr_data.image = &image;

  // Container for correspondences from reference observation to other images.
  std::vector<GroupCorrData> corrs_data;

  // Try to triangulate all group observations.
  for (group2D_t group2D_idx = 0;
       group2D_idx < static_cast<group2D_t>(num_groups); ++group2D_idx) {
    const Group2d &group2D = S.Group(group2D_idx);

    // Skip invalid group types
    if (group2D.type == GroupType::INVALID) {
      continue;
    }

    const size_t num_triangulated =
        Find(options, image_id, group2D_idx,
             static_cast<size_t>(options.max_transitivity), &corrs_data);

    if (corrs_data.empty()) {
      continue;
    }

    ref_corr_data.group2D_idx = group2D_idx;
    ref_corr_data.group2D = &group2D;

    if (num_triangulated == 0) {
      // No existing 3D group - try to create new one
      corrs_data.push_back(ref_corr_data);
      num_tris += Create(options, corrs_data);
    } else {
      // Continue correspondences to existing 3D groups.
      num_tris += Continue(options, ref_corr_data, corrs_data);
      // Create groups from correspondences that are not continued.
      corrs_data.push_back(ref_corr_data);
      num_tris += Create(options, corrs_data);
    }
  }

  return num_tris;
}

size_t IncrementalGroupTriangulator::CompleteImage(const Options &options,
                                                   colmap::image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const colmap::Image &image = point_recon.Image(image_id);
  if (!image.HasPose()) {
    return num_tris;
  }

  if (!srec.ExistsStructure2D(image_id)) {
    return num_tris;
  }

  Structure2d &S = srec.Structure2d(image_id);
  const size_t num_groups = S.NumGroups();

  for (group2D_t group2D_idx = 0;
       group2D_idx < static_cast<group2D_t>(num_groups); ++group2D_idx) {
    Group2d &group2D = S.Group(group2D_idx);

    if (group2D.group3D_id != kInvalidGroup3dId) {
      // Complete existing track.
      num_tris += Complete(options, group2D.group3D_id);
    } else {
      // Try to create new track from transitive correspondences.
      std::vector<GroupCorrData> corrs_data;
      Find(options, image_id, group2D_idx,
           static_cast<size_t>(options.complete_max_transitivity), &corrs_data);

      if (!corrs_data.empty()) {
        GroupCorrData ref_corr_data;
        ref_corr_data.image_id = image_id;
        ref_corr_data.group2D_idx = group2D_idx;
        ref_corr_data.image = &image;
        ref_corr_data.group2D = &group2D;
        corrs_data.push_back(ref_corr_data);
        num_tris += Create(options, corrs_data);
      }
    }
  }

  return num_tris;
}

size_t IncrementalGroupTriangulator::CompleteTracks(
    const Options &options, const FlatHashSet<group3D_t> &group3D_ids) {
  THROW_CHECK(options.Check());

  size_t num_completed = 0;
  for (const group3D_t group3D_id : group3D_ids) {
    num_completed += Complete(options, group3D_id);
  }
  return num_completed;
}

size_t IncrementalGroupTriangulator::CompleteAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const auto &groups3d = srec.Groups3D();

  FlatHashSet<group3D_t> group3D_ids;
  group3D_ids.reserve(groups3d.size());
  for (const auto &kv : groups3d) {
    group3D_ids.insert(kv.first);
  }

  return CompleteTracks(options, group3D_ids);
}

size_t IncrementalGroupTriangulator::MergeTracks(
    const Options &options, const FlatHashSet<group3D_t> &group3D_ids) {
  THROW_CHECK(options.Check());

  size_t num_merged = 0;
  for (const group3D_t group3D_id : group3D_ids) {
    num_merged += Merge(options, group3D_id);
  }
  return num_merged;
}

size_t IncrementalGroupTriangulator::MergeAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const auto &groups3d = srec.Groups3D();

  FlatHashSet<group3D_t> group3D_ids;
  group3D_ids.reserve(groups3d.size());
  for (const auto &kv : groups3d) {
    group3D_ids.insert(kv.first);
  }

  return MergeTracks(options, group3D_ids);
}

size_t IncrementalGroupTriangulator::UpdateGroupAssociations(
    const Options &options, const FlatHashSet<group3D_t> &group3D_ids) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  size_t num_updated = 0;

  for (const group3D_t group3D_id : group3D_ids) {
    if (!srec.ExistsGroup3D(group3D_id)) {
      continue;
    }

    auto &group3d = srec.Group(group3D_id);

    // Check active association count BEFORE rebuilding (reflects post-BA
    // state). Groups with enough active associations have BA-optimized params
    // that should be preserved; others need full re-estimation.
    const bool keep_params =
        !group3d.GetParams().empty() &&
        group3d.CountActiveAssociations() >= options.min_active_to_keep_params;

    // Build GroupCorrData from all track elements
    std::vector<GroupCorrData> corrs_data;
    for (const auto &elem : group3d.track.Elements()) {
      if (!point_recon.ExistsImage(elem.image_id)) {
        continue;
      }
      const colmap::Image &image = point_recon.Image(elem.image_id);
      if (!image.HasPose()) {
        continue;
      }
      if (!srec.ExistsStructure2D(elem.image_id)) {
        continue;
      }

      const Structure2d &S = srec.Structure2d(elem.image_id);
      const group2D_t group2D_idx = static_cast<group2D_t>(elem.point2D_idx);
      if (group2D_idx >= static_cast<group2D_t>(S.NumGroups())) {
        continue;
      }

      GroupCorrData corr;
      corr.image_id = elem.image_id;
      corr.group2D_idx = group2D_idx;
      corr.image = &image;
      corr.group2D = &S.Group(group2D_idx);
      corrs_data.push_back(corr);
    }

    // Count raw features before support filtering
    FlatHashMap<colmap::point3D_t, int> point_support_raw;
    FlatHashMap<line3D_t, int> line_support_raw;
    {
      std::vector<colmap::point3D_t> pts;
      std::vector<line3D_t> lines;
      for (const auto &corr : corrs_data) {
        Extract3DPoints(corr, pts);
        for (auto pid : pts) {
          point_support_raw[pid]++;
        }
        Extract3DLines(corr, lines);
        for (auto lid : lines) {
          line_support_raw[lid]++;
        }
      }
    }

    // Aggregate features with support counting
    NodeHashMap<colmap::point3D_t, int> point_support;
    NodeHashMap<line3D_t, int> line_support;
    AggregateSharedFeatures(corrs_data,
                            options.min_num_supports_for_association,
                            point_support, line_support);

    // Rebuild associations
    group3d.points.clear();
    for (const auto &[pid, support] : point_support) {
      if (point_recon.ExistsPoint3D(pid)) {
        group3d.AddPoint(AssociatedFeature3d(pid));
      }
    }

    group3d.lines.clear();
    for (const auto &[lid, support] : line_support) {
      if (srec.ExistsLine3D(lid)) {
        group3d.AddLine(AssociatedFeature3d(lid));
      }
    }

    VLOG(2) << "  UpdateGroupAssoc group=" << group3D_id
            << " type=" << group3d.type
            << " track_len=" << group3d.track.Length()
            << " valid_corrs=" << corrs_data.size()
            << " | raw_pts=" << point_support_raw.size()
            << " raw_lns=" << line_support_raw.size()
            << " -> support_pts=" << point_support.size()
            << " support_lns=" << line_support.size()
            << " -> assoc_pts=" << group3d.points.size()
            << " assoc_lns=" << group3d.lines.size();

    // Build 3D feature vectors for verification
    std::vector<V3D> points3d;
    std::vector<Line3d> lines3d;
    points3d.reserve(group3d.points.size());
    lines3d.reserve(group3d.lines.size());
    for (const auto &af : group3d.points) {
      points3d.push_back(point_recon.Point3D(af.idx).xyz);
    }
    for (const auto &af : group3d.lines) {
      lines3d.push_back(srec.Line(af.idx));
    }

    if (points3d.empty() && lines3d.empty()) {
      VLOG(1) << "    -> no features remaining, skipping";
      continue;
    }

    if (!keep_params) {
      // Not BA-optimized: full RANSAC re-initialization + refinement.
      double ransac_thresh = options.GetInitRansacThresh(group3d.type);
      auto new_params = estimators::group3d::InitializeGroupParams(
          group3d.type, points3d, lines3d, ransac_thresh);
      if (!new_params.has_value()) {
        LOG(WARNING) << "Group " << group3D_id
                     << ": param estimation failed, clearing associations";
        group3d.points.clear();
        group3d.lines.clear();
        continue;
      }
      group3d.SetParams(new_params.value());

      // Refine newly initialized params
      size_t num_features = group3d.points.size() + group3d.lines.size();
      if (static_cast<int>(num_features) >=
          options.association_filter_min_num_features) {
        estimators::RefineGroupByReprojError(group3d, reconstruction_);
      }
    }

    num_updated++;

    // Verify associations against params (BA-optimized or newly estimated).
    // For keep_params groups this breaks the circular fitting problem:
    // params are NOT fitted to current associations, so outliers get caught.
    size_t num_features = group3d.points.size() + group3d.lines.size();
    if (static_cast<int>(num_features) >=
        options.association_filter_min_num_features) {
      GroupVerificationOptions verify_opts;
      verify_opts.filter_outliers = true;
      verify_opts.min_num_inliers = 1; // don't reject the whole group here
      verify_opts.min_inlier_ratio = 0.0;
      verify_opts.obs_inlier_ratio = options.verification_obs_inlier_ratio;
      verify_opts.default_reproj_threshold =
          options.association_filter_reproj_threshold;
      VerifyGroup(group3d, reconstruction_, points3d, lines3d, verify_opts);

      VLOG(2) << "    -> after filter: active_pts="
              << group3d.CountActivePoints() << "/" << group3d.points.size()
              << " active_lns=" << group3d.CountActiveLines() << "/"
              << group3d.lines.size()
              << (keep_params ? " (kept BA params)" : " (re-estimated)");
    }
  }

  return num_updated;
}

void IncrementalGroupTriangulator::AddModifiedGroup3D(group3D_t group3D_id) {
  modified_group3D_ids_.insert(group3D_id);
}

const FlatHashSet<group3D_t> &
IncrementalGroupTriangulator::GetModifiedGroups3D() {
  return modified_group3D_ids_;
}

void IncrementalGroupTriangulator::ClearModifiedGroups3D() {
  modified_group3D_ids_.clear();
}

void IncrementalGroupTriangulator::ClearCaches() { merge_trials_.clear(); }

size_t IncrementalGroupTriangulator::Find(
    const Options &options, colmap::image_t image_id, group2D_t group2D_idx,
    size_t transitivity, std::vector<GroupCorrData> *corrs_data) {
  corrs_data->clear();

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const colmap::CorrespondenceGraph &corr_graph =
      structure_corr_graph_->GroupGraph();

  // Get group type for filtering
  const Structure2d &ref_S = srec.Structure2d(image_id);
  const Group2d &ref_group2D = ref_S.Group(group2D_idx);
  const GroupType ref_type = ref_group2D.type;

  // Extract correspondences for this group
  found_corrs_.clear();
  corr_graph.ExtractCorrespondences(
      image_id, static_cast<colmap::point2D_t>(group2D_idx), &found_corrs_);

  size_t num_triangulated = 0;

  for (const auto &corr : found_corrs_) {
    const colmap::image_t corr_image_id = corr.image_id;
    const group2D_t corr_group2D_idx = static_cast<group2D_t>(corr.point2D_idx);

    // Skip if the corresponding image is not registered
    if (!point_recon.ExistsImage(corr_image_id)) {
      continue;
    }
    const colmap::Image &corr_image = point_recon.Image(corr_image_id);
    if (!corr_image.HasPose()) {
      continue;
    }

    // Skip if the corresponding image doesn't have structure2d
    if (!srec.ExistsStructure2D(corr_image_id)) {
      continue;
    }

    const Structure2d &corr_S = srec.Structure2d(corr_image_id);
    if (corr_group2D_idx >= static_cast<group2D_t>(corr_S.NumGroups())) {
      continue;
    }

    const Group2d &corr_group2D = corr_S.Group(corr_group2D_idx);

    // Only match groups of the same type
    if (corr_group2D.type != ref_type) {
      continue;
    }

    GroupCorrData corr_data;
    corr_data.image_id = corr_image_id;
    corr_data.group2D_idx = corr_group2D_idx;
    corr_data.image = &corr_image;
    corr_data.group2D = &corr_group2D;

    corrs_data->push_back(corr_data);

    // Count how many correspondences already have a 3D group
    if (corr_group2D.group3D_id != kInvalidGroup3dId) {
      num_triangulated++;
    }
  }

  return num_triangulated;
}

void IncrementalGroupTriangulator::Extract3DPoints(
    const GroupCorrData &g, std::vector<colmap::point3D_t> &pts) const {
  pts.clear();

  const StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  const Group2d &group2D = *g.group2D;
  const colmap::Image &img = *g.image;

  pts.reserve(group2D.points.size());

  for (const auto &af : group2D.points) {
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

void IncrementalGroupTriangulator::Extract3DLines(
    const GroupCorrData &g, std::vector<line3D_t> &lines) const {
  lines.clear();

  const StructureReconstruction &srec = reconstruction_.StructureRecon();

  const Structure2d &S = srec.Structure2d(g.image_id);
  const Group2d &group2D = *g.group2D;

  lines.reserve(group2D.lines.size());

  for (const auto &af : group2D.lines) {
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

void IncrementalGroupTriangulator::AggregateSharedFeatures(
    const std::vector<GroupCorrData> &corrs_data, int min_support,
    NodeHashMap<colmap::point3D_t, int> &point_support,
    NodeHashMap<line3D_t, int> &line_support) const {
  point_support.clear();
  line_support.clear();

  std::vector<colmap::point3D_t> pts;
  std::vector<line3D_t> lines;

  for (const auto &corr : corrs_data) {
    Extract3DPoints(corr, pts);
    for (auto pid : pts) {
      point_support[pid]++;
    }

    Extract3DLines(corr, lines);
    for (auto lid : lines) {
      line_support[lid]++;
    }
  }

  // Filter to only features with sufficient support
  for (auto it = point_support.begin(); it != point_support.end();) {
    if (it->second < min_support) {
      it = point_support.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = line_support.begin(); it != line_support.end();) {
    if (it->second < min_support) {
      it = line_support.erase(it);
    } else {
      ++it;
    }
  }
}

bool IncrementalGroupTriangulator::CheckObservationError(
    const Group3d &group3d, const GroupCorrData &corr,
    double max_angle_error_vp, double max_reproj_error,
    double surface_dist_mad_factor) const {
  const StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  if (group3d.type == GroupType::VP) {
    // For VP groups: check angular error of lines
    const auto &params = group3d.GetParams();
    if (params.size() != 3) {
      return false;
    }
    V3D vp_dir = Eigen::Map<const V3D>(params.data()).normalized();

    // Check that most lines in the observation align with the VP direction
    const Group2d &group2D = *corr.group2D;
    const Structure2d &S = srec.Structure2d(corr.image_id);

    size_t num_passing = 0;
    size_t num_valid = 0;

    for (const auto &af : group2D.lines) {
      if (af.idx >= S.NumLines())
        continue;
      const Line2d &l2d = S.Line(af.idx);
      if (l2d.line3D_id == kInvalidLine3dId)
        continue;

      const Line3d &l3d = srec.Line(l2d.line3D_id);
      V3D line_dir = l3d.Direction().normalized();

      // Angular error: 1 - |cos(angle)| (0 = perfect alignment)
      double cos_angle = std::abs(line_dir.dot(vp_dir));
      double angle_deg = std::acos(std::min(1.0, cos_angle)) * 180.0 / M_PI;

      num_valid++;
      if (angle_deg < max_angle_error_vp) {
        num_passing++;
      }
    }

    // Require majority of lines to pass
    return num_valid == 0 ||
           (static_cast<double>(num_passing) / num_valid >= 0.5);

  } else {
    // For non-VP groups (Plane, Sphere, etc.): check 2D reprojection error
    // and adaptive 3D distance-to-surface.
    const auto &params = group3d.GetParams();
    auto group_impl = GetGroup(group3d.type);
    if (!group_impl) {
      return false;
    }

    const Group2d &group2D = *corr.group2D;
    const colmap::Image &image = *corr.image;
    const colmap::Camera &camera = point_recon.Camera(image.CameraId());

    size_t num_passing = 0;
    size_t num_valid = 0;

    // Collect candidate 3D distances for the adaptive threshold check
    std::vector<double> candidate_dists;

    // Check points: measure 2D displacement from surface projection
    for (const auto &af : group2D.points) {
      if (af.idx >= image.NumPoints2D())
        continue;
      const auto &p2d = image.Point2D(af.idx);
      if (!p2d.HasPoint3D())
        continue;

      const V3D &p3d = point_recon.Point3D(p2d.point3D_id).xyz;

      // Project through group surface
      V3D projected = group_impl->ProjectPoint(p3d, params.data());

      // Track 3D distance for adaptive check
      candidate_dists.push_back((p3d - projected).norm());

      // Reproject both original and surface-projected to image
      V3D cam_orig = image.CamFromWorld() * p3d;
      auto reproj_orig = camera.ImgFromCam(cam_orig);
      V3D cam_proj = image.CamFromWorld() * projected;
      auto reproj_proj = camera.ImgFromCam(cam_proj);
      if (!reproj_orig || !reproj_proj)
        continue;

      double error = (*reproj_orig - *reproj_proj).norm();
      num_valid++;
      if (error < max_reproj_error) {
        num_passing++;
      }
    }

    // 2D reprojection check: require majority of points to pass
    bool reproj_ok =
        num_valid == 0 || (static_cast<double>(num_passing) / num_valid >= 0.5);
    if (!reproj_ok) {
      return false;
    }

    // Adaptive 3D distance check using MAD of existing associations.
    // This catches parallel planes that pass the 2D check but are offset
    // in depth along the viewing direction.
    if (surface_dist_mad_factor > 0.0 && !candidate_dists.empty() &&
        !group3d.points.empty()) {
      // Compute distances for existing associated points
      std::vector<double> existing_dists;
      existing_dists.reserve(group3d.points.size());
      for (const auto &af : group3d.points) {
        if (!point_recon.ExistsPoint3D(af.idx))
          continue;
        const V3D &p3d = point_recon.Point3D(af.idx).xyz;
        V3D projected = group_impl->ProjectPoint(p3d, params.data());
        existing_dists.push_back((p3d - projected).norm());
      }

      if (!existing_dists.empty()) {
        auto [median, mad] = colmap::MedianAbsoluteDeviation(existing_dists);
        double threshold = median + surface_dist_mad_factor * mad;

        double candidate_median = colmap::Median(candidate_dists);
        if (candidate_median > threshold) {
          return false;
        }
      }
    }

    return true;
  }
}

bool IncrementalGroupTriangulator::CheckMergeCompatibility(
    const Options &options, const Group3d &group1, const Group3d &group2,
    const std::vector<V3D> &merged_points,
    const std::vector<Line3d> &merged_lines) const {
  // Must be same type
  if (group1.type != group2.type) {
    return false;
  }

  if (group1.type == GroupType::VP) {
    // For VP: check that merged lines align with both VP directions
    const auto &params1 = group1.GetParams();
    const auto &params2 = group2.GetParams();
    if (params1.size() != 3 || params2.size() != 3) {
      return false;
    }
    V3D vp_dir1 = Eigen::Map<const V3D>(params1.data()).normalized();
    V3D vp_dir2 = Eigen::Map<const V3D>(params2.data()).normalized();

    // Check angular error between VP directions
    double cos_angle = std::abs(vp_dir1.dot(vp_dir2));
    double angle_deg = std::acos(std::min(1.0, cos_angle)) * 180.0 / M_PI;

    return angle_deg < options.merge_max_angle_error_vp;

  } else {
    // For non-VP groups (Plane, Sphere, etc.): check that merging doesn't
    // significantly change normals or increase the 3D fitting error.
    auto merged_params = estimators::group3d::InitializeGroupParams(
        group1.type, merged_points, merged_lines,
        options.GetInitRansacThresh(group1.type));
    if (!merged_params.has_value()) {
      return false;
    }

    auto group_impl = GetGroup(group1.type);
    if (!group_impl) {
      return false;
    }

    // For PLANE groups: check that merged normal is consistent with both
    // individual normals. Params are [a, b, c, d] where (a,b,c) is unit
    // normal. Reject if either normal deviates by more than threshold.
    if (group1.type == GroupType::PLANE) {
      const auto &p1 = group1.GetParams();
      const auto &p2 = group2.GetParams();
      const auto &pm = merged_params.value();
      if (p1.size() >= 3 && p2.size() >= 3 && pm.size() >= 3) {
        V3D n1 = Eigen::Map<const V3D>(p1.data()).normalized();
        V3D n2 = Eigen::Map<const V3D>(p2.data()).normalized();
        V3D nm = Eigen::Map<const V3D>(pm.data()).normalized();

        double cos1 = std::min(1.0, std::abs(n1.dot(nm)));
        double cos2 = std::min(1.0, std::abs(n2.dot(nm)));
        double angle1_deg = std::acos(cos1) * 180.0 / M_PI;
        double angle2_deg = std::acos(cos2) * 180.0 / M_PI;

        if (angle1_deg > options.merge_max_plane_normal_angle ||
            angle2_deg > options.merge_max_plane_normal_angle) {
          return false;
        }
      }
    }

    // Compute median 3D distance-to-surface for a set of features.
    // Each point and each line counts as one feature.
    auto median_surface_dist = [&](const std::vector<V3D> &pts,
                                   const std::vector<Line3d> &lns,
                                   const double *params) -> double {
      std::vector<double> dists;
      dists.reserve(pts.size() + lns.size());
      for (const auto &p : pts) {
        dists.push_back((p - group_impl->ProjectPoint(p, params)).norm());
      }
      for (const auto &l : lns) {
        double d_start =
            (l.start - group_impl->ProjectPoint(l.start, params)).norm();
        double d_end = (l.end - group_impl->ProjectPoint(l.end, params)).norm();
        dists.push_back((d_start + d_end) / 2.0);
      }
      if (dists.empty()) {
        return 0.0;
      }
      return colmap::Median(dists);
    };

    // Extract individual feature sets from each group
    const auto &point_recon = reconstruction_.PointRecon();
    const auto &srec = reconstruction_.StructureRecon();

    std::vector<V3D> pts1, pts2;
    std::vector<Line3d> lns1, lns2;
    for (const auto &af : group1.points) {
      if (point_recon.ExistsPoint3D(af.idx))
        pts1.push_back(point_recon.Point3D(af.idx).xyz);
    }
    for (const auto &af : group2.points) {
      if (point_recon.ExistsPoint3D(af.idx))
        pts2.push_back(point_recon.Point3D(af.idx).xyz);
    }
    for (const auto &af : group1.lines) {
      if (srec.ExistsLine3D(af.idx))
        lns1.push_back(srec.Line(af.idx));
    }
    for (const auto &af : group2.lines) {
      if (srec.ExistsLine3D(af.idx))
        lns2.push_back(srec.Line(af.idx));
    }

    // Check each group's features against its own params and merged params.
    // Reject if either group's features fit significantly worse on the merged
    // plane than on their own plane.
    double err1_own =
        median_surface_dist(pts1, lns1, group1.GetParams().data());
    double err1_merged =
        median_surface_dist(pts1, lns1, merged_params.value().data());
    if (err1_merged > options.merge_max_error_ratio * err1_own) {
      return false;
    }

    double err2_own =
        median_surface_dist(pts2, lns2, group2.GetParams().data());
    double err2_merged =
        median_surface_dist(pts2, lns2, merged_params.value().data());
    if (err2_merged > options.merge_max_error_ratio * err2_own) {
      return false;
    }

    return true;
  }
}

size_t IncrementalGroupTriangulator::Create(
    const Options &options, const std::vector<GroupCorrData> &corrs_data) {
  // CHECKPOINT 1: Minimum observations
  if (corrs_data.size() < static_cast<size_t>(options.min_num_observations)) {
    return 0;
  }

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  // Filter out correspondences that already have a 3D group
  std::vector<GroupCorrData> new_corrs;
  new_corrs.reserve(corrs_data.size());
  for (const auto &corr : corrs_data) {
    if (corr.group2D->group3D_id == kInvalidGroup3dId) {
      new_corrs.push_back(corr);
    }
  }

  // CHECKPOINT 1 (re-check after filtering)
  if (new_corrs.size() < static_cast<size_t>(options.min_num_observations)) {
    return 0;
  }

  // CHECKPOINT 2: Type consistency
  const GroupType cluster_type = new_corrs[0].group2D->type;
  for (const auto &corr : new_corrs) {
    if (corr.group2D->type != cluster_type) {
      return 0; // Mixed types - skip
    }
  }

  if (cluster_type == GroupType::INVALID) {
    return 0;
  }

  // Aggregate shared 3D features from all correspondences
  NodeHashMap<colmap::point3D_t, int> point_support;
  NodeHashMap<line3D_t, int> line_support;
  AggregateSharedFeatures(new_corrs, options.min_num_supports_for_association,
                          point_support, line_support);

  // CHECKPOINT 3: Min feature associations
  size_t total_features = point_support.size() + line_support.size();
  if (total_features < options.verification_min_num_inliers) {
    return 0;
  }

  // Extract 3D coordinates for RANSAC initialization
  std::vector<V3D> points3d_vec;
  std::vector<AssociatedFeature3d> group_points;
  points3d_vec.reserve(point_support.size());
  group_points.reserve(point_support.size());
  for (const auto &[pid, support] : point_support) {
    if (!point_recon.ExistsPoint3D(pid)) {
      continue;
    }
    const auto &p3d = point_recon.Point3D(pid);
    points3d_vec.push_back(p3d.xyz);
    group_points.push_back(AssociatedFeature3d(pid));
  }

  std::vector<Line3d> lines3d_vec;
  std::vector<AssociatedFeature3d> group_lines;
  lines3d_vec.reserve(line_support.size());
  group_lines.reserve(line_support.size());
  for (const auto &[lid, support] : line_support) {
    if (!srec.ExistsLine3D(lid)) {
      continue;
    }
    lines3d_vec.push_back(srec.Line(lid));
    group_lines.push_back(AssociatedFeature3d(lid));
  }

  // Re-check feature count after filtering non-existent
  if (points3d_vec.size() + lines3d_vec.size() <
      options.verification_min_num_inliers) {
    return 0;
  }

  // CHECKPOINT 4: Parameter initialization via RANSAC
  auto params = estimators::group3d::InitializeGroupParams(
      cluster_type, points3d_vec, lines3d_vec,
      options.GetInitRansacThresh(cluster_type));
  if (!params.has_value()) {
    return 0;
  }

  // Check params validity
  if (!CheckGroupParams3D(cluster_type, params.value().data())) {
    return 0;
  }

  // Create Group3dWithActiveLabels with track and associations
  Group3dWithActiveLabels group3d{Group3d(cluster_type)};
  group3d.SetParams(params.value());
  group3d.points = std::move(group_points);
  group3d.lines = std::move(group_lines);

  // Build track from all new correspondences
  for (const auto &corr : new_corrs) {
    group3d.track.AddElement(colmap::TrackElement(
        corr.image_id, static_cast<colmap::point2D_t>(corr.group2D_idx)));
  }

  // CHECKPOINT 5: Group hypothesis verification
  GroupVerificationOptions verify_opts;
  verify_opts.inlier_threshold = options.GetVerificationThresh(cluster_type);
  verify_opts.default_vp_threshold = options.default_vp_threshold;
  verify_opts.default_reproj_threshold = options.default_reproj_threshold;
  verify_opts.min_num_inliers = options.verification_min_num_inliers;
  verify_opts.min_inlier_ratio = options.verification_min_inlier_ratio;
  verify_opts.obs_inlier_ratio = options.verification_obs_inlier_ratio;
  verify_opts.filter_outliers = options.verification_filter_outliers;

  auto stats = VerifyGroup(group3d, reconstruction_, points3d_vec, lines3d_vec,
                           verify_opts);
  if (!stats.passed) {
    return 0;
  }

  // Add to reconstruction using observation manager
  const group3D_t new_group3D_id = obs_manager_->AddGroup3D(group3d);
  AddModifiedGroup3D(new_group3D_id);

  return new_corrs.size();
}

size_t IncrementalGroupTriangulator::Continue(
    const Options &options, const GroupCorrData &ref_corr_data,
    const std::vector<GroupCorrData> &corrs_data) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();

  // Check if reference already has a 3D group
  if (ref_corr_data.group2D->group3D_id != kInvalidGroup3dId) {
    return 0;
  }

  // Find the best 3D group to continue
  group3D_t best_group3D_id = kInvalidGroup3dId;
  double best_score = -std::numeric_limits<double>::infinity();

  for (const auto &corr : corrs_data) {
    if (corr.group2D->group3D_id == kInvalidGroup3dId) {
      continue;
    }

    const group3D_t group3D_id = corr.group2D->group3D_id;
    if (!srec.ExistsGroup3D(group3D_id)) {
      continue;
    }

    const Group3d &group3d = srec.Group(group3D_id);

    // CHECKPOINT 6: Type compatibility
    if (group3d.type != ref_corr_data.group2D->type) {
      continue;
    }

    // CHECKPOINT 7: Projection error check
    if (!CheckObservationError(group3d, ref_corr_data,
                               options.continue_max_angle_error_vp,
                               options.continue_max_reproj_error,
                               options.surface_dist_mad_factor)) {
      continue;
    }

    // Score based on track length (prefer larger tracks for stability)
    double score = static_cast<double>(group3d.track.Length());
    if (score > best_score) {
      best_score = score;
      best_group3D_id = group3D_id;
    }
  }

  if (best_group3D_id == kInvalidGroup3dId) {
    return 0;
  }

  // Add observation using observation manager
  const colmap::TrackElement track_el(
      ref_corr_data.image_id,
      static_cast<colmap::point2D_t>(ref_corr_data.group2D_idx));
  obs_manager_->AddGroupObservation(best_group3D_id, track_el);
  AddModifiedGroup3D(best_group3D_id);

  return 1;
}

size_t IncrementalGroupTriangulator::Merge(const Options &options,
                                           group3D_t group3D_id) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  const colmap::CorrespondenceGraph &corr_graph =
      structure_corr_graph_->GroupGraph();

  if (!srec.ExistsGroup3D(group3D_id)) {
    return 0;
  }

  const Group3d &group3d = srec.Group(group3D_id);

  // Build candidate set via correspondence graph (like COLMAP)
  FlatHashSet<group3D_t> candidate_ids;
  for (const auto &elem : group3d.track.Elements()) {
    std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
    corr_graph.ExtractCorrespondences(
        elem.image_id, static_cast<colmap::point2D_t>(elem.point2D_idx),
        &corrs);

    for (const auto &corr : corrs) {
      // Check if correspondence's image is registered
      if (!point_recon.ExistsImage(corr.image_id)) {
        continue;
      }
      const colmap::Image &corr_image = point_recon.Image(corr.image_id);
      if (!corr_image.HasPose()) {
        continue;
      }

      // Check if structure2d exists
      if (!srec.ExistsStructure2D(corr.image_id)) {
        continue;
      }

      const Structure2d &corr_S = srec.Structure2d(corr.image_id);
      const group2D_t corr_group2D_idx =
          static_cast<group2D_t>(corr.point2D_idx);
      if (corr_group2D_idx >= static_cast<group2D_t>(corr_S.NumGroups())) {
        continue;
      }

      const Group2d &corr_group2D = corr_S.Group(corr_group2D_idx);

      // Check if this correspondence has a different 3D group
      if (corr_group2D.group3D_id == kInvalidGroup3dId) {
        continue;
      }
      const group3D_t other_id = corr_group2D.group3D_id;
      if (other_id == group3D_id) {
        continue;
      }

      // Check if already tried
      auto &tried = merge_trials_[group3D_id];
      if (tried.count(other_id) > 0) {
        continue;
      }

      candidate_ids.insert(other_id);
    }
  }

  // Try to merge each candidate
  for (const group3D_t other_id : candidate_ids) {
    if (!srec.ExistsGroup3D(other_id)) {
      continue;
    }

    // Mark as tried
    merge_trials_[group3D_id].insert(other_id);
    merge_trials_[other_id].insert(group3D_id);

    const Group3d &other_group3d = srec.Group(other_id);

    // CHECKPOINT 9: Type compatibility
    if (other_group3d.type != group3d.type) {
      continue;
    }

    // Early parameter-based rejection: check direction/normal compatibility
    // before the expensive feature collection + RANSAC.
    if (group3d.type == GroupType::VP) {
      const auto &params1 = group3d.GetParams();
      const auto &params2 = other_group3d.GetParams();
      if (params1.size() == 3 && params2.size() == 3) {
        V3D d1 = Eigen::Map<const V3D>(params1.data()).normalized();
        V3D d2 = Eigen::Map<const V3D>(params2.data()).normalized();
        double angle_deg =
            std::acos(std::min(1.0, std::abs(d1.dot(d2)))) * 180.0 / M_PI;
        if (angle_deg >= options.merge_max_angle_error_vp) {
          continue;
        }
      }
    } else if (group3d.type == GroupType::PLANE) {
      const auto &p1 = group3d.GetParams();
      const auto &p2 = other_group3d.GetParams();
      if (p1.size() >= 3 && p2.size() >= 3) {
        V3D n1 = Eigen::Map<const V3D>(p1.data()).normalized();
        V3D n2 = Eigen::Map<const V3D>(p2.data()).normalized();
        double angle_deg =
            std::acos(std::min(1.0, std::abs(n1.dot(n2)))) * 180.0 / M_PI;
        if (angle_deg >= options.merge_max_plane_normal_angle) {
          continue;
        }
      }
    }

    // Collect merged features for validation
    std::vector<V3D> merged_points;
    std::vector<Line3d> merged_lines;

    for (const auto &af : group3d.points) {
      if (point_recon.ExistsPoint3D(af.idx)) {
        merged_points.push_back(point_recon.Point3D(af.idx).xyz);
      }
    }
    for (const auto &af : other_group3d.points) {
      if (point_recon.ExistsPoint3D(af.idx)) {
        merged_points.push_back(point_recon.Point3D(af.idx).xyz);
      }
    }

    for (const auto &af : group3d.lines) {
      if (srec.ExistsLine3D(af.idx)) {
        merged_lines.push_back(srec.Line(af.idx));
      }
    }
    for (const auto &af : other_group3d.lines) {
      if (srec.ExistsLine3D(af.idx)) {
        merged_lines.push_back(srec.Line(af.idx));
      }
    }

    // CHECKPOINT 10: Merge validation
    if (!CheckMergeCompatibility(options, group3d, other_group3d, merged_points,
                                 merged_lines)) {
      continue;
    }

    // Merge the groups
    const size_t num_merged =
        group3d.track.Length() + other_group3d.track.Length();
    const group3D_t merged_id =
        obs_manager_->MergeGroups3D(group3D_id, other_id);

    // Re-initialize params for merged group
    Group3d &merged_group = srec.Group(merged_id);
    auto merged_params = estimators::group3d::InitializeGroupParams(
        merged_group.type, merged_points, merged_lines,
        options.GetInitRansacThresh(merged_group.type));
    if (merged_params.has_value()) {
      merged_group.SetParams(merged_params.value());
    }

    // Update modified set (like COLMAP)
    modified_group3D_ids_.erase(group3D_id);
    modified_group3D_ids_.erase(other_id);
    modified_group3D_ids_.insert(merged_id);

    // Recursively try to merge the newly merged group (like COLMAP)
    const size_t num_merged_recursive = Merge(options, merged_id);
    if (num_merged_recursive > 0) {
      return num_merged_recursive;
    } else {
      return num_merged;
    }
  }

  return 0;
}

size_t IncrementalGroupTriangulator::Complete(const Options &options,
                                              group3D_t group3D_id) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::CorrespondenceGraph &corr_graph =
      structure_corr_graph_->GroupGraph();

  if (!srec.ExistsGroup3D(group3D_id)) {
    return 0;
  }

  Group3d &group3d = srec.Group(group3D_id);

  // Collect all potential observations via transitive correspondences
  Node2dSet existing_obs;
  for (const auto &elem : group3d.track.Elements()) {
    existing_obs.insert(
        Node2d(elem.image_id, static_cast<feature2D_t>(elem.point2D_idx)));
  }

  Node2dSet candidate_obs;
  for (const auto &elem : group3d.track.Elements()) {
    std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
    corr_graph.ExtractCorrespondences(
        elem.image_id, static_cast<colmap::point2D_t>(elem.point2D_idx),
        &corrs);

    for (const auto &corr : corrs) {
      Node2d node(corr.image_id, static_cast<feature2D_t>(corr.point2D_idx));
      if (existing_obs.count(node) == 0) {
        candidate_obs.insert(node);
      }
    }
  }

  size_t num_completed = 0;
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  for (const Node2d &node : candidate_obs) {
    const colmap::image_t img_id = node.first;
    const group2D_t group2D_idx = static_cast<group2D_t>(node.second);

    // Check if image is registered
    if (!point_recon.ExistsImage(img_id)) {
      continue;
    }
    const colmap::Image &image = point_recon.Image(img_id);
    if (!image.HasPose()) {
      continue;
    }

    // Check structure2d exists
    if (!srec.ExistsStructure2D(img_id)) {
      continue;
    }

    Structure2d &S = srec.Structure2d(img_id);
    if (group2D_idx >= static_cast<group2D_t>(S.NumGroups())) {
      continue;
    }

    Group2d &group2D = S.Group(group2D_idx);

    // Skip if already has a 3D group
    if (group2D.group3D_id != kInvalidGroup3dId) {
      continue;
    }

    // Type compatibility
    if (group2D.type != group3d.type) {
      continue;
    }

    // CHECKPOINT 8: Projection error check
    GroupCorrData corr_data;
    corr_data.image_id = img_id;
    corr_data.group2D_idx = group2D_idx;
    corr_data.image = &image;
    corr_data.group2D = &group2D;

    if (!CheckObservationError(group3d, corr_data,
                               options.complete_max_angle_error_vp,
                               options.complete_max_reproj_error,
                               options.surface_dist_mad_factor)) {
      continue;
    }

    // Add observation using observation manager
    const colmap::TrackElement track_el(
        img_id, static_cast<colmap::point2D_t>(group2D_idx));
    obs_manager_->AddGroupObservation(group3D_id, track_el);
    num_completed++;
  }

  if (num_completed > 0) {
    AddModifiedGroup3D(group3D_id);
  }

  return num_completed;
}

} // namespace limap
