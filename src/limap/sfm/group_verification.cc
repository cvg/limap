#include "limap/sfm/group_verification.h"

#include <algorithm>
#include <cmath>

#include <colmap/scene/image.h>
#include <colmap/util/logging.h>

#include "limap/geometry/groups.h"
#include "limap/scene/structure2d.h"

namespace limap {

namespace {

// Compute 2D surface projection error for a point.
// Measures ||reproject(point) - reproject(surface_project(point))||.
// Returns error in pixels, or -1 if projection failed.
double ComputePoint2DError(const BaseGroup &group_impl, const V3D &point3d,
                           const double *params, const colmap::Image &image,
                           const colmap::Camera &camera) {
  V3D cam_orig = image.CamFromWorld() * point3d;
  auto reproj_orig = camera.ImgFromCam(cam_orig);
  if (!reproj_orig) {
    return -1.0;
  }

  V3D projected = group_impl.ProjectPoint(point3d, params);
  V3D cam_proj = image.CamFromWorld() * projected;
  auto reproj_proj = camera.ImgFromCam(cam_proj);
  if (!reproj_proj) {
    return -1.0;
  }

  return (*reproj_orig - *reproj_proj).norm();
}

// Compute 2D surface projection error for a line.
// Measures how much the 2D reprojection changes when the line endpoints
// are projected onto the surface:
//   avg(||reproject(endpoint) - reproject(surface_project(endpoint))||)
// Returns error in pixels, or -1 if projection failed.
double ComputeLine2DError(const BaseGroup &group_impl, const Line3d &line3d,
                          const double *params, const colmap::Image &image,
                          const colmap::Camera &camera) {
  // Reproject original endpoints
  V3D cam_start_orig = image.CamFromWorld() * line3d.start;
  V3D cam_end_orig = image.CamFromWorld() * line3d.end;
  auto reproj_start_orig = camera.ImgFromCam(cam_start_orig);
  auto reproj_end_orig = camera.ImgFromCam(cam_end_orig);
  if (!reproj_start_orig || !reproj_end_orig) {
    return -1.0;
  }

  // Reproject surface-projected endpoints
  V3D proj_start = group_impl.ProjectPoint(line3d.start, params);
  V3D proj_end = group_impl.ProjectPoint(line3d.end, params);
  V3D cam_start_proj = image.CamFromWorld() * proj_start;
  V3D cam_end_proj = image.CamFromWorld() * proj_end;
  auto reproj_start_proj = camera.ImgFromCam(cam_start_proj);
  auto reproj_end_proj = camera.ImgFromCam(cam_end_proj);
  if (!reproj_start_proj || !reproj_end_proj) {
    return -1.0;
  }

  double error_start = (*reproj_start_orig - *reproj_start_proj).norm();
  double error_end = (*reproj_end_orig - *reproj_end_proj).norm();
  return (error_start + error_end) / 2.0;
}

// Compute angular error between plane normal matches using camera poses.
// Transforms camera-frame normals to world frame and compares.
// Returns angular error in degrees between the two world-frame normals.
double ComputePlaneNormalMatchError(const V3D &normal1_cam,
                                    const V3D &normal2_cam,
                                    const colmap::Image &image1,
                                    const colmap::Image &image2) {
  // Transform camera-frame normals to world frame using inverse rotation
  V3D n1_world = image1.CamFromWorld().rotation().inverse() * normal1_cam;
  V3D n2_world = image2.CamFromWorld().rotation().inverse() * normal2_cam;
  n1_world.normalize();
  n2_world.normalize();

  // Angular error (use abs because normals can point either direction)
  double cos_angle = std::min(1.0, std::abs(n1_world.dot(n2_world)));
  return std::acos(cos_angle) * 180.0 / M_PI;
}

// Returns true if a single plane normal match is geometrically consistent.
bool VerifyPlaneNormalMatch(const V3D &normal1_cam, const V3D &normal2_cam,
                            const colmap::Image &image1,
                            const colmap::Image &image2,
                            double threshold_degrees = 30.0) {
  return ComputePlaneNormalMatchError(normal1_cam, normal2_cam, image1,
                                      image2) < threshold_degrees;
}

// Verify plane matches for an image pair.
GroupMatches VerifyPlaneMatchesForPair(colmap::image_t image_id1,
                                       colmap::image_t image_id2,
                                       const Structure2d &structure1,
                                       const Structure2d &structure2,
                                       const GroupMatches &matches,
                                       const colmap::Reconstruction &recon,
                                       double threshold_degrees = 30.0) {
  // If images don't exist in reconstruction, can't verify - keep all matches
  if (!recon.ExistsImage(image_id1) || !recon.ExistsImage(image_id2)) {
    return matches;
  }

  const auto &image1 = recon.Image(image_id1);
  const auto &image2 = recon.Image(image_id2);

  GroupMatches inliers;
  for (const auto &match : matches) {
    size_t g1_idx = match.group2D_idx1;
    size_t g2_idx = match.group2D_idx2;

    // Invalid indices - skip
    if (g1_idx >= structure1.NumGroups() || g2_idx >= structure2.NumGroups()) {
      continue;
    }

    const auto &group1 = structure1.Group(g1_idx);
    const auto &group2 = structure2.Group(g2_idx);

    // Only verify PLANE-to-PLANE matches, keep all other matches
    if (group1.type != GroupType::PLANE || group2.type != GroupType::PLANE) {
      inliers.push_back(match);
      continue;
    }

    // Plane groups must have exactly 3 params (camera-frame normal)
    const auto &params1 = group1.GetParams();
    const auto &params2 = group2.GetParams();
    THROW_CHECK_EQ(params1.size(), 3) << "Plane group missing normal params";
    THROW_CHECK_EQ(params2.size(), 3) << "Plane group missing normal params";

    // Verify plane normal match
    V3D normal1(params1[0], params1[1], params1[2]);
    V3D normal2(params2[0], params2[1], params2[2]);

    if (VerifyPlaneNormalMatch(normal1, normal2, image1, image2,
                               threshold_degrees)) {
      inliers.push_back(match);
    }
  }

  return inliers;
}

// Compute angular error between VP matches using camera poses.
// Back-projects 2D VP directions to 3D world directions and checks consistency.
// Returns angular error in degrees between the two 3D directions.
double ComputeVPMatchError(const V3D &vp1_params, const V3D &vp2_params,
                           const colmap::Image &image1,
                           const colmap::Image &image2) {
  const auto &cam1 = *image1.CameraPtr();
  const auto &cam2 = *image2.CameraPtr();

  // Back-project VP to camera frame direction
  auto uv1 = cam1.CamFromImg(vp1_params.hnormalized());
  auto uv2 = cam2.CamFromImg(vp2_params.hnormalized());
  if (!uv1 || !uv2) {
    return 180.0;
  }

  V3D dir1_cam((*uv1)[0], (*uv1)[1], 1.0);
  V3D dir2_cam((*uv2)[0], (*uv2)[1], 1.0);
  dir1_cam.normalize();
  dir2_cam.normalize();

  // Transform to world frame
  V3D dir1_world = image1.CamFromWorld().rotation().inverse() * dir1_cam;
  V3D dir2_world = image2.CamFromWorld().rotation().inverse() * dir2_cam;
  dir1_world.normalize();
  dir2_world.normalize();

  // Angular error (VP can point either direction)
  double cos_angle = std::min(1.0, std::abs(dir1_world.dot(dir2_world)));
  return std::acos(cos_angle) * 180.0 / M_PI;
}

// Returns true if a single VP match is geometrically consistent.
bool VerifyVPMatch(const V3D &vp1_params, const V3D &vp2_params,
                   const colmap::Image &image1, const colmap::Image &image2,
                   double threshold_degrees = 10.0) {
  return ComputeVPMatchError(vp1_params, vp2_params, image1, image2) <
         threshold_degrees;
}

// Verify VP matches for an image pair.
GroupMatches VerifyVPMatchesForPair(colmap::image_t image_id1,
                                    colmap::image_t image_id2,
                                    const Structure2d &structure1,
                                    const Structure2d &structure2,
                                    const GroupMatches &matches,
                                    const colmap::Reconstruction &recon,
                                    double threshold_degrees = 10.0) {
  // If images don't exist in reconstruction, can't verify - keep all matches
  if (!recon.ExistsImage(image_id1) || !recon.ExistsImage(image_id2)) {
    return matches;
  }

  const auto &image1 = recon.Image(image_id1);
  const auto &image2 = recon.Image(image_id2);

  GroupMatches inliers;
  for (const auto &match : matches) {
    size_t g1_idx = match.group2D_idx1;
    size_t g2_idx = match.group2D_idx2;

    // Invalid indices - skip
    if (g1_idx >= structure1.NumGroups() || g2_idx >= structure2.NumGroups()) {
      continue;
    }

    const auto &group1 = structure1.Group(g1_idx);
    const auto &group2 = structure2.Group(g2_idx);

    // Only verify VP-to-VP matches, keep all other matches
    if (group1.type != GroupType::VP || group2.type != GroupType::VP) {
      inliers.push_back(match);
      continue;
    }

    // VP groups must have exactly 3 params (homogeneous coords)
    const auto &params1 = group1.GetParams();
    const auto &params2 = group2.GetParams();
    THROW_CHECK_EQ(params1.size(), 3) << "VP group missing params";
    THROW_CHECK_EQ(params2.size(), 3) << "VP group missing params";

    // Verify VP match
    V3D vp1(params1[0], params1[1], params1[2]);
    V3D vp2(params2[0], params2[1], params2[2]);

    if (VerifyVPMatch(vp1, vp2, image1, image2, threshold_degrees)) {
      inliers.push_back(match);
    }
  }

  return inliers;
}

} // namespace

GroupVerificationStats VerifyVPGroup(Group3dWithActiveLabels &group,
                                     const std::vector<Line3d> &lines,
                                     const GroupVerificationOptions &options) {
  GroupVerificationStats stats;

  // Threshold in degrees
  const double threshold_deg = options.inlier_threshold > 0
                                   ? options.inlier_threshold
                                   : options.default_vp_threshold;
  // Convert to 1 - |cos(angle)| metric
  const double threshold = 1.0 - std::cos(threshold_deg * M_PI / 180.0);
  stats.threshold = threshold_deg; // Store in degrees for readability

  // Get VP direction from params (must be exactly 3: [x, y, z] direction)
  THROW_CHECK_EQ(group.GetParams().size(), 3) << "VP group must have 3 params";
  V3D vp_dir = Eigen::Map<const V3D>(group.GetParams().data()).normalized();

  stats.total_lines = lines.size();

  // Clear existing inactive labels, then re-mark based on current errors
  group.ClearInactiveLabels();

  for (size_t i = 0; i < lines.size() && i < group.lines.size(); ++i) {
    V3D line_dir = lines[i].Direction();
    double dir_norm = line_dir.norm();
    if (dir_norm < 1e-10) {
      group.SetLineInactive(group.lines[i].idx);
      continue; // degenerate line → inactive
    }
    line_dir /= dir_norm;
    double error = 1.0 - std::abs(line_dir.dot(vp_dir)); // 1 - |cos(angle)|
    if (error < threshold) {
      stats.num_line_inliers++;
    } else {
      group.SetLineInactive(group.lines[i].idx);
    }
  }

  stats.line_inlier_ratio =
      stats.total_lines > 0
          ? static_cast<double>(stats.num_line_inliers) / stats.total_lines
          : 0.0;

  // Check validation criteria
  stats.passed = (stats.num_line_inliers >= options.min_num_inliers) &&
                 (stats.line_inlier_ratio >= options.min_inlier_ratio);

  return stats;
}

GroupVerificationStats
VerifyGroupByReprojError(Group3dWithActiveLabels &group,
                         const HolisticReconstruction &recon,
                         const GroupVerificationOptions &options) {
  GroupVerificationStats stats;

  // Threshold in pixels
  const double threshold = options.inlier_threshold > 0
                               ? options.inlier_threshold
                               : options.default_reproj_threshold;
  stats.threshold = threshold;

  // Get group implementation for surface projection
  auto group_impl = GetGroup(group.type);
  if (!group_impl) {
    LOG(WARNING) << "Unknown group type: " << group.type;
    return stats; // passed=false
  }
  const auto &params = group.GetParams();

  const auto &point_recon = recon.PointRecon();
  const auto &struct_recon = recon.StructureRecon();

  // Clear existing inactive labels, then re-mark based on current errors
  group.ClearInactiveLabels();

  // Verify points via 2D reprojection
  for (size_t i = 0; i < group.points.size(); ++i) {
    auto pid = group.points[i].idx;
    if (!point_recon.ExistsPoint3D(pid)) {
      group.SetPointInactive(pid);
      continue;
    }

    const auto &p3d = point_recon.Point3D(pid);
    const auto &track = p3d.track;

    size_t passing_views = 0;
    size_t valid_views = 0;

    for (const auto &elem : track.Elements()) {
      if (!point_recon.ExistsImage(elem.image_id)) {
        continue;
      }
      const auto &img = point_recon.Image(elem.image_id);
      const auto &cam = point_recon.Camera(img.CameraId());

      double error =
          ComputePoint2DError(*group_impl, p3d.xyz, params.data(), img, cam);

      if (error >= 0) {
        valid_views++;
        if (error < threshold) {
          passing_views++;
        }
      }
    }

    double view_ratio =
        valid_views > 0 ? static_cast<double>(passing_views) / valid_views : 0;
    if (view_ratio >= options.obs_inlier_ratio) {
      stats.num_point_inliers++;
    } else {
      group.SetPointInactive(pid);
    }
  }
  stats.total_points = group.points.size();
  stats.point_inlier_ratio =
      stats.total_points > 0
          ? static_cast<double>(stats.num_point_inliers) / stats.total_points
          : 0.0;

  // Verify lines via 2D reprojection
  for (size_t i = 0; i < group.lines.size(); ++i) {
    auto lid = group.lines[i].idx;
    if (struct_recon.Lines3D().count(lid) == 0) {
      group.SetLineInactive(lid);
      continue;
    }

    const auto &l3d = struct_recon.Line(lid);
    const auto &track = l3d.track;

    size_t passing_views = 0;
    size_t valid_views = 0;

    for (const auto &elem : track.Elements()) {
      if (!point_recon.ExistsImage(elem.image_id)) {
        continue;
      }
      const auto &img = point_recon.Image(elem.image_id);
      if (!struct_recon.ExistsStructure2D(elem.image_id)) {
        continue;
      }

      const auto &cam = point_recon.Camera(img.CameraId());
      const auto &s2d = struct_recon.Structure2d(elem.image_id);

      // Get 2D line observation
      line2D_t line2D_idx = static_cast<line2D_t>(elem.point2D_idx);
      if (line2D_idx >= static_cast<line2D_t>(s2d.NumLines())) {
        continue;
      }
      const Line2d &obs = s2d.Line(line2D_idx);

      double error =
          ComputeLine2DError(*group_impl, l3d, params.data(), img, cam);

      if (error >= 0) {
        valid_views++;
        if (error < threshold) {
          passing_views++;
        }
      }
    }

    double view_ratio =
        valid_views > 0 ? static_cast<double>(passing_views) / valid_views : 0;
    if (view_ratio >= options.obs_inlier_ratio) {
      stats.num_line_inliers++;
    } else {
      group.SetLineInactive(lid);
    }
  }
  stats.total_lines = group.lines.size();
  stats.line_inlier_ratio =
      stats.total_lines > 0
          ? static_cast<double>(stats.num_line_inliers) / stats.total_lines
          : 0.0;

  // Combined validation (points + lines)
  size_t total_inliers = stats.num_point_inliers + stats.num_line_inliers;
  size_t total_features = stats.total_points + stats.total_lines;
  double combined_ratio =
      total_features > 0 ? static_cast<double>(total_inliers) / total_features
                         : 0.0;

  stats.passed = (total_inliers >= options.min_num_inliers) &&
                 (combined_ratio >= options.min_inlier_ratio);

  return stats;
}

GroupVerificationStats VerifyGroup(Group3dWithActiveLabels &group,
                                   const HolisticReconstruction &recon,
                                   const std::vector<V3D> &points,
                                   const std::vector<Line3d> &lines,
                                   const GroupVerificationOptions &options) {
  (void)points; // Used by plane verification (future)

  // VP is special case (direction at infinity, uses 3D angular error)
  // All other group types use 2D reprojection error (general case)
  if (group.type == GroupType::VP) {
    return VerifyVPGroup(group, lines, options);
  }
  return VerifyGroupByReprojError(group, recon, options);
}

size_t VerifyVPMatches(const colmap::Reconstruction &recon,
                       StructureDatabase &structure_db,
                       double threshold_degrees) {
  auto all_matches = structure_db.ReadAllGroupMatches();
  size_t total_removed = 0;
  size_t num_pairs_verified = 0;

  for (const auto &[pair_id, matches] : all_matches) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    if (!structure_db.ExistsStructure2d(image_id1) ||
        !structure_db.ExistsStructure2d(image_id2)) {
      continue;
    }

    Structure2d structure1 = structure_db.ReadStructure2d(image_id1);
    Structure2d structure2 = structure_db.ReadStructure2d(image_id2);

    GroupMatches inliers =
        VerifyVPMatchesForPair(image_id1, image_id2, structure1, structure2,
                               matches, recon, threshold_degrees);
    num_pairs_verified++;

    size_t removed = matches.size() - inliers.size();
    if (removed > 0) {
      structure_db.DeleteGroupMatches(image_id1, image_id2);
      if (!inliers.empty()) {
        structure_db.WriteGroupMatches(image_id1, image_id2, inliers);
      }
      total_removed += removed;
    }
  }

  LOG(INFO) << "VP matches verification: " << num_pairs_verified
            << " pairs verified";
  return total_removed;
}

size_t VerifyPlaneMatches(const colmap::Reconstruction &recon,
                          StructureDatabase &structure_db,
                          double threshold_degrees) {
  auto all_matches = structure_db.ReadAllGroupMatches();
  size_t total_removed = 0;
  size_t num_pairs_verified = 0;

  for (const auto &[pair_id, matches] : all_matches) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    if (!structure_db.ExistsStructure2d(image_id1) ||
        !structure_db.ExistsStructure2d(image_id2)) {
      continue;
    }

    Structure2d structure1 = structure_db.ReadStructure2d(image_id1);
    Structure2d structure2 = structure_db.ReadStructure2d(image_id2);

    GroupMatches inliers =
        VerifyPlaneMatchesForPair(image_id1, image_id2, structure1, structure2,
                                  matches, recon, threshold_degrees);
    num_pairs_verified++;

    size_t removed = matches.size() - inliers.size();
    if (removed > 0) {
      structure_db.DeleteGroupMatches(image_id1, image_id2);
      if (!inliers.empty()) {
        structure_db.WriteGroupMatches(image_id1, image_id2, inliers);
      }
      total_removed += removed;
    }
  }

  LOG(INFO) << "Plane matches verification: " << num_pairs_verified
            << " pairs verified";
  return total_removed;
}

FilterGroupsStats
FilterGroupAssociations(HolisticReconstruction &recon,
                        const GroupVerificationOptions &options,
                        size_t min_active_for_purge) {
  FilterGroupsStats stats;

  auto &struct_recon = recon.StructureRecon();
  const auto &point_recon = recon.PointRecon();

  // Collect group IDs to process (avoid modifying map while iterating)
  std::vector<group3D_t> group_ids;
  for (const auto &[gid, group] : struct_recon.Groups3D()) {
    group_ids.push_back(gid);
  }

  for (auto gid : group_ids) {
    auto &group = struct_recon.Group(gid);

    // Remove stale associations referencing deleted points/lines.
    // Line/point deletions (FilterLineTracks, DeleteSupportlessLineTracks)
    // don't clean up group association lists, so we do it here.
    group.points.erase(std::remove_if(group.points.begin(), group.points.end(),
                                      [&](const AssociatedFeature3d &af) {
                                        return !point_recon.ExistsPoint3D(
                                            af.idx);
                                      }),
                       group.points.end());
    group.lines.erase(std::remove_if(group.lines.begin(), group.lines.end(),
                                     [&](const AssociatedFeature3d &af) {
                                       return !struct_recon.ExistsLine3D(
                                           af.idx);
                                     }),
                      group.lines.end());
    group.CleanupInactiveSets();

    // Extract 3D point coordinates (aligned with group.points)
    std::vector<V3D> points3d_vec(group.points.size());
    for (size_t i = 0; i < group.points.size(); ++i) {
      points3d_vec[i] = point_recon.Point3D(group.points[i].idx).xyz;
    }

    // Extract 3D lines (aligned with group.lines)
    std::vector<Line3d> lines3d_vec(group.lines.size());
    for (size_t i = 0; i < group.lines.size(); ++i) {
      lines3d_vec[i] = struct_recon.Line(group.lines[i].idx);
    }

    // Verify group (marks bad associations inactive, never removes)
    auto result = VerifyGroup(group, recon, points3d_vec, lines3d_vec, options);
    stats.num_groups_processed++;

    size_t num_inactive = (result.total_points - result.num_point_inliers) +
                          (result.total_lines - result.num_line_inliers);
    stats.num_associations_marked += num_inactive;

    if (result.passed) {
      stats.num_groups_passed++;
    } else {
      stats.num_groups_failed++;
    }

    // Purge threshold: groups with enough active associations have their
    // inactive ones permanently removed. Below threshold, keep everything.
    size_t num_active = group.CountActiveAssociations();
    if (num_active >= min_active_for_purge) {
      size_t purged = group.PurgeInactiveAssociations();
      stats.num_associations_purged += purged;
    }
  }

  VLOG(1) << "FilterGroupAssociations: " << stats.num_groups_passed
          << " passed, " << stats.num_groups_failed << " failed, "
          << stats.num_associations_marked << " marked inactive, "
          << stats.num_associations_purged << " purged";

  return stats;
}

size_t DeleteSupportlessGroups(HolisticReconstruction &recon) {
  auto &struct_recon = recon.StructureRecon();

  // Collect group IDs to process (avoid modifying map while iterating)
  std::vector<group3D_t> group_ids;
  for (const auto &[gid, group] : struct_recon.Groups3D()) {
    group_ids.push_back(gid);
  }

  size_t num_deleted = 0;
  for (auto gid : group_ids) {
    const auto &group = struct_recon.Group(gid);

    bool should_delete =
        (group.CountActiveAssociations() == 0) || (group.track.Length() < 2);

    if (should_delete) {
      struct_recon.DeleteGroup3D(gid);
      num_deleted++;
    }
  }

  if (num_deleted > 0) {
    VLOG(1) << "DeleteSupportlessGroups: deleted " << num_deleted << " groups";
  }

  return num_deleted;
}

} // namespace limap
