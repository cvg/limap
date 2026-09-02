#include "limap/estimators/bundle_adjustment/synthetic_structured_dataset.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <colmap/math/math.h>
#include <colmap/math/random.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/sensor/models.h>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/scene/group2d.h"
#include "limap/scene/group3d.h"
#include "limap/scene/structure2d.h"
#include "limap/scene/wireframe.h"
#include "limap/util/types.h"

namespace limap {
namespace estimators {
namespace {

using Eigen::Vector2d;
using Eigen::Vector3d;

// Random point uniformly in [-extent, +extent]^3.
Vector3d RandomPointInCube(double extent) {
  return Vector3d(colmap::RandomUniformReal(-extent, extent),
                  colmap::RandomUniformReal(-extent, extent),
                  colmap::RandomUniformReal(-extent, extent));
}

// Random point on a Manhattan-aligned plane.
// normal_axis: 0=X, 1=Y, 2=Z.  offset: position along normal axis.
Vector3d RandomPointOnPlane(int normal_axis, double offset, double extent) {
  Vector3d pt;
  pt[normal_axis] = offset;
  pt[(normal_axis + 1) % 3] = colmap::RandomUniformReal(-extent, extent);
  pt[(normal_axis + 2) % 3] = colmap::RandomUniformReal(-extent, extent);
  return pt;
}

// Clamp a point to [-extent, +extent]^3.
void ClampToCube(Vector3d &pt, double extent) {
  for (int i = 0; i < 3; ++i)
    pt[i] = std::clamp(pt[i], -extent, extent);
}

// Build plane params (a, b, c, d) from axis and offset.
// Plane equation: n.x + d = 0, so d = -offset.
std::vector<double> MakePlaneParams(int normal_axis, double offset) {
  std::vector<double> params(4, 0.0);
  params[normal_axis] = 1.0;
  params[3] = -offset;
  return params;
}

// Tangent directions for a Manhattan plane with normal along |axis|.
std::pair<int, int> PlaneTangentAxes(int normal_axis) {
  return {(normal_axis + 1) % 3, (normal_axis + 2) % 3};
}

// Check if a 2D point is within image bounds.
bool InBounds(const Vector2d &px, int width, int height) {
  return px.x() >= 0 && px.x() <= width && px.y() >= 0 && px.y() <= height;
}

} // namespace

void SynthesizeStructuredDataset(
    const SyntheticStructuredDatasetOptions &options,
    HolisticReconstruction *reconstruction) {
  THROW_CHECK_GT(options.num_rigs, 0);
  THROW_CHECK_GT(options.num_cameras_per_rig, 0);
  THROW_CHECK_GT(options.num_frames_per_rig, 0);
  THROW_CHECK_GE(options.num_points3D, 0);
  THROW_CHECK_GE(options.num_lines3D, 0);
  THROW_CHECK_GE(options.num_planes, 0);
  const int num_lines_per_plane = options.num_points_per_plane / 5;
  const int num_points_on_planes = std::min(
      options.num_points_per_plane * options.num_planes, options.num_points3D);
  const int num_lines_on_planes =
      std::min(num_lines_per_plane * options.num_planes, options.num_lines3D);
  THROW_CHECK(num_lines_on_planes >= 2 || options.num_wireframe_edges == 0);
  if (options.num_wireframe_edges > 0) {
    THROW_CHECK_GE(options.num_points3D,
                   std::max(options.num_points_per_plane * options.num_planes,
                            options.num_wireframe_edges / 2))
        << "num_points3D too small for planes + wireframe";
  }
  THROW_CHECK_GT(options.camera_sphere_radius, options.cube_half_extent);

  auto &point_recon = reconstruction->PointRecon();
  auto &structure_recon = reconstruction->StructureRecon();
  const double extent = options.cube_half_extent;
  const int num_images = options.num_rigs * options.num_cameras_per_rig *
                         options.num_frames_per_rig;

  // =========================================================================
  // Create cameras, rigs, frames, and images
  // =========================================================================

  int total_num_images = 0;
  // Store cam_from_world for each image_id (needed for projection).
  FlatHashMap<colmap::image_t, colmap::Rigid3d> cam_from_world_map;

  for (int rig_idx = 0; rig_idx < options.num_rigs; ++rig_idx) {
    colmap::Rig rig;

    std::vector<colmap::sensor_t> camera_sensor_ids;
    camera_sensor_ids.reserve(options.num_cameras_per_rig);

    for (int cam_idx = 0; cam_idx < options.num_cameras_per_rig; ++cam_idx) {
      colmap::Camera camera;
      camera.width = options.camera_width;
      camera.height = options.camera_height;
      camera.model_id = colmap::SimplePinholeCameraModel::model_id;
      camera.params = {1280.0, 512.0, 384.0};
      camera.camera_id = rig_idx * options.num_cameras_per_rig + cam_idx + 1;
      point_recon.AddCamera(camera);

      if (rig.NumSensors() == 0) {
        rig.AddRefSensor(camera.SensorId());
      } else {
        // Small random offset for non-reference sensors.
        colmap::Rigid3d sensor_from_rig;
        sensor_from_rig.translation() =
            Vector3d(colmap::RandomGaussian(0.0, 0.05),
                     colmap::RandomGaussian(0.0, 0.05),
                     colmap::RandomGaussian(0.0, 0.05));
        rig.AddSensor(camera.SensorId(), sensor_from_rig);
      }
      camera_sensor_ids.push_back(camera.SensorId());
    }

    const colmap::rig_t rig_id = rig_idx + 1;
    rig.SetRigId(rig_id);
    point_recon.AddRig(rig);

    for (int frame_idx = 0; frame_idx < options.num_frames_per_rig;
         ++frame_idx) {
      colmap::Frame frame;
      frame.SetRigId(rig.RigId());

      // Camera on sphere looking inward toward origin.
      const Vector3d view_dir = -Eigen::Vector3d::Random().normalized();
      const Vector3d proj_center = -options.camera_sphere_radius * view_dir;
      colmap::Rigid3d rig_from_world;
      rig_from_world.rotation() =
          Eigen::Quaterniond::FromTwoVectors(view_dir, Vector3d(0, 0, 1));
      rig_from_world.translation() = rig_from_world.rotation() * -proj_center;
      frame.SetRigFromWorld(rig_from_world);

      std::vector<colmap::Image> images;
      images.reserve(options.num_cameras_per_rig);
      std::vector<colmap::Rigid3d> cams_from_world;
      cams_from_world.reserve(options.num_cameras_per_rig);

      for (int cam_idx = 0; cam_idx < options.num_cameras_per_rig; ++cam_idx) {
        ++total_num_images;
        const auto &sensor_id = camera_sensor_ids[cam_idx];

        colmap::Image image;
        image.SetName("synth_" + std::to_string(total_num_images));
        image.SetCameraId(sensor_id.id);
        image.SetImageId(total_num_images);

        frame.AddDataId(image.DataId());

        // Compose cam_from_world (sensor_from_rig * rig_from_world).
        const colmap::Rigid3d sensor_from_rig =
            rig.IsRefSensor(sensor_id) ? colmap::Rigid3d()
                                       : rig.SensorFromRig(sensor_id);
        const colmap::Rigid3d cam_from_world = sensor_from_rig * rig_from_world;
        cams_from_world.push_back(cam_from_world);
        cam_from_world_map[image.ImageId()] = cam_from_world;

        images.push_back(std::move(image));
      }

      const colmap::frame_t frame_id =
          rig_idx * options.num_frames_per_rig + frame_idx + 1;
      frame.SetFrameId(frame_id);
      point_recon.AddFrame(std::move(frame));

      for (int cam_idx = 0; cam_idx < options.num_cameras_per_rig; ++cam_idx) {
        images[cam_idx].SetFrameId(frame_id);
        point_recon.AddImage(std::move(images[cam_idx]));
      }
    }
  }

  // Collect all registered image IDs for projection.
  std::vector<colmap::image_t> all_image_ids;
  all_image_ids.reserve(num_images);
  for (const auto &[img_id, _] : point_recon.Images()) {
    all_image_ids.push_back(img_id);
  }
  std::sort(all_image_ids.begin(), all_image_ids.end());

  // =========================================================================
  // Generate random Manhattan planes
  // =========================================================================
  struct PlaneInfo {
    int normal_axis;
    double offset;
    std::vector<double> params; // (a, b, c, d)
  };
  std::vector<PlaneInfo> planes(options.num_planes);
  for (int i = 0; i < options.num_planes; ++i) {
    planes[i].normal_axis = colmap::RandomUniformInteger(0, 2);
    planes[i].offset = colmap::RandomUniformReal(-extent, extent);
    planes[i].params = MakePlaneParams(planes[i].normal_axis, planes[i].offset);
  }

  // =========================================================================
  // Generate 3D lines and wireframe structure
  // =========================================================================
  // Grid-based approach: create a regular grid of junction points on each
  // plane, then create lines along grid rows and columns. We control which
  // (point, line) pairs become wireframe edges to match the target count.

  struct LineInfo {
    Line3d line;
    int plane_id; // -1 if free-floating
  };
  // Generate ~10% extra to compensate for visibility filtering.
  const int extra_factor_num = 11;
  const int extra_factor_den = 10;
  std::vector<LineInfo> all_lines;

  struct WireframeEdgeInfo {
    colmap::point3D_t point_id;
    size_t line_idx;
    double weight;
  };
  std::vector<WireframeEdgeInfo> wireframe_edges;
  wireframe_edges.reserve(options.num_wireframe_edges * 2);

  // Junction point IDs → plane mapping.
  FlatHashMap<colmap::point3D_t, int> junction_point_to_plane;

  // Count junctions per plane (for point budget calculation).
  FlatHashMap<int, int> junctions_per_plane;

  const int target_wf_edges = options.num_wireframe_edges;

  // Track how many wireframe lines we create (for line budget).
  int wireframe_lines_count = 0;

  // Store potential (junction, line) pairs for wireframe edge selection.
  struct PotentialEdge {
    colmap::point3D_t point_id;
    size_t line_idx;
    int plane_id;
  };
  std::vector<PotentialEdge> potential_edges;

  // Generate wireframe grid (junction points and grid lines)
  if (options.num_wireframe_edges > 0 && options.num_planes > 0) {
    // Each junction is on 2 lines (horizontal + vertical) → 2 potential edges.
    // Grid size G where G^2 junctions per plane fit in the point budget.

    // Constrain grid size by point budget: grid^2 * planes <= num_points3D
    const int max_grid_from_points = static_cast<int>(std::sqrt(
        static_cast<double>(options.num_points3D) / options.num_planes));
    const int grid_size = std::max(2, max_grid_from_points);

    // Use all planes for wireframe.
    const int num_wf_planes = options.num_planes;
    const int junctions_per_plane_count = grid_size * grid_size;

    // Create all rows and columns (2*grid_size lines per plane).
    // Line budget is handled later when we trim excess lines.
    const int lines_per_plane = 2 * grid_size;

    // Grid spacing within the cube extent.
    const double grid_spacing = (2.0 * extent) / (grid_size + 1);

    // For each plane, create the full grid of junctions but only some lines.
    for (int pi = 0; pi < num_wf_planes; ++pi) {
      const auto &plane = planes[pi];
      auto [t1, t2] = PlaneTangentAxes(plane.normal_axis);

      // Random offset for this plane's grid to add variety.
      double offset_t1 =
          colmap::RandomUniformReal(-grid_spacing / 2, grid_spacing / 2);
      double offset_t2 =
          colmap::RandomUniformReal(-grid_spacing / 2, grid_spacing / 2);

      // Generate full grid of junction points.
      // junctions_grid[row][col] = point_id
      std::vector<std::vector<colmap::point3D_t>> junctions_grid(
          grid_size, std::vector<colmap::point3D_t>(grid_size));

      for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
          Vector3d pos;
          pos[plane.normal_axis] = plane.offset;
          pos[t1] = -extent + (col + 1) * grid_spacing + offset_t1;
          pos[t2] = -extent + (row + 1) * grid_spacing + offset_t2;
          ClampToCube(pos, extent);

          colmap::point3D_t pt_id =
              point_recon.AddPoint3D(pos, colmap::Track());
          junctions_grid[row][col] = pt_id;
          junction_point_to_plane[pt_id] = pi;
          junctions_per_plane[pi]++;
        }
      }

      // Create horizontal lines for all rows.
      for (int row = 0; row < grid_size; ++row) {
        const auto &first_pt = point_recon.Point3D(junctions_grid[row][0]).xyz;
        const auto &last_pt =
            point_recon.Point3D(junctions_grid[row][grid_size - 1]).xyz;

        size_t line_idx = all_lines.size();
        all_lines.push_back({Line3d(first_pt, last_pt), pi});
        wireframe_lines_count++;

        for (int col = 0; col < grid_size; ++col) {
          potential_edges.push_back({junctions_grid[row][col], line_idx, pi});
        }
      }

      // Create vertical lines for all columns.
      for (int col = 0; col < grid_size; ++col) {
        const auto &first_pt = point_recon.Point3D(junctions_grid[0][col]).xyz;
        const auto &last_pt =
            point_recon.Point3D(junctions_grid[grid_size - 1][col]).xyz;

        size_t line_idx = all_lines.size();
        all_lines.push_back({Line3d(first_pt, last_pt), pi});
        wireframe_lines_count++;

        for (int row = 0; row < grid_size; ++row) {
          potential_edges.push_back({junctions_grid[row][col], line_idx, pi});
        }
      }
    }

    // Select which potential edges become actual wireframe edges.
    // Over-generate by 2x to account for visibility filtering.
    const int edges_to_select =
        std::min(static_cast<int>(potential_edges.size()), 2 * target_wf_edges);

    // Shuffle and select.
    std::shuffle(potential_edges.begin(), potential_edges.end(), *colmap::PRNG);
    for (int i = 0; i < edges_to_select; ++i) {
      const auto &pe = potential_edges[i];
      wireframe_edges.push_back({pe.point_id, pe.line_idx, 1.0});
    }
  }

  // Generate remaining non-wireframe lines
  const int remaining_lines_on_planes =
      std::max(0, num_lines_on_planes * extra_factor_num / extra_factor_den -
                      wireframe_lines_count);
  const int gen_free_lines = (options.num_lines3D - num_lines_on_planes) *
                             extra_factor_num / extra_factor_den;

  all_lines.reserve(all_lines.size() + remaining_lines_on_planes +
                    gen_free_lines);

  // Additional lines on planes (non-wireframe).
  for (int i = 0; i < remaining_lines_on_planes; ++i) {
    const int plane_id =
        colmap::RandomUniformInteger(0, options.num_planes - 1);
    const auto &plane = planes[plane_id];
    auto [t1, t2] = PlaneTangentAxes(plane.normal_axis);
    // Pick a random tangent direction.
    int dir_axis = (colmap::RandomUniformInteger(0, 1) == 0) ? t1 : t2;
    Vector3d dir = Vector3d::Zero();
    dir[dir_axis] = (colmap::RandomUniformInteger(0, 1) == 0) ? 1.0 : -1.0;

    Vector3d base = RandomPointOnPlane(plane.normal_axis, plane.offset, extent);
    double length = colmap::RandomUniformReal(options.line_min_length,
                                              options.line_max_length);
    Vector3d start = base - (length / 2.0) * dir;
    Vector3d end = base + (length / 2.0) * dir;
    ClampToCube(start, extent);
    ClampToCube(end, extent);

    all_lines.push_back({Line3d(start, end), plane_id});
  }

  // Free-floating lines.
  for (int i = 0; i < gen_free_lines; ++i) {
    Vector3d start = RandomPointInCube(extent);
    Vector3d dir = Eigen::Vector3d::Random().normalized();
    double length = colmap::RandomUniformReal(options.line_min_length,
                                              options.line_max_length);
    Vector3d end = start + length * dir;
    ClampToCube(end, extent);

    all_lines.push_back({Line3d(start, end), -1});
  }

  // =========================================================================
  // Generate 3D points
  // =========================================================================
  // Junction points fill part of each plane's budget. Additional plane points
  // are generated to reach the target per plane. Remaining budget goes to
  // free-floating points.

  // Track which point3D_id is on which plane (-1 if free).
  FlatHashMap<colmap::point3D_t, int> point_to_plane;

  // Junction points get plane associations (up to ppp per plane).
  // Track how many associations have been assigned per plane.
  FlatHashMap<int, int> plane_assoc_count;
  for (const auto &[pt_id, pid] : junction_point_to_plane) {
    if (plane_assoc_count[pid] < options.num_points_per_plane) {
      point_to_plane[pt_id] = pid;
      plane_assoc_count[pid]++;
    } else {
      // Excess junctions become free points (no plane association).
      point_to_plane[pt_id] = -1;
    }
  }

  // Generate additional plane points to fill each plane's budget.
  int gen_additional_plane_points = 0;
  for (int pi = 0; pi < options.num_planes; ++pi) {
    const int assigned =
        plane_assoc_count.count(pi) ? plane_assoc_count[pi] : 0;
    const int need = std::max(0, options.num_points_per_plane - assigned);
    for (int i = 0; i < need * extra_factor_num / extra_factor_den; ++i) {
      Vector3d pt =
          RandomPointOnPlane(planes[pi].normal_axis, planes[pi].offset, extent);
      colmap::point3D_t pt_id = point_recon.AddPoint3D(pt, colmap::Track());
      point_to_plane[pt_id] = pi;
    }
    gen_additional_plane_points += need;
  }

  // Free-floating points (remaining budget).
  const int num_junction_points =
      static_cast<int>(junction_point_to_plane.size());
  const int gen_free_points =
      std::max(0, options.num_points3D - gen_additional_plane_points -
                      num_junction_points) *
      extra_factor_num / extra_factor_den;
  for (int i = 0; i < gen_free_points; ++i) {
    Vector3d pt = RandomPointInCube(extent);
    colmap::point3D_t pt_id = point_recon.AddPoint3D(pt, colmap::Track());
    point_to_plane[pt_id] = -1;
  }

  // =========================================================================
  // Project to 2D and build tracks
  // =========================================================================
  // Sample random views for each feature and project. The scene is compact
  // with cameras surrounding it, so we randomly sample a small candidate set
  // per feature instead of testing all cameras.
  struct PointVisibility {
    colmap::image_t image_id;
    Vector2d pixel;
  };
  FlatHashMap<colmap::point3D_t, std::vector<PointVisibility>> point_visible;

  // Number of candidate views to try per feature (>= track_length).
  const int pt_candidates = std::min(static_cast<int>(all_image_ids.size()),
                                     std::max(options.point_track_length * 3,
                                              options.min_visible_cameras * 3));

  // Identify junction points — their views will be sampled from connected
  // lines after line sampling, not randomly.
  FlatHashSet<colmap::point3D_t> junction_point_ids;
  for (const auto &edge : wireframe_edges) {
    junction_point_ids.insert(edge.point_id);
  }

  for (const auto &[pt_id, pt3d] : point_recon.Points3D()) {
    if (junction_point_ids.count(pt_id))
      continue; // Junction points handled after line sampling.
    auto &vis = point_visible[pt_id];
    // Sample random images without replacement using rejection sampling.
    FlatHashSet<int> sampled;
    int attempts = 0;
    const int max_attempts = pt_candidates * 4;
    while (static_cast<int>(vis.size()) < pt_candidates &&
           attempts < max_attempts) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++attempts;
      if (!sampled.insert(idx).second)
        continue;
      colmap::image_t img_id = all_image_ids[idx];
      const auto &cam =
          point_recon.Camera(point_recon.Image(img_id).CameraId());
      const auto &cfw = cam_from_world_map[img_id];
      Vector3d pt_cam = cfw * pt3d.xyz;
      if (pt_cam.z() <= 0)
        continue;
      auto proj = cam.ImgFromCam(pt_cam);
      if (!proj.has_value())
        continue;
      if (InBounds(*proj, options.camera_width, options.camera_height)) {
        vis.push_back({img_id, *proj});
      }
    }
  }

  // Select shared cameras per plane (for co-visibility within planes).
  // Lines on the same plane will share (ratio * line_track_length) views.
  const int shared_count = static_cast<int>(options.plane_shared_track_ratio *
                                            options.line_track_length);
  FlatHashMap<int, std::vector<colmap::image_t>> plane_shared_cameras;
  FlatHashMap<int, FlatHashSet<colmap::image_t>> plane_shared_camera_set;
  for (int pi = 0; pi < options.num_planes; ++pi) {
    auto &cameras = plane_shared_cameras[pi];
    auto &cam_set = plane_shared_camera_set[pi];
    FlatHashSet<int> sampled;
    int attempts = 0;
    while (static_cast<int>(cameras.size()) < shared_count &&
           attempts < shared_count * 4) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++attempts;
      if (!sampled.insert(idx).second)
        continue;
      cameras.push_back(all_image_ids[idx]);
      cam_set.insert(all_image_ids[idx]);
    }
  }

  // Sample random views for each line and project.
  struct LineVisibility {
    colmap::image_t image_id;
    Line2d line2d;
  };
  std::vector<std::vector<LineVisibility>> line_visible(all_lines.size());

  const int ln_candidates = std::min(
      static_cast<int>(all_image_ids.size()),
      std::max(options.line_track_length * 3, options.min_visible_cameras * 3));

  for (size_t li = 0; li < all_lines.size(); ++li) {
    const auto &line = all_lines[li].line;
    auto &vis = line_visible[li];
    FlatHashSet<int> sampled;
    FlatHashSet<colmap::image_t> shared_sampled;

    // For plane lines, project through shared cameras first.
    if (all_lines[li].plane_id >= 0) {
      for (colmap::image_t img_id :
           plane_shared_cameras[all_lines[li].plane_id]) {
        if (!shared_sampled.insert(img_id).second)
          continue;
        const auto &cam =
            point_recon.Camera(point_recon.Image(img_id).CameraId());
        const auto &cfw = cam_from_world_map[img_id];
        Vector3d s_cam = cfw * line.start;
        Vector3d e_cam = cfw * line.end;
        if (s_cam.z() <= 0 && e_cam.z() <= 0)
          continue;
        auto p1 = cam.ImgFromCam(s_cam);
        auto p2 = cam.ImgFromCam(e_cam);
        if (!p1.has_value() || !p2.has_value())
          continue;
        if (InBounds(*p1, options.camera_width, options.camera_height) ||
            InBounds(*p2, options.camera_width, options.camera_height)) {
          vis.push_back({img_id, Line2d(*p1, *p2)});
        }
      }
    }

    // Sample additional random cameras.
    int attempts = 0;
    const int max_attempts = ln_candidates * 4;
    while (static_cast<int>(vis.size()) < ln_candidates &&
           attempts < max_attempts) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++attempts;
      if (!sampled.insert(idx).second)
        continue;
      colmap::image_t img_id = all_image_ids[idx];
      if (shared_sampled.count(img_id))
        continue;
      const auto &cam =
          point_recon.Camera(point_recon.Image(img_id).CameraId());
      const auto &cfw = cam_from_world_map[img_id];
      Vector3d s_cam = cfw * line.start;
      Vector3d e_cam = cfw * line.end;
      if (s_cam.z() <= 0 && e_cam.z() <= 0)
        continue;
      auto p1 = cam.ImgFromCam(s_cam);
      auto p2 = cam.ImgFromCam(e_cam);
      if (!p1.has_value() || !p2.has_value())
        continue;
      if (InBounds(*p1, options.camera_width, options.camera_height) ||
          InBounds(*p2, options.camera_width, options.camera_height)) {
        vis.push_back({img_id, Line2d(*p1, *p2)});
      }
    }
  }

  // Filter and subsample line tracks.
  // For plane lines, preserve shared views and randomly fill the rest.
  std::vector<size_t> valid_line_indices;
  for (size_t li = 0; li < all_lines.size(); ++li) {
    auto &vis = line_visible[li];
    if (static_cast<int>(vis.size()) < options.min_visible_cameras)
      continue;
    if (static_cast<int>(vis.size()) > options.line_track_length) {
      if (all_lines[li].plane_id >= 0 && shared_count > 0) {
        // Partition into shared and non-shared views.
        const auto &shared_set =
            plane_shared_camera_set[all_lines[li].plane_id];
        std::vector<LineVisibility> shared_vis, other_vis;
        for (auto &v : vis) {
          if (shared_set.count(v.image_id)) {
            shared_vis.push_back(std::move(v));
          } else {
            other_vis.push_back(std::move(v));
          }
        }
        int rest =
            options.line_track_length - static_cast<int>(shared_vis.size());
        if (rest > 0) {
          std::shuffle(other_vis.begin(), other_vis.end(), *colmap::PRNG);
          if (static_cast<int>(other_vis.size()) > rest)
            other_vis.resize(rest);
        } else {
          other_vis.clear();
        }
        vis.clear();
        vis.insert(vis.end(), shared_vis.begin(), shared_vis.end());
        vis.insert(vis.end(), other_vis.begin(), other_vis.end());
      } else {
        std::shuffle(vis.begin(), vis.end(), *colmap::PRNG);
        vis.resize(options.line_track_length);
      }
    }
    valid_line_indices.push_back(li);
  }

  // Sample junction point views from lines' final tracks.
  // Each junction gets the plane's shared views, then fills the rest from
  // connected lines' non-shared views (split evenly per connected line).
  {
    FlatHashMap<colmap::point3D_t, std::vector<size_t>> junc_to_lines;
    for (const auto &edge : wireframe_edges) {
      junc_to_lines[edge.point_id].push_back(edge.line_idx);
    }
    const int target = options.point_track_length;
    for (const auto &[pt_id, connected] : junc_to_lines) {
      if (!point_recon.ExistsPoint3D(pt_id))
        continue;
      const auto &pt3d = point_recon.Point3D(pt_id);
      auto &vis = point_visible[pt_id];
      FlatHashSet<colmap::image_t> used;

      // 1. Add shared plane views.
      auto jp_it = junction_point_to_plane.find(pt_id);
      const int plane_id =
          (jp_it != junction_point_to_plane.end()) ? jp_it->second : -1;
      const FlatHashSet<colmap::image_t> *shared_set_ptr = nullptr;
      if (plane_id >= 0) {
        auto sc_it = plane_shared_cameras.find(plane_id);
        if (sc_it != plane_shared_cameras.end()) {
          for (colmap::image_t img_id : sc_it->second) {
            if (used.count(img_id))
              continue;
            const auto &cam =
                point_recon.Camera(point_recon.Image(img_id).CameraId());
            const auto &cfw = cam_from_world_map[img_id];
            Vector3d pt_cam = cfw * pt3d.xyz;
            if (pt_cam.z() <= 0)
              continue;
            auto proj = cam.ImgFromCam(pt_cam);
            if (!proj.has_value())
              continue;
            if (InBounds(*proj, options.camera_width, options.camera_height)) {
              vis.push_back({img_id, *proj});
              used.insert(img_id);
            }
          }
        }
        auto ss_it = plane_shared_camera_set.find(plane_id);
        if (ss_it != plane_shared_camera_set.end())
          shared_set_ptr = &ss_it->second;
      }

      // 2. Fill rest from connected lines' non-shared final track views.
      const int rest = target - static_cast<int>(vis.size());
      const int per_line =
          std::max(1, rest / static_cast<int>(connected.size()));
      for (size_t li : connected) {
        std::vector<colmap::image_t> non_shared;
        for (const auto &lv : line_visible[li]) {
          if (used.count(lv.image_id))
            continue;
          if (shared_set_ptr && shared_set_ptr->count(lv.image_id))
            continue;
          non_shared.push_back(lv.image_id);
        }
        std::shuffle(non_shared.begin(), non_shared.end(), *colmap::PRNG);
        int taken = 0;
        for (colmap::image_t img_id : non_shared) {
          if (taken >= per_line)
            break;
          const auto &cam =
              point_recon.Camera(point_recon.Image(img_id).CameraId());
          const auto &cfw = cam_from_world_map[img_id];
          Vector3d pt_cam = cfw * pt3d.xyz;
          if (pt_cam.z() <= 0)
            continue;
          auto proj = cam.ImgFromCam(pt_cam);
          if (!proj.has_value())
            continue;
          if (InBounds(*proj, options.camera_width, options.camera_height)) {
            vis.push_back({img_id, *proj});
            used.insert(img_id);
            ++taken;
          }
        }
      }
    }
  }

  // Filter and subsample non-junction point tracks.
  std::vector<colmap::point3D_t> valid_point_ids;
  for (auto &[pt_id, vis] : point_visible) {
    if (static_cast<int>(vis.size()) < options.min_visible_cameras) {
      point_recon.DeletePoint3D(pt_id);
      continue;
    }
    const int target = options.point_track_length;
    if (static_cast<int>(vis.size()) > target) {
      std::shuffle(vis.begin(), vis.end(), *colmap::PRNG);
      vis.resize(target);
    }
    valid_point_ids.push_back(pt_id);
  }

  // Trim to requested counts.
  if (static_cast<int>(valid_point_ids.size()) > options.num_points3D) {
    std::shuffle(valid_point_ids.begin(), valid_point_ids.end(), *colmap::PRNG);
    for (size_t i = options.num_points3D; i < valid_point_ids.size(); ++i) {
      point_recon.DeletePoint3D(valid_point_ids[i]);
    }
    valid_point_ids.resize(options.num_points3D);
  }
  if (static_cast<int>(valid_line_indices.size()) > options.num_lines3D) {
    std::shuffle(valid_line_indices.begin(), valid_line_indices.end(),
                 *colmap::PRNG);
    valid_line_indices.resize(options.num_lines3D);
  }

  // Generate additional free-floating points to reach requested count.
  while (static_cast<int>(valid_point_ids.size()) < options.num_points3D) {
    Vector3d pt = RandomPointInCube(extent);
    colmap::point3D_t pt_id = point_recon.AddPoint3D(pt, colmap::Track());

    std::vector<PointVisibility> vis;
    FlatHashSet<int> sampled_ff;
    int ff_attempts = 0;
    while (static_cast<int>(vis.size()) < pt_candidates &&
           ff_attempts < pt_candidates * 4) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++ff_attempts;
      if (!sampled_ff.insert(idx).second)
        continue;
      colmap::image_t img_id = all_image_ids[idx];
      const auto &cam =
          point_recon.Camera(point_recon.Image(img_id).CameraId());
      const auto &cfw = cam_from_world_map[img_id];
      Vector3d pt_cam = cfw * pt;
      if (pt_cam.z() <= 0)
        continue;
      auto proj = cam.ImgFromCam(pt_cam);
      if (!proj.has_value())
        continue;
      if (InBounds(*proj, options.camera_width, options.camera_height)) {
        vis.push_back({img_id, *proj});
      }
    }

    if (static_cast<int>(vis.size()) < options.min_visible_cameras) {
      point_recon.DeletePoint3D(pt_id);
      continue;
    }
    std::shuffle(vis.begin(), vis.end(), *colmap::PRNG);
    if (static_cast<int>(vis.size()) > options.point_track_length) {
      vis.resize(options.point_track_length);
    }
    point_visible[pt_id] = std::move(vis);
    point_to_plane[pt_id] = -1;
    valid_point_ids.push_back(pt_id);
  }

  // Generate additional free-floating lines to reach requested count.
  while (static_cast<int>(valid_line_indices.size()) < options.num_lines3D) {
    Vector3d start = RandomPointInCube(extent);
    Vector3d dir = Eigen::Vector3d::Random().normalized();
    double length = colmap::RandomUniformReal(options.line_min_length,
                                              options.line_max_length);
    Vector3d end = start + length * dir;
    ClampToCube(end, extent);

    size_t li = all_lines.size();
    all_lines.push_back({Line3d(start, end), -1});
    line_visible.emplace_back();

    auto &new_vis = line_visible[li];
    FlatHashSet<int> sampled_ff;
    int ff_attempts = 0;
    while (static_cast<int>(new_vis.size()) < ln_candidates &&
           ff_attempts < ln_candidates * 4) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++ff_attempts;
      if (!sampled_ff.insert(idx).second)
        continue;
      colmap::image_t img_id = all_image_ids[idx];
      const auto &cam =
          point_recon.Camera(point_recon.Image(img_id).CameraId());
      const auto &cfw = cam_from_world_map[img_id];
      Vector3d s_cam = cfw * start;
      Vector3d e_cam = cfw * end;
      if (s_cam.z() <= 0 && e_cam.z() <= 0)
        continue;
      auto p1 = cam.ImgFromCam(s_cam);
      auto p2 = cam.ImgFromCam(e_cam);
      if (!p1.has_value() || !p2.has_value())
        continue;
      if (InBounds(*p1, options.camera_width, options.camera_height) ||
          InBounds(*p2, options.camera_width, options.camera_height)) {
        new_vis.push_back({img_id, Line2d(*p1, *p2)});
      }
    }

    if (static_cast<int>(new_vis.size()) < options.min_visible_cameras) {
      all_lines.pop_back();
      line_visible.pop_back();
      continue;
    }
    std::shuffle(new_vis.begin(), new_vis.end(), *colmap::PRNG);
    if (static_cast<int>(new_vis.size()) > options.line_track_length) {
      new_vis.resize(options.line_track_length);
    }
    valid_line_indices.push_back(li);
  }

  // Create a set for quick lookup.
  FlatHashSet<colmap::point3D_t> valid_point_set(valid_point_ids.begin(),
                                                 valid_point_ids.end());
  FlatHashSet<size_t> valid_line_set(valid_line_indices.begin(),
                                     valid_line_indices.end());

  // Delete orphan points from point_recon (generated but not visible from
  // enough cameras and thus not tracked in valid_point_set).
  {
    std::vector<colmap::point3D_t> orphan_ids;
    for (const auto &[pt_id, _] : point_recon.Points3D()) {
      if (valid_point_set.count(pt_id) == 0) {
        orphan_ids.push_back(pt_id);
      }
    }
    for (colmap::point3D_t pt_id : orphan_ids) {
      point_recon.DeletePoint3D(pt_id);
    }
  }

  // =========================================================================
  // Filter wireframe edges and retry if needed
  // =========================================================================
  {
    // Keep only edges where both junction point and line survived filtering.
    std::vector<WireframeEdgeInfo> valid_edges;
    for (const auto &edge : wireframe_edges) {
      if (valid_point_set.count(edge.point_id) &&
          valid_line_set.count(edge.line_idx)) {
        valid_edges.push_back(edge);
      }
    }

    // Retry: if not enough edges, generate more junctions on existing lines.
    constexpr int kMaxRetries = 10;
    int retry = 0;
    while (static_cast<int>(valid_edges.size()) < target_wf_edges &&
           retry < kMaxRetries) {
      retry++;
      int deficit = target_wf_edges - static_cast<int>(valid_edges.size());

      // For each valid wireframe line, we can add more junction points along
      // it.
      for (size_t li : valid_line_indices) {
        if (static_cast<int>(valid_edges.size()) >= target_wf_edges)
          break;
        if (all_lines[li].plane_id < 0)
          continue; // Skip free-floating lines

        const auto &line = all_lines[li].line;
        int plane_id = all_lines[li].plane_id;

        // Generate a random point along this line.
        double t = colmap::RandomUniformReal(0.1, 0.9);
        Vector3d pos = line.start + t * (line.end - line.start);

        // Check visibility.
        std::vector<PointVisibility> vis;
        FlatHashSet<int> sampled;
        int attempts = 0;
        while (static_cast<int>(vis.size()) < pt_candidates &&
               attempts < pt_candidates * 4) {
          const int idx = colmap::RandomUniformInteger(0, num_images - 1);
          ++attempts;
          if (!sampled.insert(idx).second)
            continue;
          colmap::image_t img_id = all_image_ids[idx];
          const auto &cam =
              point_recon.Camera(point_recon.Image(img_id).CameraId());
          const auto &cfw = cam_from_world_map[img_id];
          Vector3d pt_cam = cfw * pos;
          if (pt_cam.z() <= 0)
            continue;
          auto proj = cam.ImgFromCam(pt_cam);
          if (!proj.has_value())
            continue;
          if (InBounds(*proj, options.camera_width, options.camera_height)) {
            vis.push_back({img_id, *proj});
          }
        }

        if (static_cast<int>(vis.size()) < options.min_visible_cameras)
          continue;
        if (static_cast<int>(vis.size()) > options.point_track_length)
          vis.resize(options.point_track_length);

        // Create the new junction point.
        colmap::point3D_t pt_id = point_recon.AddPoint3D(pos, colmap::Track());
        junction_point_to_plane[pt_id] = plane_id;
        point_visible[pt_id] = std::move(vis);
        valid_point_set.insert(pt_id);
        valid_point_ids.push_back(pt_id);
        point_to_plane[pt_id] = plane_id;

        // Add wireframe edge.
        valid_edges.push_back({pt_id, li, 1.0});
      }
    }

    // Shuffle and select exactly target_wf_edges.
    std::shuffle(valid_edges.begin(), valid_edges.end(), *colmap::PRNG);
    if (static_cast<int>(valid_edges.size()) > target_wf_edges) {
      valid_edges.resize(target_wf_edges);
    }

    THROW_CHECK_GE(static_cast<int>(valid_edges.size()), target_wf_edges)
        << "Could not generate enough wireframe edges after retries: got "
        << valid_edges.size() << ", need " << target_wf_edges;

    // Collect junction points referenced by surviving edges.
    FlatHashSet<colmap::point3D_t> used_junction_pts;
    for (const auto &edge : valid_edges) {
      used_junction_pts.insert(edge.point_id);
    }
    // Delete unused junction points.
    for (const auto &[pt_id, _] : junction_point_to_plane) {
      if (used_junction_pts.count(pt_id) == 0 &&
          point_recon.ExistsPoint3D(pt_id)) {
        valid_point_set.erase(pt_id);
        point_visible.erase(pt_id);
        point_recon.DeletePoint3D(pt_id);
      }
    }

    // Check that we didn't exceed point budget.
    THROW_CHECK_LE(static_cast<int>(valid_point_set.size()),
                   options.num_points3D)
        << "Wireframe generation exceeded point budget: have "
        << valid_point_set.size() << " points, budget " << options.num_points3D
        << ". Reduce num_wireframe_edges or increase num_points3D.";

    wireframe_edges = std::move(valid_edges);

    // Collect lines used by wireframe edges.
    FlatHashSet<size_t> wireframe_line_indices;
    for (const auto &edge : wireframe_edges) {
      wireframe_line_indices.insert(edge.line_idx);
    }

    // Trim excess lines to num_lines3D, preserving wireframe lines.
    if (static_cast<int>(valid_line_indices.size()) > options.num_lines3D) {
      std::vector<size_t> wf_lines, non_wf_lines;
      for (size_t li : valid_line_indices) {
        if (wireframe_line_indices.count(li)) {
          wf_lines.push_back(li);
        } else {
          non_wf_lines.push_back(li);
        }
      }
      // Shuffle non-wireframe lines and trim.
      std::shuffle(non_wf_lines.begin(), non_wf_lines.end(), *colmap::PRNG);
      int keep_non_wf =
          std::max(0, options.num_lines3D - static_cast<int>(wf_lines.size()));
      if (static_cast<int>(non_wf_lines.size()) > keep_non_wf) {
        non_wf_lines.resize(keep_non_wf);
      }
      // Rebuild valid_line_indices and valid_line_set.
      valid_line_indices.clear();
      valid_line_set.clear();
      for (size_t li : wf_lines) {
        valid_line_indices.push_back(li);
        valid_line_set.insert(li);
      }
      for (size_t li : non_wf_lines) {
        valid_line_indices.push_back(li);
        valid_line_set.insert(li);
      }
    }
  }

  // Reconcile valid_point_ids after junction point deletions.
  {
    std::vector<colmap::point3D_t> reconciled;
    reconciled.reserve(valid_point_ids.size());
    for (colmap::point3D_t pt_id : valid_point_ids) {
      if (valid_point_set.count(pt_id)) {
        reconciled.push_back(pt_id);
      }
    }
    valid_point_ids = std::move(reconciled);
  }

  // Trim excess points if grid junction points pushed us over the budget.
  // Protect junction points used by wireframe edges.
  if (static_cast<int>(valid_point_ids.size()) > options.num_points3D) {
    int excess =
        static_cast<int>(valid_point_ids.size()) - options.num_points3D;
    // Collect junction points referenced by wireframe edges.
    FlatHashSet<colmap::point3D_t> wf_junction_pts;
    for (const auto &edge : wireframe_edges) {
      wf_junction_pts.insert(edge.point_id);
    }
    // Partition: removable free-floating points vs everything else.
    std::vector<colmap::point3D_t> kept, removable;
    for (colmap::point3D_t pt_id : valid_point_ids) {
      if (point_to_plane.count(pt_id) && point_to_plane[pt_id] == -1 &&
          !wf_junction_pts.count(pt_id)) {
        removable.push_back(pt_id);
      } else {
        kept.push_back(pt_id);
      }
    }
    // Delete excess removable points.
    int to_remove = std::min(excess, static_cast<int>(removable.size()));
    for (int i = 0; i < to_remove; ++i) {
      colmap::point3D_t pt_id = removable.back();
      removable.pop_back();
      valid_point_set.erase(pt_id);
      point_visible.erase(pt_id);
      point_recon.DeletePoint3D(pt_id);
    }
    // Reassemble.
    kept.insert(kept.end(), removable.begin(), removable.end());
    valid_point_ids = std::move(kept);
  }

  // Backfill free-floating points to reach requested count after deletions.
  while (static_cast<int>(valid_point_ids.size()) < options.num_points3D) {
    Vector3d pt = RandomPointInCube(extent);
    colmap::point3D_t pt_id = point_recon.AddPoint3D(pt, colmap::Track());

    std::vector<PointVisibility> vis;
    FlatHashSet<int> sampled_bf;
    int bf_attempts = 0;
    while (static_cast<int>(vis.size()) < pt_candidates &&
           bf_attempts < pt_candidates * 4) {
      const int idx = colmap::RandomUniformInteger(0, num_images - 1);
      ++bf_attempts;
      if (!sampled_bf.insert(idx).second)
        continue;
      colmap::image_t img_id = all_image_ids[idx];
      const auto &cam =
          point_recon.Camera(point_recon.Image(img_id).CameraId());
      const auto &cfw = cam_from_world_map[img_id];
      Vector3d pt_cam = cfw * pt;
      if (pt_cam.z() <= 0)
        continue;
      auto proj = cam.ImgFromCam(pt_cam);
      if (!proj.has_value())
        continue;
      if (InBounds(*proj, options.camera_width, options.camera_height)) {
        vis.push_back({img_id, *proj});
      }
    }

    if (static_cast<int>(vis.size()) < options.min_visible_cameras) {
      point_recon.DeletePoint3D(pt_id);
      continue;
    }
    if (static_cast<int>(vis.size()) > options.point_track_length) {
      vis.resize(options.point_track_length);
    }
    point_visible[pt_id] = std::move(vis);
    point_to_plane[pt_id] = -1;
    valid_point_ids.push_back(pt_id);
    valid_point_set.insert(pt_id);
  }

  // Mapping from original line index → final line3D_t ID (0-based).
  FlatHashMap<size_t, line3D_t> line_idx_to_id;
  {
    line3D_t next_id = 0;
    for (size_t li : valid_line_indices) {
      line_idx_to_id[li] = next_id++;
    }
  }

  // Accumulate per-image 2D observations.
  struct PointObs {
    colmap::point3D_t point3D_id;
    Vector2d pixel;
  };
  struct LineObs {
    size_t orig_line_idx;
    Line2d line2d;
  };
  FlatHashMap<colmap::image_t, std::vector<PointObs>> per_image_points;
  FlatHashMap<colmap::image_t, std::vector<LineObs>> per_image_lines;

  for (colmap::point3D_t pt_id : valid_point_ids) {
    for (const auto &vis : point_visible[pt_id]) {
      per_image_points[vis.image_id].push_back({pt_id, vis.pixel});
    }
  }
  for (size_t li : valid_line_indices) {
    for (const auto &vis : line_visible[li]) {
      per_image_lines[vis.image_id].push_back({li, vis.line2d});
    }
  }

  // Populate reconstruction with 2D observations per image.
  // Track per-image indices for group/wireframe construction.
  FlatHashMap<colmap::image_t,
              FlatHashMap<colmap::point3D_t, colmap::point2D_t>>
      pt_to_p2d_idx;
  FlatHashMap<colmap::image_t, FlatHashMap<size_t, line2D_t>> line_to_l2d_idx;

  // Build Line3d objects (without tracks — we'll fill tracks below).
  FlatHashMap<line3D_t, Line3d> line3d_map;
  for (size_t li : valid_line_indices) {
    line3d_map[line_idx_to_id[li]] = all_lines[li].line;
  }

  // Pre-build plane → points map to avoid O(planes * points) inner loops.
  FlatHashMap<int, std::vector<colmap::point3D_t>> points_by_plane;
  for (colmap::point3D_t pt_id : valid_point_ids) {
    auto it = point_to_plane.find(pt_id);
    if (it != point_to_plane.end() && it->second >= 0) {
      points_by_plane[it->second].push_back(pt_id);
    }
  }

  for (colmap::image_t img_id : all_image_ids) {
    // --- Set Point2D observations on colmap::Image ---
    auto &image = point_recon.Image(img_id);
    std::vector<colmap::Point2D> points2D;
    const auto &pt_obs = per_image_points[img_id];
    points2D.reserve(pt_obs.size());
    for (const auto &obs : pt_obs) {
      colmap::Point2D p2d;
      p2d.xy = obs.pixel;
      p2d.point3D_id = obs.point3D_id;
      colmap::point2D_t p2d_idx =
          static_cast<colmap::point2D_t>(points2D.size());
      points2D.push_back(p2d);
      // Add to point3D track.
      point_recon.Point3D(obs.point3D_id).track.AddElement(img_id, p2d_idx);
      pt_to_p2d_idx[img_id][obs.point3D_id] = p2d_idx;
    }
    image.SetPoints2D(points2D);

    // --- Build Line2d observations ---
    std::vector<Line2d> lines2d;
    const auto &ln_obs = per_image_lines[img_id];
    lines2d.reserve(ln_obs.size());
    for (const auto &obs : ln_obs) {
      line2D_t l2d_idx = static_cast<line2D_t>(lines2d.size());
      lines2d.push_back(obs.line2d);
      // Add to Line3d track.
      line3D_t lid = line_idx_to_id[obs.orig_line_idx];
      line3d_map[lid].track.AddElement(img_id, l2d_idx);
      line_to_l2d_idx[img_id][obs.orig_line_idx] = l2d_idx;
    }

    // =========================================================================
    // Build Group2d observations for this image
    // =========================================================================
    std::vector<Group2d> groups2d;
    const auto &img_pt_idx = pt_to_p2d_idx[img_id];
    const auto &img_ln_idx = line_to_l2d_idx[img_id];

    for (int pi = 0; pi < options.num_planes; ++pi) {
      Group2d g2d(GroupType::PLANE);
      // Collect point2D associations using pre-built plane→points map.
      if (points_by_plane.count(pi)) {
        for (colmap::point3D_t pt_id : points_by_plane.at(pi)) {
          if (img_pt_idx.count(pt_id)) {
            g2d.AddPoint(AssociatedFeature2d(img_pt_idx.at(pt_id), 1.0));
          }
        }
      }
      groups2d.push_back(std::move(g2d));
    }

    // =========================================================================
    // Build Wireframe2d for this image
    // =========================================================================
    Wireframe2d wf2d;
    for (const auto &edge : wireframe_edges) {
      // Check if both the point and line are valid and visible in this image.
      if (valid_point_set.count(edge.point_id) &&
          valid_line_set.count(edge.line_idx)) {
        if (pt_to_p2d_idx[img_id].count(edge.point_id) &&
            line_to_l2d_idx[img_id].count(edge.line_idx)) {
          wf2d.AddEdge(pt_to_p2d_idx[img_id][edge.point_id],
                       line_to_l2d_idx[img_id][edge.line_idx], edge.weight);
        }
      }
    }

    // Create Structure2d (lines, groups, wireframe, num_points).
    Structure2d s2d(lines2d, groups2d, wf2d,
                    static_cast<point2D_t>(points2D.size()));
    structure_recon.AddStructure2D(img_id, s2d);
  }

  // Register all frames (this makes images "registered" in COLMAP).
  for (const auto &[frame_id, _] : point_recon.Frames()) {
    if (point_recon.Frame(frame_id).HasPose()) {
      point_recon.RegisterFrame(frame_id);
    }
  }

  // =========================================================================
  // Add Line3D to structure_recon (sets line2D.line3D_id via track).
  // =========================================================================
  for (auto &[lid, line] : line3d_map) {
    structure_recon.AddLine3D(lid, line);
  }

  // =========================================================================
  // Build Group3d and add to structure_recon.
  // =========================================================================
  for (int pi = 0; pi < options.num_planes; ++pi) {
    Group3d group(GroupType::PLANE, planes[pi].params);

    // Add point associations.
    for (colmap::point3D_t pt_id : valid_point_ids) {
      if (point_to_plane.count(pt_id) && point_to_plane[pt_id] == pi) {
        group.AddPoint(AssociatedFeature3d(pt_id, 1.0));
      }
    }
    // Add line associations.
    for (size_t li : valid_line_indices) {
      if (all_lines[li].plane_id == pi) {
        group.AddLine(AssociatedFeature3d(line_idx_to_id[li], 1.0));
      }
    }

    // Build track: add track elements for each image where the group has
    // observations (group index = pi in every image's groups2d).
    for (colmap::image_t img_id : all_image_ids) {
      const auto &s2d = structure_recon.Structure2d(img_id);
      if (static_cast<int>(s2d.NumGroups()) <= pi)
        continue;
      const auto &g2d = s2d.Group(pi);
      if (!g2d.points.empty() || !g2d.lines.empty()) {
        group.track.AddElement(img_id, static_cast<colmap::point2D_t>(pi));
      }
    }

    structure_recon.AddGroup3D(static_cast<group3D_t>(pi), group);
  }

  // =========================================================================

  // Add 3D wireframe edges to StructureReconstruction.
  // =========================================================================
  for (const auto &edge : wireframe_edges) {
    if (valid_point_set.count(edge.point_id) &&
        valid_line_set.count(edge.line_idx)) {
      structure_recon.Wireframe().AddEdge(
          edge.point_id, line_idx_to_id[edge.line_idx], edge.weight);
    }
  }
}

void SynthesizeStructuredNoise(const SyntheticStructuredNoiseOptions &options,
                               HolisticReconstruction *reconstruction) {
  auto &point_recon = reconstruction->PointRecon();
  auto &structure_recon = reconstruction->StructureRecon();

  // Perturb poses (via frame rig_from_world).
  for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
    colmap::Rigid3d &rig_from_world =
        point_recon.Frame(frame_id).RigFromWorld();

    if (options.pose_rotation_stddev > 0.0) {
      const double angle =
          std::clamp(colmap::RandomGaussian(0.0, options.pose_rotation_stddev),
                     -180.0, 180.0);
      rig_from_world.rotation() *= Eigen::Quaterniond(
          Eigen::AngleAxisd(colmap::DegToRad(angle), Eigen::Vector3d::UnitZ()));
    }

    if (options.pose_translation_stddev > 0.0) {
      rig_from_world.translation() += Vector3d(
          colmap::RandomGaussian(0.0, options.pose_translation_stddev),
          colmap::RandomGaussian(0.0, options.pose_translation_stddev),
          colmap::RandomGaussian(0.0, options.pose_translation_stddev));
    }
  }

  // Perturb 2D point observations.
  if (options.point2D_stddev > 0.0) {
    for (const auto &[image_id, _] : point_recon.Images()) {
      auto &image = point_recon.Image(image_id);
      for (auto &point2D : image.Points2D()) {
        point2D.xy +=
            Vector2d(colmap::RandomGaussian(0.0, options.point2D_stddev),
                     colmap::RandomGaussian(0.0, options.point2D_stddev));
      }
    }
  }

  // Perturb 3D point positions.
  if (options.point3D_stddev > 0.0) {
    for (auto &[pt_id, _] : point_recon.Points3D()) {
      point_recon.Point3D(pt_id).xyz +=
          Vector3d(colmap::RandomGaussian(0.0, options.point3D_stddev),
                   colmap::RandomGaussian(0.0, options.point3D_stddev),
                   colmap::RandomGaussian(0.0, options.point3D_stddev));
    }
  }

  // Perturb 2D line observations.
  if (options.line2D_endpoint_stddev > 0.0) {
    for (auto &[img_id, s2d] : structure_recon.Structures2d()) {
      for (auto &line2d : s2d.Lines()) {
        line2d.start += Vector2d(
            colmap::RandomGaussian(0.0, options.line2D_endpoint_stddev),
            colmap::RandomGaussian(0.0, options.line2D_endpoint_stddev));
        line2d.end += Vector2d(
            colmap::RandomGaussian(0.0, options.line2D_endpoint_stddev),
            colmap::RandomGaussian(0.0, options.line2D_endpoint_stddev));
      }
    }
  }

  // Perturb 3D line endpoints.
  if (options.line3D_endpoint_stddev > 0.0) {
    for (auto &[lid, line] : structure_recon.Lines3D()) {
      line.start +=
          Vector3d(colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev),
                   colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev),
                   colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev));
      line.end +=
          Vector3d(colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev),
                   colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev),
                   colmap::RandomGaussian(0.0, options.line3D_endpoint_stddev));
    }
  }

  // Perturb plane parameters.
  if (options.plane_normal_stddev > 0.0 || options.plane_offset_stddev > 0.0) {
    for (auto &[gid, group] : structure_recon.Groups3D()) {
      if (group.type != GroupType::PLANE)
        continue;
      auto &params = group.GetParams();
      if (params.size() < 4)
        continue;

      if (options.plane_normal_stddev > 0.0) {
        // Perturb normal by small random rotation.
        Vector3d normal(params[0], params[1], params[2]);
        Vector3d axis = normal.cross(Eigen::Vector3d::Random()).normalized();
        double angle = colmap::RandomGaussian(0.0, options.plane_normal_stddev);
        normal = Eigen::AngleAxisd(angle, axis) * normal;
        normal.normalize();
        params[0] = normal.x();
        params[1] = normal.y();
        params[2] = normal.z();
      }

      if (options.plane_offset_stddev > 0.0) {
        params[3] += colmap::RandomGaussian(0.0, options.plane_offset_stddev);
      }

      group.NormalizeParams();
    }
  }
}

} // namespace estimators
} // namespace limap
