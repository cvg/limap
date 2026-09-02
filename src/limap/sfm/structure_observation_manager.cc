#include "limap/sfm/structure_observation_manager.h"

#include <algorithm>
#include <cmath>

#include <colmap/scene/image.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/line_metrics.h"
#include "limap/geometry/line_pixel_uncertainty.h"
#include "limap/geometry/minimal_inf_line3d.h"
#include "limap/util/types.h"

namespace limap {

// ============ Line CRUD Operations (delegate to StructureReconstruction) =====

line3D_t StructureObservationManager::AddLine3D(const Line3d &line3d) {
  return srec_.AddLine3D(line3d);
}

void StructureObservationManager::AddLineObservation(
    line3D_t line3D_id, const colmap::TrackElement &track_el) {
  srec_.AddLineObservation(line3D_id, track_el);
}

void StructureObservationManager::DeleteLine3D(line3D_t line3D_id) {
  if (!srec_.ExistsLine3D(line3D_id)) {
    return;
  }
  srec_.DeleteLine3D(line3D_id);
}

void StructureObservationManager::DeleteLineObservation(
    colmap::image_t image_id, line2D_t line2D_idx) {
  if (!srec_.ExistsStructure2D(image_id)) {
    return;
  }
  Structure2d &S = srec_.Structure2d(image_id);
  if (line2D_idx >= static_cast<line2D_t>(S.NumLines())) {
    return;
  }
  if (!S.Line(line2D_idx).HasLine3D()) {
    return;
  }
  srec_.DeleteLineObservation(image_id, line2D_idx);
}

line3D_t StructureObservationManager::MergeLines3D(line3D_t line3D_id1,
                                                   line3D_t line3D_id2) {
  return srec_.MergeLines3D(line3D_id1, line3D_id2);
}

// ============ Group CRUD Operations (delegate to StructureReconstruction) ====

group3D_t StructureObservationManager::AddGroup3D(const Group3d &group3d) {
  return srec_.AddGroup3D(group3d);
}

void StructureObservationManager::AddGroupObservation(
    group3D_t group3D_id, const colmap::TrackElement &track_el) {
  srec_.AddGroupObservation(group3D_id, track_el);
}

void StructureObservationManager::DeleteGroup3D(group3D_t group3D_id) {
  if (!srec_.ExistsGroup3D(group3D_id)) {
    return;
  }
  srec_.DeleteGroup3D(group3D_id);
}

void StructureObservationManager::DeleteGroupObservation(
    colmap::image_t image_id, group2D_t group2D_idx) {
  if (!srec_.ExistsStructure2D(image_id)) {
    return;
  }
  Structure2d &S = srec_.Structure2d(image_id);
  if (group2D_idx >= static_cast<group2D_t>(S.NumGroups())) {
    return;
  }
  if (S.Group(group2D_idx).group3D_id == kInvalidGroup3dId) {
    return;
  }
  srec_.DeleteGroupObservation(image_id, group2D_idx);
}

group3D_t StructureObservationManager::MergeGroups3D(group3D_t group3D_id1,
                                                     group3D_t group3D_id2) {
  return srec_.MergeGroups3D(group3D_id1, group3D_id2);
}

// ============ Frame De-registration
// ===========================================

void StructureObservationManager::DeRegisterFrame(
    const colmap::frame_t frame_id) {
  const auto &point_recon = srec_.PointRecon();
  const auto &frame = point_recon.Frame(frame_id);

  for (const auto &data_id : frame.ImageIds()) {
    const colmap::image_t image_id = data_id.id;
    if (!srec_.ExistsStructure2D(image_id)) {
      continue;
    }
    const auto &s2d = srec_.Structure2d(image_id);

    // Delete line observations
    for (line2D_t i = 0; i < static_cast<line2D_t>(s2d.NumLines()); ++i) {
      if (s2d.Line(i).HasLine3D()) {
        srec_.DeleteLineObservation(image_id, i);
      }
    }

    // Delete group observations
    for (group2D_t i = 0; i < static_cast<group2D_t>(s2d.NumGroups()); ++i) {
      if (s2d.Group(i).group3D_id != kInvalidGroup3dId) {
        srec_.DeleteGroupObservation(image_id, i);
      }
    }
  }
}

// ============ Filtering Operations ===========================================

void StructureObservationManager::FilterLines3dByReprojection(
    const double th_angular_2d, const double th_perp_2d,
    const int num_outliers) {
  NodeHashMap<line3D_t, Line3dWithActiveLabels> new_lines3d;
  auto &lines3d = srec_.Lines3D();
  new_lines3d.reserve(lines3d.size());

  for (const auto &kv : lines3d) {
    const line3D_t line_id = kv.first;
    const Line3d &line3d = kv.second;

    const std::vector<colmap::TrackElement> &elements = line3d.track.Elements();
    if (elements.empty()) {
      continue;
    }

    // Keep only supports that pass reprojection checks.
    std::vector<colmap::Image> kept_images;
    std::vector<Line2d> kept_lines2d;
    kept_images.reserve(elements.size());
    kept_lines2d.reserve(elements.size());

    colmap::Track new_track;

    for (const colmap::TrackElement &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const colmap::point2D_t feat_idx = elem.point2D_idx;
      const line2D_t line2d_id = static_cast<line2D_t>(feat_idx);

      const Structure2d &S2 = srec_.Structure2d(image_id);
      const Line2d &line2d = S2.Line(line2d_id);

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        continue;
      }
      const Line2d proj = res.value();

      const double angle = ComputeAngle<Line2d>(proj, line2d);
      if (angle > th_angular_2d) {
        continue;
      }

      const double dist =
          EndpointsPerpendicularDistanceOneway<Line2d>(proj, line2d);
      if (dist > th_perp_2d) {
        continue;
      }

      kept_images.push_back(image);
      kept_lines2d.push_back(line2d);
      new_track.AddElement(elem);
    }

    // Not enough supports left.
    if (new_track.Length() < 2) {
      continue;
    }

    // Re-estimate a finite segment along the original infinite 3D line.
    // The infinite line is defined by the current segment direction.
    Line3d updated_line = line3d;
    if (kept_images.size() > static_cast<size_t>(num_outliers) * 2) {
      InfiniteLine3d inf_line(line3d);
      Line3d updated_line = GetLineSegmentFromInfiniteLine3d(
          inf_line, kept_images, kept_lines2d, num_outliers);
    }

    // Keep only the filtered supports in the track.
    updated_line.track = new_track;
    new_lines3d.emplace(line_id, std::move(updated_line));
  }

  lines3d = std::move(new_lines3d);
}

void StructureObservationManager::FilterLines3dBySensitivity(
    const double th_sensitivity_3d, const int min_support_ns) {
  NodeHashMap<line3D_t, Line3dWithActiveLabels> new_lines3d;
  auto &lines3d = srec_.Lines3D();
  new_lines3d.reserve(lines3d.size());

  for (const auto &kv : lines3d) {
    const line3D_t line_id = kv.first;
    const Line3d &line3d = kv.second;

    const std::vector<colmap::TrackElement> &elements = line3d.track.Elements();
    if (elements.empty()) {
      continue;
    }

    FlatHashSet<colmap::image_t> supporting_views;

    for (const colmap::TrackElement &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const colmap::Image &image = srec_.PointRecon().Image(image_id);

      const double sensitivity = line3d.Sensitivity(image);
      if (sensitivity <= th_sensitivity_3d) {
        supporting_views.insert(image_id);
      }
    }

    if (static_cast<int>(supporting_views.size()) >= min_support_ns) {
      new_lines3d.emplace(line_id, line3d);
    }
  }

  lines3d = std::move(new_lines3d);
}

void StructureObservationManager::FilterLines3dByOverlap(
    const double th_overlap, const int min_support_ns) {
  NodeHashMap<line3D_t, Line3dWithActiveLabels> new_lines3d;
  auto &lines3d = srec_.Lines3D();
  new_lines3d.reserve(lines3d.size());

  for (const auto &kv : lines3d) {
    const line3D_t line_id = kv.first;
    const Line3d &line3d = kv.second;

    const std::vector<colmap::TrackElement> &elements = line3d.track.Elements();
    if (elements.empty()) {
      continue;
    }

    FlatHashSet<colmap::image_t> supporting_views;

    for (const colmap::TrackElement &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const colmap::point2D_t feat_idx = elem.point2D_idx;
      const line2D_t line2d_id = static_cast<line2D_t>(feat_idx);

      const Structure2d &S2 = srec_.Structure2d(image_id);
      const Line2d &line2d = S2.Line(line2d_id);

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        continue;
      }
      const Line2d proj = res.value();

      const double overlap = ComputeOverlap<Line2d>(proj, line2d);
      if (overlap >= th_overlap) {
        supporting_views.insert(image_id);
      }
    }

    if (static_cast<int>(supporting_views.size()) >= min_support_ns) {
      new_lines3d.emplace(line_id, line3d);
    }
  }

  lines3d = std::move(new_lines3d);
}

void StructureObservationManager::FilterLines3dByMinVisibleViews(
    const int min_visible_views) {
  NodeHashMap<line3D_t, Line3dWithActiveLabels> new_lines3d;
  auto &lines3d = srec_.Lines3D();
  new_lines3d.reserve(lines3d.size());

  for (const auto &kv : lines3d) {
    const line3D_t line_id = kv.first;
    const Line3d &line3d = kv.second;

    if (static_cast<int>(line3d.track.Length()) >= min_visible_views) {
      new_lines3d.emplace(line_id, line3d);
    }
  }

  lines3d = std::move(new_lines3d);
}

size_t
StructureObservationManager::FilterAllLines3D(const double max_angular_error,
                                              const double max_perp_error) {
  size_t num_filtered = 0;

  // Collect line IDs first (we'll be modifying lines3d during iteration)
  std::vector<line3D_t> line_ids;
  line_ids.reserve(srec_.Lines3D().size());
  for (const auto &[line_id, _] : srec_.Lines3D()) {
    line_ids.push_back(line_id);
  }

  for (const line3D_t line_id : line_ids) {
    if (!srec_.ExistsLine3D(line_id)) {
      continue; // may have been deleted by a previous DeleteLineObservation
    }
    const Line3d &line3d = srec_.Line(line_id);
    const auto &elements = line3d.track.Elements();

    // Collect bad observations
    std::vector<std::pair<colmap::image_t, line2D_t>> bad_obs;
    for (const auto &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const line2D_t line2d_id = static_cast<line2D_t>(elem.point2D_idx);

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        bad_obs.emplace_back(image_id, line2d_id);
        continue;
      }
      const Line2d proj = res.value();
      const Line2d &line2d = srec_.Structure2d(image_id).Line(line2d_id);

      const double angle = ComputeAngle<Line2d>(proj, line2d);
      if (angle > max_angular_error) {
        bad_obs.emplace_back(image_id, line2d_id);
        continue;
      }

      const double dist =
          EndpointsPerpendicularDistanceOneway<Line2d>(proj, line2d);
      if (dist > max_perp_error) {
        bad_obs.emplace_back(image_id, line2d_id);
        continue;
      }
    }

    // Delete bad observations (DeleteLineObservation also deletes line if
    // track becomes < 2)
    for (const auto &[image_id, line2d_id] : bad_obs) {
      DeleteLineObservation(image_id, line2d_id);
      ++num_filtered;
    }
  }

  return num_filtered;
}

// ============ Active/Inactive Operations (soft filtering) ==================

bool StructureObservationManager::IsReliableTrack(line3D_t line3D_id,
                                                  size_t min_active) const {
  if (!srec_.ExistsLine3D(line3D_id)) {
    return false;
  }
  return srec_.Line(line3D_id).IsReliable(min_active);
}

void StructureObservationManager::ClassifyLineTracks(
    size_t min_active_observations, FlatHashSet<line3D_t> &reliable_ids,
    FlatHashSet<line3D_t> &unreliable_ids) const {
  reliable_ids.clear();
  unreliable_ids.clear();
  for (const auto &[line_id, line3d] : srec_.Lines3D()) {
    if (line3d.IsReliable(min_active_observations)) {
      reliable_ids.insert(line_id);
    } else {
      unreliable_ids.insert(line_id);
    }
  }
}

void StructureObservationManager::ClassifyLineTracks(
    size_t min_active_observations, double pixel_uncertainty_threshold,
    const ceres::LossFunction *loss_function,
    FlatHashSet<line3D_t> &reliable_ids,
    FlatHashSet<line3D_t> &unreliable_ids) const {
  // First pass: observation count check (cheap)
  ClassifyLineTracks(min_active_observations, reliable_ids, unreliable_ids);

  if (pixel_uncertainty_threshold <= 0 || reliable_ids.empty()) {
    return;
  }

  // Collect count-reliable line IDs for parallel uncertainty computation
  std::vector<line3D_t> candidates(reliable_ids.begin(), reliable_ids.end());

  std::vector<double> uncertainties(candidates.size(), -1.0);

  colmap::ThreadPool pool(colmap::GetEffectiveNumThreads(-1));
  for (size_t i = 0; i < candidates.size(); ++i) {
    pool.AddTask([&, i]() {
      const line3D_t line_id = candidates[i];
      const auto &line3d = srec_.Lines3D().at(line_id);
      const auto &elements = line3d.track.Elements();

      // Gather per-observation data
      std::vector<Eigen::Quaterniond> rotations;
      std::vector<Eigen::Vector3d> translations;
      std::vector<Eigen::Vector4d> kvecs;
      std::vector<Line2d> lines2d;
      rotations.reserve(elements.size());
      translations.reserve(elements.size());
      kvecs.reserve(elements.size());
      lines2d.reserve(elements.size());

      for (const auto &elem : elements) {
        const colmap::image_t image_id = elem.image_id;
        // Skip inactive observations — only build the information matrix
        // from trusted observations, consistent with CountActiveObservations()
        if (!line3d.IsObservationActive(image_id)) {
          continue;
        }
        const line2D_t line2d_id = static_cast<line2D_t>(elem.point2D_idx);
        const colmap::Image &image = srec_.PointRecon().Image(image_id);
        const colmap::Camera &camera =
            srec_.PointRecon().Camera(image.CameraId());

        const auto &cam_from_world = image.CamFromWorld();
        rotations.push_back(Eigen::Quaterniond(cam_from_world.rotation()));
        translations.push_back(Eigen::Vector3d(cam_from_world.translation()));

        Eigen::Vector4d kvec;
        ParamsToKvec<double>(camera.model_id, camera.params.data(),
                             kvec.data());
        kvecs.push_back(kvec);

        lines2d.push_back(srec_.Structure2d(image_id).Line(line2d_id));
      }

      // Convert to MinimalInfiniteLine3d for params
      MinimalInfiniteLine3d minimal(line3d);

      uncertainties[i] = ComputeLinePixelUncertainty(
          minimal.data.data(), line3d, rotations, translations, kvecs, lines2d,
          loss_function);
    });
  }
  pool.Wait();

  // Log variance distribution (histogram + percentiles)
  {
    std::vector<double> valid_uncertainties;
    size_t num_inf = 0, num_neg = 0;
    for (size_t i = 0; i < uncertainties.size(); ++i) {
      if (uncertainties[i] < 0) {
        ++num_neg;
      } else if (std::isinf(uncertainties[i])) {
        ++num_inf;
      } else {
        valid_uncertainties.push_back(uncertainties[i]);
      }
    }
    std::sort(valid_uncertainties.begin(), valid_uncertainties.end());
    const size_t n = valid_uncertainties.size();
    if (n > 0) {
      auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * (n - 1));
        return valid_uncertainties[std::min(idx, n - 1)];
      };
      // Histogram: count lines in each bucket
      size_t b0_1 = 0, b1_5 = 0, b5_10 = 0, b10_20 = 0, b20_50 = 0, b50_100 = 0,
             b100_500 = 0, b500 = 0;
      for (double v : valid_uncertainties) {
        if (v <= 1)
          ++b0_1;
        else if (v <= 5)
          ++b1_5;
        else if (v <= 10)
          ++b5_10;
        else if (v <= 20)
          ++b10_20;
        else if (v <= 50)
          ++b20_50;
        else if (v <= 100)
          ++b50_100;
        else if (v <= 500)
          ++b100_500;
        else
          ++b500;
      }
      LOG(INFO) << "  PixelUncertainty distribution (N=" << n
                << ", inf=" << num_inf << ", invalid=" << num_neg
                << "):" << " min=" << valid_uncertainties.front()
                << " p25=" << pct(0.25) << " p50=" << pct(0.50)
                << " p75=" << pct(0.75) << " p90=" << pct(0.90)
                << " p95=" << pct(0.95)
                << " max=" << valid_uncertainties.back();
      LOG(INFO) << "  PixelUncertainty histogram:" << " [0,1]=" << b0_1
                << " (1,5]=" << b1_5 << " (5,10]=" << b5_10
                << " (10,20]=" << b10_20 << " (20,50]=" << b20_50
                << " (50,100]=" << b50_100 << " (100,500]=" << b100_500
                << " (500,inf)=" << b500;
    }
  }

  // Second pass: demote high-uncertainty lines
  size_t num_demoted = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (uncertainties[i] > pixel_uncertainty_threshold) {
      reliable_ids.erase(candidates[i]);
      unreliable_ids.insert(candidates[i]);
      ++num_demoted;
    }
  }

  LOG(INFO) << "  ClassifyLineTracks: " << reliable_ids.size() << " reliable, "
            << unreliable_ids.size() << " unreliable (" << num_demoted
            << " demoted by pixel uncertainty > " << pixel_uncertainty_threshold
            << ")";
}

size_t StructureObservationManager::UpdateLineObservationActivity(
    const double max_angular_error, const double max_perp_error) {
  size_t num_changed = 0;

  for (auto &[line_id, line3d] : srec_.Lines3D()) {
    const auto &elements = line3d.track.Elements();
    for (const auto &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const line2D_t line2d_id = static_cast<line2D_t>(elem.point2D_idx);

      bool is_good = true;

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        is_good = false;
      } else {
        const Line2d proj = res.value();
        const Line2d &line2d = srec_.Structure2d(image_id).Line(line2d_id);

        const double angle = ComputeAngle<Line2d>(proj, line2d);
        if (angle > max_angular_error) {
          is_good = false;
        } else {
          const double dist =
              EndpointsPerpendicularDistanceOneway<Line2d>(proj, line2d);
          if (dist > max_perp_error) {
            is_good = false;
          }
        }
      }

      const bool was_active = line3d.IsObservationActive(image_id);
      if (is_good && !was_active) {
        // Observation passes now but was inactive — don't reactivate here.
        // Reactivation is done in UpdateActiveSupports after BA.
      } else if (!is_good && was_active) {
        line3d.SetObservationInactive(image_id);
        ++num_changed;
      }
    }
  }

  return num_changed;
}

size_t StructureObservationManager::UpdateActiveSupports(
    const double max_angular_error, const double max_perp_error) {
  size_t num_changed = 0;

  for (auto &[line_id, line3d] : srec_.Lines3D()) {
    const auto &elements = line3d.track.Elements();
    for (const auto &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const line2D_t line2d_id = static_cast<line2D_t>(elem.point2D_idx);

      bool is_good = true;

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        is_good = false;
      } else {
        const Line2d proj = res.value();
        const Line2d &line2d = srec_.Structure2d(image_id).Line(line2d_id);

        const double angle = ComputeAngle<Line2d>(proj, line2d);
        if (angle > max_angular_error) {
          is_good = false;
        } else {
          const double dist =
              EndpointsPerpendicularDistanceOneway<Line2d>(proj, line2d);
          if (dist > max_perp_error) {
            is_good = false;
          }
        }
      }

      const bool was_active = line3d.IsObservationActive(image_id);
      if (is_good && !was_active) {
        line3d.SetObservationActive(image_id);
        ++num_changed;
      } else if (!is_good && was_active) {
        line3d.SetObservationInactive(image_id);
        ++num_changed;
      }
    }
  }

  return num_changed;
}

size_t StructureObservationManager::FilterLineTracks(
    const double max_angular_error, const double max_perp_error,
    const double min_active_ratio_for_deletion,
    const FlatHashSet<colmap::image_t> *local_image_ids) {
  // Collect line IDs to filter (scoped to local images if provided).
  // Uses per-image iteration (like COLMAP's FilterPoints3DInImages) instead
  // of scanning all lines — O(local_images × lines_per_image) vs
  // O(total_lines × track_length).
  std::vector<line3D_t> line_ids;
  if (local_image_ids) {
    FlatHashSet<line3D_t> line_id_set;
    for (const colmap::image_t image_id : *local_image_ids) {
      if (!srec_.ExistsStructure2D(image_id)) {
        continue;
      }
      const auto &S = srec_.Structure2d(image_id);
      for (size_t i = 0; i < S.NumLines(); ++i) {
        const Line2d &line2d = S.Line(i);
        if (line2d.HasLine3D()) {
          line_id_set.insert(line2d.line3D_id);
        }
      }
    }
    line_ids.assign(line_id_set.begin(), line_id_set.end());
  } else {
    line_ids.reserve(srec_.Lines3D().size());
    for (const auto &[line_id, _] : srec_.Lines3D()) {
      line_ids.push_back(line_id);
    }
  }

  for (const line3D_t line_id : line_ids) {
    if (!srec_.ExistsLine3D(line_id)) {
      continue;
    }
    auto &line3d = srec_.Line(line_id);
    const auto &elements = line3d.track.Elements();

    std::set<colmap::image_t> active_img_ids;
    std::vector<std::pair<colmap::image_t, line2D_t>> inactive_obs;

    // Evaluate all observations (can both activate and deactivate)
    for (const auto &elem : elements) {
      const colmap::image_t image_id = elem.image_id;
      const line2D_t line2d_id = static_cast<line2D_t>(elem.point2D_idx);

      bool is_good = true;

      const colmap::Image &image = srec_.PointRecon().Image(image_id);
      auto res = line3d.Projection(image);
      if (!res.has_value()) {
        is_good = false;
      } else {
        const Line2d proj = res.value();
        const Line2d &line2d = srec_.Structure2d(image_id).Line(line2d_id);

        const double angle = ComputeAngle<Line2d>(proj, line2d);
        if (angle > max_angular_error) {
          is_good = false;
        } else {
          const double dist =
              EndpointsPerpendicularDistanceOneway<Line2d>(proj, line2d);
          if (dist > max_perp_error) {
            is_good = false;
          }
        }
      }

      const bool was_active = line3d.IsObservationActive(image_id);
      if (is_good) {
        if (!was_active) {
          line3d.SetObservationActive(image_id);
        }
        active_img_ids.insert(image_id);
      } else {
        if (was_active) {
          line3d.SetObservationInactive(image_id);
        }
        inactive_obs.emplace_back(image_id, line2d_id);
      }
    }

    // Hard-delete individual inactive observations from tracks with >10 active
    // images. Keeps large tracks clean without over-pruning small tracks.
    if (active_img_ids.size() > 10 && !inactive_obs.empty()) {
      for (const auto &[image_id, line2d_id] : inactive_obs) {
        DeleteLineObservation(image_id, line2d_id);
      }

      // Recompute the 3D line segment from remaining observations
      if (srec_.ExistsLine3D(line_id)) {
        auto &updated = srec_.Line(line_id);
        const auto &remaining = updated.track.Elements();
        if (remaining.size() >= 2) {
          std::vector<colmap::Image> images;
          std::vector<Line2d> lines2d;
          images.reserve(remaining.size());
          lines2d.reserve(remaining.size());
          for (const auto &elem : remaining) {
            images.push_back(srec_.PointRecon().Image(elem.image_id));
            lines2d.push_back(
                srec_.Structure2d(elem.image_id)
                    .Line(static_cast<line2D_t>(elem.point2D_idx)));
          }
          InfiniteLine3d inf_line(updated);
          Line3d recomputed = GetLineSegmentFromInfiniteLine3d(
              inf_line, images, lines2d, /*num_outliers=*/2);
          updated.start = recomputed.start;
          updated.end = recomputed.end;
        }
      }
    }
  }

  // Delete entire tracks that are hopeless (scoped to the filtered lines)
  size_t num_deleted = 0;
  for (const line3D_t line_id : line_ids) {
    if (!srec_.ExistsLine3D(line_id)) {
      continue;
    }
    const auto &line3d = srec_.Line(line_id);
    if (line3d.track.Length() < 2) {
      srec_.DeleteLine3D(line_id);
      ++num_deleted;
    } else if (min_active_ratio_for_deletion > 0 &&
               line3d.ActiveRatio() < min_active_ratio_for_deletion) {
      srec_.DeleteLine3D(line_id);
      ++num_deleted;
    }
  }
  return num_deleted;
}

size_t StructureObservationManager::DeleteSupportlessLineTracks(
    const double min_active_ratio) {
  size_t num_deleted = 0;

  std::vector<line3D_t> to_delete;
  for (const auto &[line_id, line3d] : srec_.Lines3D()) {
    // Always delete tracks with <2 observations
    if (line3d.track.Length() < 2) {
      to_delete.push_back(line_id);
    } else if (min_active_ratio > 0 &&
               line3d.ActiveRatio() < min_active_ratio) {
      to_delete.push_back(line_id);
    }
  }

  for (const line3D_t line_id : to_delete) {
    if (srec_.ExistsLine3D(line_id)) {
      srec_.DeleteLine3D(line_id);
      ++num_deleted;
    }
  }

  return num_deleted;
}

} // namespace limap
