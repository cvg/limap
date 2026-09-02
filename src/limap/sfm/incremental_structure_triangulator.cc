#include "limap/sfm/incremental_structure_triangulator.h"

#include <colmap/util/logging.h>

#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/sfm/group_verification.h"

namespace limap {

bool IncrementalStructureTriangulator::Options::Check() const {
  if (triangulate_points && !point_options.Check()) {
    return false;
  }
  if (triangulate_lines && !line_options.Check()) {
    return false;
  }
  if (triangulate_groups && !group_options.Check()) {
    return false;
  }
  return true;
}

IncrementalStructureTriangulator::IncrementalStructureTriangulator(
    std::shared_ptr<const colmap::CorrespondenceGraph> point_corr_graph,
    std::shared_ptr<const StructureCorrespondenceGraph> structure_corr_graph,
    HolisticReconstruction &reconstruction,
    std::shared_ptr<colmap::ObservationManager> point_obs_manager)
    : reconstruction_(reconstruction),
      point_corr_graph_(std::move(point_corr_graph)) {

  // Create shared observation manager
  obs_manager_ = std::make_shared<StructureObservationManager>(
      reconstruction_.StructureRecon());

  // Create point triangulator (uses COLMAP's implementation).
  // When point_obs_manager is provided, the triangulator updates pair
  // statistics (num_tri_corrs) on point creation/modification.
  point_triangulator_ = std::make_unique<colmap::IncrementalTriangulator>(
      point_corr_graph_, reconstruction_.PointRecon(),
      std::move(point_obs_manager));

  // Create line triangulator with shared observation manager
  line_triangulator_ = std::make_unique<IncrementalLineTriangulator>(
      structure_corr_graph, reconstruction_, obs_manager_);

  // Create group triangulator with shared observation manager
  group_triangulator_ = std::make_unique<IncrementalGroupTriangulator>(
      structure_corr_graph, reconstruction_, obs_manager_);
}

size_t
IncrementalStructureTriangulator::TriangulateImage(const Options &options,
                                                   colmap::image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  // Triangulate in order: points -> lines -> groups
  // (Groups depend on triangulated points and lines)

  if (options.triangulate_points) {
    num_tris +=
        point_triangulator_->TriangulateImage(options.point_options, image_id);
  }

  if (options.triangulate_lines) {
    num_tris +=
        line_triangulator_->TriangulateImage(options.line_options, image_id);
  }

  if (options.triangulate_groups) {
    num_tris +=
        group_triangulator_->TriangulateImage(options.group_options, image_id);

    // Update feature associations and re-estimate params for modified groups
    const auto &modified_groups = group_triangulator_->GetModifiedGroups3D();
    if (!modified_groups.empty()) {
      group_triangulator_->UpdateGroupAssociations(options.group_options,
                                                   modified_groups);
      group_triangulator_->ClearModifiedGroups3D();
    }
  }

  return num_tris;
}

size_t
IncrementalStructureTriangulator::CompleteImage(const Options &options,
                                                colmap::image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_completed = 0;

  if (options.triangulate_points) {
    num_completed +=
        point_triangulator_->CompleteImage(options.point_options, image_id);
  }

  if (options.triangulate_lines) {
    num_completed +=
        line_triangulator_->CompleteImage(options.line_options, image_id);
  }

  if (options.triangulate_groups) {
    num_completed +=
        group_triangulator_->CompleteImage(options.group_options, image_id);
  }

  return num_completed;
}

size_t
IncrementalStructureTriangulator::CompleteAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  size_t num_completed = 0;

  if (options.triangulate_points) {
    num_completed +=
        point_triangulator_->CompleteAllTracks(options.point_options);
  }

  if (options.triangulate_lines) {
    num_completed +=
        line_triangulator_->CompleteAllTracks(options.line_options);
  }

  if (options.triangulate_groups) {
    num_completed +=
        group_triangulator_->CompleteAllTracks(options.group_options);
  }

  return num_completed;
}

size_t
IncrementalStructureTriangulator::MergeAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  size_t num_merged = 0;

  if (options.triangulate_points) {
    num_merged += point_triangulator_->MergeAllTracks(options.point_options);
  }

  if (options.triangulate_lines) {
    num_merged += line_triangulator_->MergeAllTracks(options.line_options);
  }

  if (options.triangulate_groups) {
    num_merged += group_triangulator_->MergeAllTracks(options.group_options);
  }

  return num_merged;
}

colmap::IncrementalTriangulator &
IncrementalStructureTriangulator::PointTriangulator() {
  return *point_triangulator_;
}

IncrementalLineTriangulator &
IncrementalStructureTriangulator::LineTriangulator() {
  return *line_triangulator_;
}

IncrementalGroupTriangulator &
IncrementalStructureTriangulator::GroupTriangulator() {
  return *group_triangulator_;
}

const colmap::IncrementalTriangulator &
IncrementalStructureTriangulator::PointTriangulator() const {
  return *point_triangulator_;
}

const IncrementalLineTriangulator &
IncrementalStructureTriangulator::LineTriangulator() const {
  return *line_triangulator_;
}

const IncrementalGroupTriangulator &
IncrementalStructureTriangulator::GroupTriangulator() const {
  return *group_triangulator_;
}

FlatHashSet<colmap::point3D_t>
IncrementalStructureTriangulator::GetModifiedPoints3D() {
  return point_triangulator_->GetModifiedPoints3D();
}

FlatHashSet<line3D_t> IncrementalStructureTriangulator::GetModifiedLines3D() {
  return line_triangulator_->GetModifiedLines3D();
}

FlatHashSet<group3D_t> IncrementalStructureTriangulator::GetModifiedGroups3D() {
  return group_triangulator_->GetModifiedGroups3D();
}

void IncrementalStructureTriangulator::ClearModified() {
  point_triangulator_->ClearModifiedPoints3D();
  line_triangulator_->ClearModifiedLines3D();
  group_triangulator_->ClearModifiedGroups3D();
}

void IncrementalTriangulateStructure(
    std::shared_ptr<const colmap::CorrespondenceGraph> point_corr_graph,
    const StructureDatabaseCache &structure_db_cache,
    HolisticReconstruction &reconstruction,
    const IncrementalStructureTriangulator::Options &options,
    const estimators::StructureBundleAdjustmentOptions *ba_options,
    const GroupVerificationOptions *filter_options) {
  THROW_CHECK(options.Check());

  reconstruction.StructureRecon().Load(structure_db_cache);
  reconstruction.StructureRecon().InitializeAllWireframes();

  auto structure_corr_graph = structure_db_cache.StructureCorrespondenceGraph();

  IncrementalStructureTriangulator triangulator(
      std::move(point_corr_graph), structure_corr_graph, reconstruction);

  auto &point_recon = reconstruction.PointRecon();
  std::vector<colmap::image_t> image_ids;
  image_ids.reserve(point_recon.NumImages());
  for (const auto &[image_id, image] : point_recon.Images()) {
    if (image.HasPose()) {
      image_ids.push_back(image_id);
    }
  }
  std::sort(image_ids.begin(), image_ids.end());

  LOG(INFO) << "Triangulating " << image_ids.size()
            << " images incrementally...";

  size_t total_tris = 0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    total_tris += triangulator.TriangulateImage(options, image_ids[i]);
    if ((i + 1) % 10 == 0 || i == image_ids.size() - 1) {
      LOG(INFO) << "  Processed " << (i + 1) << "/" << image_ids.size()
                << " images, " << total_tris << " triangulations";
    }
  }

  if (options.complete_tracks) {
    LOG(INFO) << "Completing tracks...";
    const size_t num_completed = triangulator.CompleteAllTracks(options);
    LOG(INFO) << "  Completed " << num_completed << " observations";
  }

  if (options.merge_tracks) {
    LOG(INFO) << "Merging tracks...";
    const size_t num_merged = triangulator.MergeAllTracks(options);
    LOG(INFO) << "  Merged " << num_merged << " observations";
  }

  // Final update of group associations after Complete/Merge
  if (options.triangulate_groups) {
    const auto &modified_groups =
        triangulator.GroupTriangulator().GetModifiedGroups3D();
    if (!modified_groups.empty()) {
      LOG(INFO) << "Updating associations for " << modified_groups.size()
                << " groups modified during Complete/Merge...";
      const size_t num_updated =
          triangulator.GroupTriangulator().UpdateGroupAssociations(
              options.group_options, modified_groups);
      triangulator.GroupTriangulator().ClearModifiedGroups3D();
      LOG(INFO) << "  Updated " << num_updated << " groups";
    }
  }

  const auto &srec = reconstruction.StructureRecon();
  LOG(INFO) << "Incremental structure triangulation complete: "
            << point_recon.NumPoints3D() << " points, " << srec.NumLines3D()
            << " lines, " << srec.NumGroups3D() << " groups";

  if (ba_options) {
    LOG(INFO) << "Running bundle adjustment...";
    // During triangulation we only refine structure, never cameras/rigs.
    auto opts = *ba_options;
    opts.refine_focal_length = false;
    opts.refine_principal_point = false;
    opts.refine_extra_params = false;
    opts.refine_sensor_from_rig = false;
    opts.refine_rig_from_world = false;

    estimators::StructureBundleAdjustmentConfig ba_config;
    for (const auto &[image_id, image] : reconstruction.PointRecon().Images()) {
      if (image.HasPose())
        ba_config.AddImage(image_id);
    }
    if (opts.refine_groups) {
      for (const auto &[gid, _] : reconstruction.StructureRecon().Groups3D()) {
        ba_config.AddVariableGroup(gid);
      }
    }
    auto adjuster = estimators::CreateStructureBundleAdjuster(
        opts, std::move(ba_config), reconstruction);
    auto ba_summary = adjuster->Solve();
    const auto &summary =
        static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary)
            .ceres_summary;
    LOG(INFO) << "Bundle adjustment completed: initial_cost="
              << summary.initial_cost << ", final_cost=" << summary.final_cost;
  }

  if (filter_options) {
    auto filter_stats =
        FilterGroupAssociations(reconstruction, *filter_options);
    size_t num_deleted = DeleteSupportlessGroups(reconstruction);
    LOG(INFO) << "Post-BA group filtering: " << filter_stats.num_groups_passed
              << " passed, " << filter_stats.num_groups_failed << " failed, "
              << filter_stats.num_associations_marked << " marked inactive, "
              << filter_stats.num_associations_purged << " purged, "
              << num_deleted << " groups deleted";
  }
}

} // namespace limap
