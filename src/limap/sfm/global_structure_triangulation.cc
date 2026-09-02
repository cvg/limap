#include "limap/sfm/global_structure_triangulation.h"

#include <colmap/util/logging.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/sfm/group_verification.h"
#include "limap/sfm/structure_observation_manager.h"

namespace limap {

void GlobalTriangulateStructure(
    const StructureDatabaseCache &structure_db_cache,
    const std::shared_ptr<HolisticReconstruction> &reconstruction,
    const GlobalLineTriangulationOptions &line_options,
    const GlobalGroupTriangulationOptions &group_options,
    const estimators::StructureBundleAdjustmentOptions *ba_options,
    const GroupVerificationOptions *filter_options,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
  THROW_CHECK(line_options.Check());
  THROW_CHECK(group_options.Check());

  reconstruction->StructureRecon().Load(structure_db_cache);
  reconstruction->StructureRecon().InitializeAllWireframes(
      line_options.wireframe2d_th);

  auto structure_corr_graph = structure_db_cache.StructureCorrespondenceGraph();

  LOG(INFO) << "Running global line triangulation...";
  GlobalLineTriangulationController line_ctrl(line_options, reconstruction,
                                              structure_corr_graph->LineGraph(),
                                              exhaustive_match_neighbors);
  line_ctrl.Run();

  LOG(INFO) << "Global line triangulation complete: "
            << reconstruction->StructureRecon().NumLines3D() << " 3D lines";

  // Classify lines by observation count + pixel uncertainty.
  // All observations are active by default after fresh triangulation.
  FlatHashSet<line3D_t> reliable_ids, unreliable_ids;
  if (group_options.pixel_uncertainty_threshold > 0 && ba_options) {
    StructureObservationManager obs_mgr(reconstruction->StructureRecon());
    auto line_loss =
        estimators::CreateLossFunction(ba_options->loss_function_type_line,
                                       ba_options->loss_function_scale_line);
    obs_mgr.ClassifyLineTracks(
        /*min_active_observations=*/4,
        group_options.pixel_uncertainty_threshold, line_loss.get(),
        reliable_ids, unreliable_ids);
    LOG(INFO) << "Line classification: " << reliable_ids.size() << " reliable, "
              << unreliable_ids.size() << " unreliable";
  }

  LOG(INFO) << "Running global group triangulation...";
  GlobalGroupTriangulationController group_ctrl(
      group_options, reconstruction, structure_corr_graph->GroupGraph());
  group_ctrl.Run();

  LOG(INFO) << "Global structure triangulation complete: "
            << reconstruction->StructureRecon().NumLines3D() << " lines, "
            << reconstruction->StructureRecon().NumGroups3D() << " groups";

  // Always filter group associations after triangulation so that
  // active/inactive labels are meaningful (they default to all-active).
  // This ensures downstream BA's min_active_group_associations threshold
  // filters by quality, not just total count.
  {
    GroupVerificationOptions verify_opts;
    verify_opts.default_vp_threshold =
        group_options.verification_default_vp_threshold;
    verify_opts.default_reproj_threshold =
        group_options.verification_default_reproj_threshold;
    verify_opts.min_num_inliers = group_options.verification_min_num_inliers;
    verify_opts.min_inlier_ratio = group_options.verification_min_inlier_ratio;
    verify_opts.obs_inlier_ratio = group_options.verification_obs_inlier_ratio;
    verify_opts.filter_outliers = group_options.verification_filter_outliers;

    auto filter_stats = FilterGroupAssociations(*reconstruction, verify_opts);
    size_t num_deleted = DeleteSupportlessGroups(*reconstruction);
    LOG(INFO) << "Post-triangulation group filtering: "
              << filter_stats.num_groups_passed << " groups passed, "
              << filter_stats.num_groups_failed << " failed, "
              << filter_stats.num_associations_marked
              << " associations marked inactive, " << num_deleted
              << " groups deleted ("
              << reconstruction->StructureRecon().NumGroups3D()
              << " remaining)";
  }

  if (ba_options) {
    LOG(INFO) << "Running bundle adjustment...";
    // During triangulation we only refine structure, never cameras/rigs.
    auto opts = *ba_options;
    opts.refine_focal_length = false;
    opts.refine_principal_point = false;
    opts.refine_extra_params = false;
    opts.refine_sensor_from_rig = false;
    opts.refine_rig_from_world = false;

    auto &point_recon = reconstruction->PointRecon();
    auto &srec = reconstruction->StructureRecon();

    if (!unreliable_ids.empty()) {
      // ---- 2-step BA: reliable lines + groups + wireframe, then unreliable
      // ----

      // Step 1: reliable lines + groups + wireframe (full structure BA)
      {
        estimators::StructureBundleAdjustmentConfig ba_config;
        for (const auto &[image_id, image] : point_recon.Images()) {
          if (image.HasPose())
            ba_config.AddImage(image_id);
        }

        // Add only reliable lines
        for (const line3D_t lid : reliable_ids) {
          if (srec.ExistsLine3D(lid)) {
            ba_config.AddVariableLine(lid);
          }
        }

        // Add all groups — BA layer filters by min_active_group_associations
        if (opts.refine_groups) {
          for (const auto &[gid, _] : srec.Groups3D()) {
            ba_config.AddVariableGroup(gid);
          }
        }

        LOG(INFO) << "  2-step BA Step 1: " << reliable_ids.size()
                  << " reliable lines, " << srec.NumGroups3D() << " groups";
        auto adjuster = estimators::CreateStructureBundleAdjuster(
            opts, std::move(ba_config), *reconstruction);
        auto ba_summary1 = adjuster->Solve();
        const auto &summary1 =
            static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary1)
                .ceres_summary;
        LOG(INFO) << "  Step 1 completed: initial_cost="
                  << summary1.initial_cost
                  << ", final_cost=" << summary1.final_cost;
      }

      // Step 2: unreliable lines with structure constraints
      {
        for (const line3D_t lid : unreliable_ids) {
          if (srec.ExistsLine3D(lid)) {
            srec.Line(lid).ClearInactiveLabels();
          }
        }

        estimators::StructureBundleAdjustmentConfig ba_config_step2;

        for (const auto &[image_id, image] : point_recon.Images()) {
          if (image.HasPose())
            ba_config_step2.AddImage(image_id);
        }
        for (const colmap::frame_t frame_id : point_recon.RegFrameIds()) {
          ba_config_step2.SetConstantRigFromWorldPose(frame_id);
        }
        for (const colmap::image_t img_id : point_recon.RegImageIds()) {
          ba_config_step2.SetConstantCamIntrinsics(
              point_recon.Image(img_id).CameraId());
        }

        // Only unreliable lines — everything else automatically skipped
        for (const line3D_t lid : unreliable_ids) {
          if (srec.ExistsLine3D(lid))
            ba_config_step2.AddVariableLine(lid);
        }
        // Only include groups with enough reliable associations in step 2.
        // Groups with poorly-constrained geometry can pull unreliable lines
        // to wrong positions.
        size_t num_step2_groups = 0;
        if (opts.refine_groups) {
          const size_t min_reliable =
              opts.min_reliable_group_associations_step2;
          for (const auto &[gid, group] : srec.Groups3D()) {
            size_t n_reliable = group.CountActivePoints();
            for (const auto &assoc : group.lines) {
              if (reliable_ids.count(assoc.idx))
                n_reliable++;
            }
            if (n_reliable >= min_reliable) {
              ba_config_step2.AddConstantGroup(gid);
              ++num_step2_groups;
            }
          }
        }

        estimators::StructureBundleAdjustmentOptions ba_opts_step2(opts);
        ba_opts_step2.refine_rig_from_world = false;
        ba_opts_step2.refine_sensor_from_rig = false;
        ba_opts_step2.refine_focal_length = false;
        ba_opts_step2.refine_principal_point = false;
        ba_opts_step2.refine_extra_params = false;
        ba_opts_step2.refine_points = false;
        ba_opts_step2.refine_groups = false;
        ba_opts_step2.disable_point_residuals = true;
        ba_opts_step2.force_reconstruct_wireframe3d = false;

        LOG(INFO) << "  2-step BA Step 2: " << unreliable_ids.size()
                  << " unreliable lines, " << num_step2_groups << "/"
                  << srec.NumGroups3D() << " groups (min_reliable="
                  << opts.min_reliable_group_associations_step2 << ")";
        auto adjuster2 = estimators::CreateStructureBundleAdjuster(
            ba_opts_step2, std::move(ba_config_step2), *reconstruction);
        auto ba_summary2 = adjuster2->Solve();
        const auto &summary2 =
            static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary2)
                .ceres_summary;
        LOG(INFO) << "  Step 2 completed: initial_cost="
                  << summary2.initial_cost
                  << ", final_cost=" << summary2.final_cost;
      }
    } else {
      // Single-step BA (no classification or all lines reliable)
      estimators::StructureBundleAdjustmentConfig ba_config;
      for (const auto &[image_id, image] : point_recon.Images()) {
        if (image.HasPose())
          ba_config.AddImage(image_id);
      }
      if (opts.refine_groups) {
        for (const auto &[gid, _] : srec.Groups3D()) {
          ba_config.AddVariableGroup(gid);
        }
      }
      auto adjuster = estimators::CreateStructureBundleAdjuster(
          opts, std::move(ba_config), *reconstruction);
      auto ba_summary = adjuster->Solve();
      const auto &summary =
          static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary)
              .ceres_summary;
      LOG(INFO) << "Bundle adjustment completed: initial_cost="
                << summary.initial_cost
                << ", final_cost=" << summary.final_cost;
    }
  }

  if (filter_options) {
    auto filter_stats =
        FilterGroupAssociations(*reconstruction, *filter_options);
    size_t num_deleted = DeleteSupportlessGroups(*reconstruction);
    LOG(INFO) << "Post-BA group filtering: " << filter_stats.num_groups_passed
              << " passed, " << filter_stats.num_groups_failed << " failed, "
              << filter_stats.num_associations_marked << " marked inactive, "
              << filter_stats.num_associations_purged << " purged, "
              << num_deleted << " groups deleted";
  }
}

} // namespace limap
