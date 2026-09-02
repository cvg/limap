#include "limap/sfm/incremental_line_triangulator.h"

#include "limap/estimators/triangulation/functions.h"

#include <algorithm>
#include <future>

#include <colmap/scene/reconstruction.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/geometry/inf_line3d.h"
#include "limap/scene/structure2d.h"
#include "limap/scene/structure_reconstruction.h"

namespace limap {

bool IncrementalLineTriangulator::Options::Check() const {
  CHECK_OPTION_GE(max_transitivity, 0);
  CHECK_OPTION_GE(min_length_2d, 0.0);
  CHECK_OPTION_GE(IoU_threshold, 0.0);
  CHECK_OPTION_GE(sensitivity_threshold, 0.0);
  CHECK_OPTION_GT(var2d, 0.0);
  CHECK_OPTION_GE(fullscore_th, 0.0);
  CHECK_OPTION_GT(create_max_angle_error, 0.0);
  CHECK_OPTION_GT(continue_max_angle_error, 0.0);
  CHECK_OPTION_GT(merge_max_reproj_error, 0.0);
  CHECK_OPTION_GT(merge_max_angle_error_2d, 0.0);
  CHECK_OPTION_GT(merge_max_angle_error_3d, 0.0);
  CHECK_OPTION_GT(complete_max_reproj_error, 0.0);
  CHECK_OPTION_GE(complete_max_transitivity, 0);
  CHECK_OPTION_GT(re_max_angle_error, 0.0);
  CHECK_OPTION_GE(re_min_ratio, 0.0);
  CHECK_OPTION_LE(re_min_ratio, 1.0);
  CHECK_OPTION_GE(re_max_trials, 0);
  CHECK_OPTION_GT(min_angle, 0.0);
  return true;
}

IncrementalLineTriangulator::IncrementalLineTriangulator(
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
IncrementalLineTriangulator::TriangulateImage(const Options &options,
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
  const size_t num_lines = S.NumLines();

  // Collect eligible line indices
  std::vector<line2D_t> eligible_lines;
  eligible_lines.reserve(num_lines);
  for (line2D_t idx = 0; idx < static_cast<line2D_t>(num_lines); ++idx) {
    if (S.Line(idx).Length() > options.min_length_2d) {
      eligible_lines.push_back(idx);
    }
  }

  if (eligible_lines.empty()) {
    return num_tris;
  }

  // Phase 1: parallel evaluation (read-only)
  const int num_threads = colmap::GetEffectiveNumThreads(options.num_threads);
  std::vector<std::shared_future<TriLineResult>> futures;
  futures.reserve(eligible_lines.size());

  {
    colmap::ThreadPool pool(num_threads);
    for (const line2D_t idx : eligible_lines) {
      futures.push_back(pool.AddTask(&IncrementalLineTriangulator::EvaluateLine,
                                     this, std::cref(options), image_id, idx));
    }
    pool.Wait();
  }

  // Phase 2: sequential application of results
  for (auto &future : futures) {
    TriLineResult result = future.get();
    const line2D_t line2D_idx = result.line2D_idx;

    // Re-check: ref might have gotten a 3D line from an earlier iteration
    const Line2d &ref_line2D = srec.Structure2d(image_id).Line(line2D_idx);
    if (ref_line2D.HasLine3D()) {
      continue;
    }

    // Try Continue first
    if (result.best_continue_id != kInvalidLine3dId &&
        result.best_continue_score > 0.0) {
      // Re-check: target 3D line must still exist
      if (srec.ExistsLine3D(result.best_continue_id)) {
        const colmap::TrackElement track_el(
            image_id, static_cast<colmap::point2D_t>(line2D_idx));
        obs_manager_->AddLineObservation(result.best_continue_id, track_el);
        AddModifiedLine3D(result.best_continue_id);
        num_tris += 1;
        continue;
      }
    }

    // Try Create
    if (result.best_proposal.has_value() && result.new_corrs.size() >= 2) {
      // Re-filter new_corrs: some may have acquired 3D lines from earlier
      // creates in this Phase 2 loop
      std::vector<LineCorrData> valid_corrs;
      valid_corrs.reserve(result.new_corrs.size());
      for (const auto &corr : result.new_corrs) {
        const Structure2d &corr_S = srec.Structure2d(corr.image_id);
        const Line2d &corr_line2D = corr_S.Line(corr.line2D_idx);
        if (!corr_line2D.HasLine3D()) {
          valid_corrs.push_back(corr);
        }
      }

      if (valid_corrs.size() >= 2) {
        Line3d line3d = result.best_proposal.value();

        // Build track from valid correspondences
        for (const auto &corr : valid_corrs) {
          line3d.track.AddElement(colmap::TrackElement(
              corr.image_id, static_cast<colmap::point2D_t>(corr.line2D_idx)));
        }

        const line3D_t new_line3D_id = obs_manager_->AddLine3D(line3d);
        AddModifiedLine3D(new_line3D_id);
        num_tris += valid_corrs.size();
      }
    }
  }

  return num_tris;
}

size_t IncrementalLineTriangulator::CompleteImage(const Options &options,
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

  // Build linker once for all Complete evaluations (thread-safe: const).
  LineLinker linker(options.linker2d_options, options.linker3d_options);

  Structure2d &S = srec.Structure2d(image_id);
  const size_t num_lines = S.NumLines();

  // Classify lines into Complete (has 3D) and Create (no 3D) paths.
  FlatHashSet<line3D_t> seen_line3D_ids;
  std::vector<line3D_t> unique_complete_ids;
  std::vector<line2D_t> create_indices;

  for (line2D_t idx = 0; idx < static_cast<line2D_t>(num_lines); ++idx) {
    const Line2d &line2D = S.Line(idx);
    if (line2D.HasLine3D()) {
      if (seen_line3D_ids.insert(line2D.line3D_id).second) {
        unique_complete_ids.push_back(line2D.line3D_id);
      }
    } else {
      create_indices.push_back(idx);
    }
  }

  // Result type for parallel Complete evaluation.
  struct CompleteCand {
    colmap::image_t img_id;
    line2D_t line2D_idx;
  };
  struct CompleteEvalResult {
    line3D_t line3D_id = kInvalidLine3dId;
    std::vector<CompleteCand> candidates;
  };

  // Phase 1: parallel read-only evaluation.
  const int num_threads = colmap::GetEffectiveNumThreads(options.num_threads);
  std::vector<std::shared_future<CompleteEvalResult>> complete_futures;
  std::vector<std::shared_future<TriLineResult>> create_futures;

  {
    colmap::ThreadPool pool(num_threads);

    // Evaluate Complete for each unique line3D.
    for (const line3D_t lid : unique_complete_ids) {
      complete_futures.push_back(pool.AddTask([&srec, &point_recon, &linker,
                                               &options, this,
                                               lid]() -> CompleteEvalResult {
        CompleteEvalResult result;
        result.line3D_id = lid;

        if (!srec.ExistsLine3D(lid)) {
          return result;
        }
        const Line3d &line3d = srec.Line(lid);
        // Collect existing observations.
        Node2dSet existing_obs;
        for (const auto &elem : line3d.track.Elements()) {
          existing_obs.insert(Node2d(
              elem.image_id, static_cast<feature2D_t>(elem.point2D_idx)));
        }

        // Collect candidate observations via correspondences.
        Node2dSet candidate_obs;
        for (const auto &elem : line3d.track.Elements()) {
          std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
          GetCorrespondences(options, elem.image_id,
                             static_cast<line2D_t>(elem.point2D_idx), &corrs);
          for (const auto &corr : corrs) {
            Node2d node(corr.image_id,
                        static_cast<feature2D_t>(corr.point2D_idx));
            if (existing_obs.count(node) == 0) {
              candidate_obs.insert(node);
            }
          }
        }

        // Evaluate each candidate (read-only).
        for (const Node2d &node : candidate_obs) {
          const colmap::image_t img_id = node.first;
          const line2D_t l2d_idx = static_cast<line2D_t>(node.second);

          if (!point_recon.ExistsImage(img_id))
            continue;
          const colmap::Image &img = point_recon.Image(img_id);
          if (!img.HasPose())
            continue;
          if (!srec.ExistsStructure2D(img_id))
            continue;
          const Structure2d &cS = srec.Structure2d(img_id);
          if (l2d_idx >= static_cast<line2D_t>(cS.NumLines()))
            continue;
          const Line2d &l2d = cS.Line(l2d_idx);
          if (l2d.HasLine3D())
            continue;

          auto proj_res = line3d.Projection(img);
          if (!proj_res.has_value())
            continue;

          double score = linker.ComputeScore2d(proj_res.value(), l2d);
          if (score <= 0.0)
            continue;

          result.candidates.push_back({img_id, l2d_idx});
        }

        return result;
      }));
    }

    // Evaluate Create for each unassigned line (reuse EvaluateLine).
    for (const line2D_t idx : create_indices) {
      create_futures.push_back(
          pool.AddTask(&IncrementalLineTriangulator::EvaluateLine, this,
                       std::cref(options), image_id, idx));
    }

    pool.Wait();
  }

  // Phase 2a: sequential application of Complete results.
  for (auto &future : complete_futures) {
    CompleteEvalResult result = future.get();
    if (result.candidates.empty()) {
      continue;
    }
    if (!srec.ExistsLine3D(result.line3D_id)) {
      continue;
    }

    bool modified = false;
    for (const auto &cand : result.candidates) {
      // Re-check: 2D line may have been assigned by an earlier iteration.
      if (!srec.ExistsStructure2D(cand.img_id))
        continue;
      Structure2d &cS = srec.Structure2d(cand.img_id);
      if (cand.line2D_idx >= static_cast<line2D_t>(cS.NumLines()))
        continue;
      Line2d &l2d = cS.Line(cand.line2D_idx);
      if (l2d.HasLine3D())
        continue;

      const colmap::TrackElement track_el(
          cand.img_id, static_cast<colmap::point2D_t>(cand.line2D_idx));
      obs_manager_->AddLineObservation(result.line3D_id, track_el);
      num_tris++;
      modified = true;
    }
    if (modified) {
      AddModifiedLine3D(result.line3D_id);
    }
  }

  // Phase 2b: sequential application of Create results.
  for (auto &future : create_futures) {
    TriLineResult result = future.get();
    const line2D_t line2D_idx = result.line2D_idx;

    // Re-check: ref might have gotten a 3D line from an earlier iteration.
    const Line2d &ref_line2D = srec.Structure2d(image_id).Line(line2D_idx);
    if (ref_line2D.HasLine3D()) {
      continue;
    }

    if (result.best_proposal.has_value() && result.new_corrs.size() >= 2) {
      // Re-filter: some corrs may have acquired 3D lines from earlier creates.
      std::vector<LineCorrData> valid_corrs;
      valid_corrs.reserve(result.new_corrs.size());
      for (const auto &corr : result.new_corrs) {
        const Structure2d &corr_S = srec.Structure2d(corr.image_id);
        const Line2d &corr_line2D = corr_S.Line(corr.line2D_idx);
        if (!corr_line2D.HasLine3D()) {
          valid_corrs.push_back(corr);
        }
      }

      if (valid_corrs.size() >= 2) {
        Line3d line3d = result.best_proposal.value();
        for (const auto &corr : valid_corrs) {
          line3d.track.AddElement(colmap::TrackElement(
              corr.image_id, static_cast<colmap::point2D_t>(corr.line2D_idx)));
        }
        const line3D_t new_line3D_id = obs_manager_->AddLine3D(line3d);
        AddModifiedLine3D(new_line3D_id);
        num_tris += valid_corrs.size();
      }
    }
  }

  return num_tris;
}

size_t IncrementalLineTriangulator::CompleteTracks(
    const Options &options, const FlatHashSet<line3D_t> &line3D_ids) {
  THROW_CHECK(options.Check());

  // Build linker once for all Complete calls (avoids per-call std::function
  // allocation in the Complete loop).
  LineLinker linker(options.linker2d_options, options.linker3d_options);

  size_t num_completed = 0;
  for (const line3D_t line3D_id : line3D_ids) {
    num_completed += Complete(options, line3D_id, linker);
  }
  return num_completed;
}

size_t IncrementalLineTriangulator::CompleteAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const auto &lines3d = srec.Lines3D();

  FlatHashSet<line3D_t> line3D_ids;
  line3D_ids.reserve(lines3d.size());
  for (const auto &kv : lines3d) {
    line3D_ids.insert(kv.first);
  }

  return CompleteTracks(options, line3D_ids);
}

size_t IncrementalLineTriangulator::MergeTracks(
    const Options &options, const FlatHashSet<line3D_t> &line3D_ids) {
  THROW_CHECK(options.Check());

  // Build linkers once for all merge calls (avoids per-call std::function
  // allocation in the Merge loop).
  LineLinker2dOptions linker2d_merge_options;
  linker2d_merge_options.use_angle = true;
  linker2d_merge_options.th_angle = options.merge_max_angle_error_2d;
  linker2d_merge_options.use_overlap = false;
  linker2d_merge_options.use_smartangle = false;
  linker2d_merge_options.use_perp = true;
  linker2d_merge_options.th_perp = options.merge_max_reproj_error;
  linker2d_merge_options.use_innerseg = false;
  LineLinker2d merge_linker2d(linker2d_merge_options);

  LineLinker3dOptions linker3d_opts;
  linker3d_opts.use_angle = true;
  linker3d_opts.th_angle = options.merge_max_angle_error_3d;
  linker3d_opts.use_overlap = false;
  linker3d_opts.use_smartangle = false;
  linker3d_opts.use_perp = false;
  linker3d_opts.use_innerseg = false;
  LineLinker3d merge_linker3d(linker3d_opts);

  size_t num_merged = 0;
  for (const line3D_t line3D_id : line3D_ids) {
    num_merged += Merge(options, line3D_id, merge_linker2d, merge_linker3d);
  }
  return num_merged;
}

size_t IncrementalLineTriangulator::MergeAllTracks(const Options &options) {
  THROW_CHECK(options.Check());

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const auto &lines3d = srec.Lines3D();

  FlatHashSet<line3D_t> line3D_ids;
  line3D_ids.reserve(lines3d.size());
  for (const auto &kv : lines3d) {
    line3D_ids.insert(kv.first);
  }

  return MergeTracks(options, line3D_ids);
}

size_t IncrementalLineTriangulator::Retriangulate(const Options &options) {
  THROW_CHECK(options.Check());
  // TODO: Implement retriangulation for under-reconstructed image pairs
  // This follows COLMAP's Retriangulate pattern
  return 0;
}

void IncrementalLineTriangulator::AddModifiedLine3D(line3D_t line3D_id) {
  modified_line3D_ids_.insert(line3D_id);
}

const FlatHashSet<line3D_t> &IncrementalLineTriangulator::GetModifiedLines3D() {
  return modified_line3D_ids_;
}

void IncrementalLineTriangulator::ClearModifiedLines3D() {
  modified_line3D_ids_.clear();
}

void IncrementalLineTriangulator::ClearCaches() { merge_trials_.clear(); }

IncrementalLineTriangulator::TriLineResult
IncrementalLineTriangulator::EvaluateLine(const Options &options,
                                          colmap::image_t image_id,
                                          line2D_t line2D_idx) const {
  TriLineResult result;
  result.line2D_idx = line2D_idx;

  const StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  const colmap::Image &image = point_recon.Image(image_id);
  const Structure2d &S = srec.Structure2d(image_id);
  const Line2d &ref_line2D = S.Line(line2D_idx);

  // --- Find correspondences (same as Find(), but uses local vector) ---
  std::vector<colmap::CorrespondenceGraph::Correspondence> local_found_corrs;
  GetCorrespondences(options, image_id, line2D_idx, &local_found_corrs);

  std::vector<LineCorrData> corrs_data;
  result.num_triangulated = 0;

  for (const auto &corr : local_found_corrs) {
    const colmap::image_t corr_image_id = corr.image_id;
    const line2D_t corr_line2D_idx = static_cast<line2D_t>(corr.point2D_idx);

    if (!point_recon.ExistsImage(corr_image_id)) {
      continue;
    }
    const colmap::Image &corr_image = point_recon.Image(corr_image_id);
    if (!corr_image.HasPose()) {
      continue;
    }
    if (!srec.ExistsStructure2D(corr_image_id)) {
      continue;
    }
    const Structure2d &corr_S = srec.Structure2d(corr_image_id);
    if (corr_line2D_idx >= static_cast<line2D_t>(corr_S.NumLines())) {
      continue;
    }
    const Line2d &corr_line2D = corr_S.Line(corr_line2D_idx);
    if (corr_line2D.Length() <= options.min_length_2d) {
      continue;
    }

    LineCorrData corr_data;
    corr_data.image_id = corr_image_id;
    corr_data.line2D_idx = corr_line2D_idx;
    corr_data.image = &corr_image;
    corr_data.line2D = &corr_line2D;
    corrs_data.push_back(corr_data);

    if (corr_line2D.HasLine3D()) {
      result.num_triangulated++;
    }
  }

  if (corrs_data.empty()) {
    return result;
  }

  // --- Continue scoring (read-only) ---
  if (result.num_triangulated > 0) {
    LineLinker linker(options.linker2d_options, options.linker3d_options);

    for (const auto &corr : corrs_data) {
      const Structure2d &corr_S = srec.Structure2d(corr.image_id);
      const Line2d &corr_line2D = corr_S.Line(corr.line2D_idx);

      if (!corr_line2D.HasLine3D()) {
        continue;
      }

      const line3D_t line3D_id = corr_line2D.line3D_id;
      if (!srec.ExistsLine3D(line3D_id)) {
        continue;
      }
      const Line3d &line3d = srec.Line(line3D_id);

      if (!srec.Line(line3D_id).IsObservationActive(corr.image_id)) {
        continue;
      }

      auto proj_res = line3d.Projection(image);
      if (!proj_res.has_value()) {
        continue;
      }
      const Line2d proj_line2d = proj_res.value();

      double score = linker.ComputeScore2d(proj_line2d, ref_line2D);
      if (score > result.best_continue_score) {
        result.best_continue_score = score;
        result.best_continue_id = line3D_id;
      }
    }
  }

  // --- Create proposal generation + scoring (read-only) ---
  // Build new_corrs: correspondences without existing 3D lines + ref
  LineCorrData ref_corr_data;
  ref_corr_data.image_id = image_id;
  ref_corr_data.line2D_idx = line2D_idx;
  ref_corr_data.image = &image;
  ref_corr_data.line2D = &ref_line2D;

  result.new_corrs.reserve(corrs_data.size() + 1);
  for (const auto &corr : corrs_data) {
    const Structure2d &corr_S = srec.Structure2d(corr.image_id);
    const Line2d &corr_line2D = corr_S.Line(corr.line2D_idx);
    if (!corr_line2D.HasLine3D()) {
      result.new_corrs.push_back(corr);
    }
  }
  result.new_corrs.push_back(ref_corr_data);

  if (result.new_corrs.size() < 2) {
    result.new_corrs.clear();
    return result;
  }

  // Reference for proposal generation. Matched candidates are all views of
  // the same 3D line, so any of them serves. Exhaustive candidates are not,
  // so the query line (appended last) must be the reference.
  const size_t prop_ref_idx = options.exhaustive_match_neighbors.empty()
                                  ? 0
                                  : result.new_corrs.size() - 1;
  const LineCorrData &prop_ref_corr = result.new_corrs[prop_ref_idx];
  const Node2d ref_node(prop_ref_corr.image_id,
                        static_cast<feature2D_t>(prop_ref_corr.line2D_idx));
  const Line2d &prop_ref_line2D = *prop_ref_corr.line2D;
  const colmap::Image &prop_ref_image = *prop_ref_corr.image;
  const Structure2d &prop_ref_S = srec.Structure2d(prop_ref_corr.image_id);

  internal::line_triangulation::ProposalList proposals;
  const LineTriangulationParams tri_params = options.GetTriangulationParams();

  for (size_t i = 0; i < result.new_corrs.size(); ++i) {
    if (i == prop_ref_idx) {
      continue;
    }
    const LineCorrData &ng_corr = result.new_corrs[i];
    const Node2d ng_node(ng_corr.image_id,
                         static_cast<feature2D_t>(ng_corr.line2D_idx));
    const Line2d &ng_line2D = *ng_corr.line2D;
    const colmap::Image &ng_image = *ng_corr.image;
    const Structure2d &ng_S = srec.Structure2d(ng_corr.image_id);

    if (options.use_point_assisted_triangulation) {
      internal::line_triangulation::AddPointBasedProposalsForMatch(
          ref_node, prop_ref_line2D, prop_ref_image, ng_node, ng_line2D,
          ng_image, srec, tri_params, proposals);
    }

    if (options.use_vp_assisted_triangulation) {
      internal::line_triangulation::AddVPBasedProposalsForMatch(
          ref_node, prop_ref_line2D, prop_ref_image, ng_node, ng_line2D,
          ng_image, prop_ref_S, ng_S, srec, tri_params, proposals);
    }

    internal::line_triangulation::AddEpipolarProposalsForMatch(
        prop_ref_line2D, prop_ref_image, ng_line2D, ng_image, ng_node,
        tri_params, proposals);
  }

  if (!proposals.empty()) {
    LineLinker3dOptions linker3d_scoring_options = options.linker3d_options;
    linker3d_scoring_options.SetToSharedParentScoring();
    LineLinker linker_scoring(options.linker2d_options,
                              linker3d_scoring_options);

    const auto best = internal::line_triangulation::SelectBestSupportedProposal(
        proposals, srec, linker_scoring, options.scale_inv_th,
        options.fullscore_th);

    const int best_idx = best.first;
    if (best_idx >= 0) {
      result.best_proposal = proposals[static_cast<size_t>(best_idx)].line;
    }
  }

  return result;
}

void IncrementalLineTriangulator::GetCorrespondences(
    const Options &options, colmap::image_t image_id, line2D_t line2D_idx,
    std::vector<colmap::CorrespondenceGraph::Correspondence> *corrs) const {
  corrs->clear();
  const StructureReconstruction &srec = reconstruction_.StructureRecon();

  if (!options.exhaustive_match_neighbors.empty()) {
    // No matches to follow: keep every line in a neighbor whose epipolar band
    // overlaps the query. Without this the unfiltered candidates swamp the
    // proposal pool and nothing wins the support vote.
    const auto it_ng = options.exhaustive_match_neighbors.find(image_id);
    if (it_ng == options.exhaustive_match_neighbors.end()) {
      return;
    }
    const colmap::Reconstruction &point_recon = srec.PointRecon();
    if (!srec.ExistsStructure2D(image_id) ||
        !point_recon.ExistsImage(image_id)) {
      return;
    }
    const colmap::Image &image1 = point_recon.Image(image_id);
    if (!image1.HasPose()) {
      return;
    }
    const Structure2d &S1 = srec.Structure2d(image_id);
    if (line2D_idx >= static_cast<line2D_t>(S1.NumLines())) {
      return;
    }
    const Line2d &l1 = S1.Line(line2D_idx);

    const auto &structures2d = srec.Structures2d();
    for (const colmap::image_t ng_img_id : it_ng->second) {
      const auto it_s2 = structures2d.find(ng_img_id);
      if (it_s2 == structures2d.end() || !point_recon.ExistsImage(ng_img_id)) {
        continue;
      }
      const colmap::Image &image2 = point_recon.Image(ng_img_id);
      if (!image2.HasPose()) {
        continue;
      }
      // The fundamental matrix and the query's epipolar lines depend only on
      // the image pair and l1, so hoist them out of the per-candidate loop.
      const M3D F =
          estimators::triangulation::ComputeFundamentalMatrix(image1, image2);
      const V3D ep_start =
          (F * V3D(l1.start[0], l1.start[1], 1.0)).normalized();
      const V3D ep_end = (F * V3D(l1.end[0], l1.end[1], 1.0)).normalized();

      const Structure2d &S2 = it_s2->second;
      const size_t num_ng_lines = S2.NumLines();
      for (size_t ng_line_id = 0; ng_line_id < num_ng_lines; ++ng_line_id) {
        const Line2d &l2 = S2.Line(static_cast<line2D_t>(ng_line_id));
        const double len2 = l2.Length();
        if (len2 <= 0.0) {
          continue;
        }
        const V3D coor_l2 = l2.Coords();
        const V2D c_start = coor_l2.cross(ep_start).hnormalized();
        const V2D c_end = coor_l2.cross(ep_end).hnormalized();
        const V2D dir2 = l2.Direction();
        double c1 = (c_start - l2.start).dot(dir2) / len2;
        double c2 = (c_end - l2.start).dot(dir2) / len2;
        if (c1 > c2) {
          std::swap(c1, c2);
        }
        const double iou = (std::min(c2, 1.0) - std::max(c1, 0.0)) /
                           (std::max(c2, 1.0) - std::min(c1, 0.0));
        if (iou < options.IoU_threshold) {
          continue;
        }
        corrs->emplace_back(ng_img_id,
                            static_cast<colmap::point2D_t>(ng_line_id));
      }
    }
    return;
  }

  structure_corr_graph_->LineGraph().ExtractCorrespondences(
      image_id, static_cast<colmap::point2D_t>(line2D_idx), corrs);
}

size_t IncrementalLineTriangulator::Find(
    const Options &options, colmap::image_t image_id, line2D_t line2D_idx,
    size_t transitivity, std::vector<LineCorrData> *corrs_data) {
  corrs_data->clear();

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  // Extract correspondences for this line
  GetCorrespondences(options, image_id, line2D_idx, &found_corrs_);

  size_t num_triangulated = 0;

  for (const auto &corr : found_corrs_) {
    const colmap::image_t corr_image_id = corr.image_id;
    const line2D_t corr_line2D_idx = static_cast<line2D_t>(corr.point2D_idx);

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
    if (corr_line2D_idx >= static_cast<line2D_t>(corr_S.NumLines())) {
      continue;
    }

    const Line2d &corr_line2D = corr_S.Line(corr_line2D_idx);

    // Skip short lines
    if (corr_line2D.Length() <= options.min_length_2d) {
      continue;
    }

    LineCorrData corr_data;
    corr_data.image_id = corr_image_id;
    corr_data.line2D_idx = corr_line2D_idx;
    corr_data.image = &corr_image;
    corr_data.line2D = &corr_line2D;

    corrs_data->push_back(corr_data);

    // Count how many correspondences already have a 3D line
    if (corr_line2D.HasLine3D()) {
      num_triangulated++;
    }
  }

  return num_triangulated;
}

size_t IncrementalLineTriangulator::Create(
    const Options &options, const std::vector<LineCorrData> &corrs_data) {
  if (corrs_data.size() < 2) {
    return 0;
  }

  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  // Filter out correspondences that already have a 3D line
  std::vector<LineCorrData> new_corrs;
  new_corrs.reserve(corrs_data.size());
  for (const auto &corr : corrs_data) {
    const Structure2d &S = srec.Structure2d(corr.image_id);
    const Line2d &line2D = S.Line(corr.line2D_idx);
    if (!line2D.HasLine3D()) {
      new_corrs.push_back(corr);
    }
  }

  if (new_corrs.size() < 2) {
    return 0;
  }

  // Use the first correspondence as reference
  const LineCorrData &ref_corr = new_corrs[0];
  const Node2d ref_node(ref_corr.image_id,
                        static_cast<feature2D_t>(ref_corr.line2D_idx));
  const Line2d &ref_line2D = *ref_corr.line2D;
  const colmap::Image &ref_image = *ref_corr.image;
  const Structure2d &ref_S = srec.Structure2d(ref_corr.image_id);

  // Generate triangulation proposals
  internal::line_triangulation::ProposalList proposals;
  const LineTriangulationParams tri_params = options.GetTriangulationParams();

  for (size_t i = 1; i < new_corrs.size(); ++i) {
    const LineCorrData &ng_corr = new_corrs[i];
    const Node2d ng_node(ng_corr.image_id,
                         static_cast<feature2D_t>(ng_corr.line2D_idx));
    const Line2d &ng_line2D = *ng_corr.line2D;
    const colmap::Image &ng_image = *ng_corr.image;
    const Structure2d &ng_S = srec.Structure2d(ng_corr.image_id);

    // Point-based triangulation
    if (options.use_point_assisted_triangulation) {
      internal::line_triangulation::AddPointBasedProposalsForMatch(
          ref_node, ref_line2D, ref_image, ng_node, ng_line2D, ng_image, srec,
          tri_params, proposals);
    }

    // VP-based triangulation
    if (options.use_vp_assisted_triangulation) {
      internal::line_triangulation::AddVPBasedProposalsForMatch(
          ref_node, ref_line2D, ref_image, ng_node, ng_line2D, ng_image, ref_S,
          ng_S, srec, tri_params, proposals);
    }

    // Epipolar triangulation
    internal::line_triangulation::AddEpipolarProposalsForMatch(
        ref_line2D, ref_image, ng_line2D, ng_image, ng_node, tri_params,
        proposals);
  }

  if (proposals.empty()) {
    return 0;
  }

  // Select best proposal
  LineLinker3dOptions linker3d_scoring_options = options.linker3d_options;
  linker3d_scoring_options.SetToSharedParentScoring();
  LineLinker linker_scoring(options.linker2d_options, linker3d_scoring_options);

  const auto best = internal::line_triangulation::SelectBestSupportedProposal(
      proposals, srec, linker_scoring, options.scale_inv_th,
      options.fullscore_th);

  const int best_idx = best.first;
  if (best_idx < 0) {
    return 0;
  }

  // Create new 3D line
  Line3d line3d = proposals[static_cast<size_t>(best_idx)].line;

  // Build track from all new correspondences
  for (const auto &corr : new_corrs) {
    line3d.track.AddElement(colmap::TrackElement(
        corr.image_id, static_cast<colmap::point2D_t>(corr.line2D_idx)));
  }

  // Add to reconstruction using observation manager
  const line3D_t new_line3D_id = obs_manager_->AddLine3D(line3d);
  AddModifiedLine3D(new_line3D_id);

  return new_corrs.size();
}

size_t IncrementalLineTriangulator::Continue(
    const Options &options, const LineCorrData &ref_corr_data,
    const std::vector<LineCorrData> &corrs_data) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();

  // Check if reference already has a 3D line
  const Structure2d &ref_S = srec.Structure2d(ref_corr_data.image_id);
  const Line2d &ref_line2D = ref_S.Line(ref_corr_data.line2D_idx);
  if (ref_line2D.HasLine3D()) {
    return 0;
  }

  // Find the best 3D line to continue
  line3D_t best_line3D_id = kInvalidLine3dId;
  double best_score = -std::numeric_limits<double>::infinity();

  LineLinker linker(options.linker2d_options, options.linker3d_options);

  for (const auto &corr : corrs_data) {
    const Structure2d &corr_S = srec.Structure2d(corr.image_id);
    const Line2d &corr_line2D = corr_S.Line(corr.line2D_idx);

    if (!corr_line2D.HasLine3D()) {
      continue;
    }

    const line3D_t line3D_id = corr_line2D.line3D_id;
    if (!srec.ExistsLine3D(line3D_id)) {
      continue;
    }
    const Line3d &line3d = srec.Line(line3D_id);

    // Skip if this correspondence is inactive in its track (matching
    // hybridsfm). An inactive observation means the 3D line doesn't fit
    // well from this view, so don't use it as a basis for continuation.
    if (!srec.Line(line3D_id).IsObservationActive(corr.image_id)) {
      continue;
    }

    // Check reprojection error
    const colmap::Image &ref_image = *ref_corr_data.image;
    auto proj_res = line3d.Projection(ref_image);
    if (!proj_res.has_value()) {
      continue;
    }
    const Line2d proj_line2d = proj_res.value();

    // Score the match
    double score = linker.ComputeScore2d(proj_line2d, ref_line2D);
    if (score > best_score) {
      best_score = score;
      best_line3D_id = line3D_id;
    }
  }

  if (best_line3D_id == kInvalidLine3dId || best_score <= 0.0) {
    return 0;
  }

  // Add observation using observation manager
  const colmap::TrackElement track_el(
      ref_corr_data.image_id,
      static_cast<colmap::point2D_t>(ref_corr_data.line2D_idx));
  obs_manager_->AddLineObservation(best_line3D_id, track_el);
  AddModifiedLine3D(best_line3D_id);

  return 1;
}

size_t IncrementalLineTriangulator::Merge(const Options &options,
                                          line3D_t line3D_id,
                                          const LineLinker2d &linker2d,
                                          const LineLinker3d &linker3d) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  const colmap::Reconstruction &point_recon = srec.PointRecon();
  if (!srec.ExistsLine3D(line3D_id)) {
    return 0;
  }

  const Line3d &line3d = srec.Line(line3D_id);

  // Candidate selection via correspondence graph (like COLMAP)
  FlatHashSet<line3D_t> candidate_ids;
  for (const auto &elem : line3d.track.Elements()) {
    std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
    GetCorrespondences(options, elem.image_id,
                       static_cast<line2D_t>(elem.point2D_idx), &corrs);

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
      const line2D_t corr_line2D_idx = static_cast<line2D_t>(corr.point2D_idx);
      if (corr_line2D_idx >= static_cast<line2D_t>(corr_S.NumLines())) {
        continue;
      }

      const Line2d &corr_line2D = corr_S.Line(corr_line2D_idx);

      // Check if this correspondence has a different 3D line
      if (!corr_line2D.HasLine3D()) {
        continue;
      }
      const line3D_t other_id = corr_line2D.line3D_id;
      if (other_id == line3D_id) {
        continue;
      }

      // Check if already tried
      auto &tried = merge_trials_[line3D_id];
      if (tried.count(other_id) > 0) {
        continue;
      }

      candidate_ids.insert(other_id);
    }
  }

  // Try to merge each candidate
  for (const line3D_t other_id : candidate_ids) {
    if (!srec.ExistsLine3D(other_id)) {
      continue;
    }

    // Mark as tried
    merge_trials_[line3D_id].insert(other_id);
    merge_trials_[other_id].insert(line3D_id);

    const Line3d &other_line3d = srec.Line(other_id);

    // Early 3D rejection via LineLinker3d
    if (!linker3d.CheckConnection(line3d, other_line3d)) {
      continue;
    }

    // Compute infinite line via SVD on the 4 endpoints (fixed-size matrix)
    V3D center =
        (line3d.start + line3d.end + other_line3d.start + other_line3d.end) /
        4.0;

    Eigen::Matrix<double, 4, 3> endpoints;
    endpoints.row(0) = (line3d.start - center).transpose();
    endpoints.row(1) = (line3d.end - center).transpose();
    endpoints.row(2) = (other_line3d.start - center).transpose();
    endpoints.row(3) = (other_line3d.end - center).transpose();

    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 3>> svd(endpoints,
                                                      Eigen::ComputeThinV);
    V3D direc = svd.matrixV().col(0).normalized();
    InfiniteLine3d inf_line(center, direc);

    // Get merged segment using 3D endpoints: O(4 dot products) instead of
    // O(N projections) from the 2D version.
    const std::vector<Line3d> lines_to_merge = {line3d, other_line3d};
    Line3d merged_line =
        GetLineSegmentFromInfiniteLine3d(inf_line, lines_to_merge, 0);

    // Validate observations against merged geometry. Iterates track elements
    // directly using const references (no colmap::Image copies).
    std::vector<colmap::image_t> outlier_image_ids;
    size_t num_inliers = 0;

    auto validate_track = [&](const colmap::Track &track) {
      for (const auto &elem : track.Elements()) {
        const colmap::Image &image = point_recon.Image(elem.image_id);
        const Line2d &observed_line2d =
            srec.Structure2d(elem.image_id).Line(elem.point2D_idx);

        auto proj_res = merged_line.Projection(image);
        if (!proj_res.has_value()) {
          outlier_image_ids.push_back(elem.image_id);
          continue;
        }
        const Line2d proj_line2d = proj_res.value();

        if (linker2d.CheckConnection(proj_line2d, observed_line2d)) {
          ++num_inliers;
        } else {
          outlier_image_ids.push_back(elem.image_id);
        }
      }
    };

    validate_track(line3d.track);
    validate_track(other_line3d.track);

    // Require at least 2 inlier observations to accept merge
    if (num_inliers < 2) {
      continue;
    }

    // Merge the lines
    const size_t num_merged =
        line3d.track.Length() + other_line3d.track.Length();
    const line3D_t merged_id = obs_manager_->MergeLines3D(line3D_id, other_id);

    // Mark outlier observations as inactive on the merged line (soft filter)
    for (const colmap::image_t outlier_img_id : outlier_image_ids) {
      srec.Line(merged_id).SetObservationInactive(outlier_img_id);
    }

    // Update modified set (like COLMAP)
    modified_line3D_ids_.erase(line3D_id);
    modified_line3D_ids_.erase(other_id);
    modified_line3D_ids_.insert(merged_id);

    // Recursively try to merge the newly merged line (like COLMAP)
    const size_t num_merged_recursive =
        Merge(options, merged_id, linker2d, linker3d);
    if (num_merged_recursive > 0) {
      return num_merged_recursive;
    } else {
      return num_merged;
    }
  }

  return 0;
}

size_t IncrementalLineTriangulator::Complete(const Options &options,
                                             line3D_t line3D_id,
                                             const LineLinker &linker) {
  StructureReconstruction &srec = reconstruction_.StructureRecon();
  if (!srec.ExistsLine3D(line3D_id)) {
    return 0;
  }

  Line3d &line3d = srec.Line(line3D_id);

  // Collect all potential observations via transitive correspondences
  Node2dSet existing_obs;
  for (const auto &elem : line3d.track.Elements()) {
    existing_obs.insert(
        Node2d(elem.image_id, static_cast<feature2D_t>(elem.point2D_idx)));
  }

  Node2dSet candidate_obs;
  for (const auto &elem : line3d.track.Elements()) {
    std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
    GetCorrespondences(options, elem.image_id,
                       static_cast<line2D_t>(elem.point2D_idx), &corrs);

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
    const line2D_t line2D_idx = static_cast<line2D_t>(node.second);

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
    if (line2D_idx >= static_cast<line2D_t>(S.NumLines())) {
      continue;
    }

    Line2d &line2D = S.Line(line2D_idx);

    // Skip if already has a 3D line assignment.
    // Note: unlike the earlier attempt to reassign inactive observations,
    // we keep them in their original tracks. Removing an inactive observation
    // from a 2-element track cascades to deleting the entire line (track < 2),
    // which destroys too many lines early in reconstruction.
    if (line2D.HasLine3D()) {
      continue;
    }

    // Check reprojection
    auto proj_res = line3d.Projection(image);
    if (!proj_res.has_value()) {
      continue;
    }
    const Line2d proj_line2d = proj_res.value();

    double score = linker.ComputeScore2d(proj_line2d, line2D);
    if (score <= 0.0) {
      continue;
    }

    // Add observation using observation manager
    const colmap::TrackElement track_el(
        img_id, static_cast<colmap::point2D_t>(line2D_idx));
    obs_manager_->AddLineObservation(line3D_id, track_el);
    num_completed++;
  }

  if (num_completed > 0) {
    AddModifiedLine3D(line3D_id);
  }

  return num_completed;
}

void IncrementalLineTriangulationPipeline(
    const StructureDatabaseCache &structure_db_cache,
    HolisticReconstruction &reconstruction,
    const IncrementalLineTriangulator::Options &options,
    const estimators::PointLineBundleAdjustmentOptions *ba_options) {
  THROW_CHECK(options.Check());

  reconstruction.StructureRecon().Load(structure_db_cache);

  // Reset points and wireframes from point reconstruction (line-only
  // pipelines may not have correct points/wireframes in the structure DB).
  {
    auto &srec = reconstruction.StructureRecon();
    bool warned = false;
    for (const auto &[img_id, image] : reconstruction.PointRecon().Images()) {
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

  reconstruction.StructureRecon().InitializeAllWireframes();

  auto structure_corr_graph = structure_db_cache.StructureCorrespondenceGraph();

  IncrementalLineTriangulator triangulator(structure_corr_graph,
                                           reconstruction);

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
            << " images incrementally (line-only)...";

  size_t total_tris = 0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    total_tris += triangulator.TriangulateImage(options, image_ids[i]);
    if ((i + 1) % 10 == 0 || i == image_ids.size() - 1) {
      LOG(INFO) << "  Processed " << (i + 1) << "/" << image_ids.size()
                << " images, " << total_tris << " triangulations";
    }
  }

  LOG(INFO) << "Completing tracks...";
  const size_t num_completed = triangulator.CompleteAllTracks(options);
  LOG(INFO) << "  Completed " << num_completed << " observations";

  LOG(INFO) << "Merging tracks...";
  const size_t num_merged = triangulator.MergeAllTracks(options);
  LOG(INFO) << "  Merged " << num_merged << " observations";

  const auto &srec = reconstruction.StructureRecon();
  LOG(INFO) << "Incremental line triangulation complete: " << srec.NumLines3D()
            << " lines";

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
    for (const auto &[image_id, image] : reconstruction.PointRecon().Images()) {
      if (image.HasPose())
        ba_config.AddImage(image_id);
    }
    auto adjuster = estimators::CreatePointLineBundleAdjuster(
        opts, std::move(ba_config), reconstruction);
    auto ba_summary = adjuster->Solve();
    const auto &summary =
        static_cast<colmap::CeresBundleAdjustmentSummary &>(*ba_summary)
            .ceres_summary;
    LOG(INFO) << "Bundle adjustment completed: initial_cost="
              << summary.initial_cost << ", final_cost=" << summary.final_cost;
  }
}

} // namespace limap
