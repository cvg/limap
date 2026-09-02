#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/group_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/synthetic_structured_dataset.h"
#include "limap/scene/holistic_reconstruction.h"

#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/math/random.h>
#include <colmap/scene/reconstruction.h>

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>

using namespace limap;

// Global linear solver type, set via --solver={sparse,dense} CLI flag.
static ceres::LinearSolverType g_solver_type = ceres::SPARSE_SCHUR;

// ---------------------------------------------------------------------------
// Quick PointBA solve to check if a synthetic dataset is "hard enough"
// (i.e. PointBA does not converge early).  Returns the number of Ceres
// iterations (successful + unsuccessful).
// ---------------------------------------------------------------------------
static int QuickPointBAIterations(const HolisticReconstruction &recon) {
  colmap::Reconstruction recon_copy = recon.PointRecon();
  colmap::BundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  colmap::BundleAdjustmentOptions ba_options;
  ba_options.print_summary = false;
  ba_options.ceres->auto_select_solver_type = false;
  ba_options.ceres->solver_options.linear_solver_type = g_solver_type;
  ba_options.ceres->solver_options.max_num_iterations = 20;
  ba_options.ceres->solver_options.function_tolerance = 0;
  ba_options.ceres->solver_options.gradient_tolerance = 0;
  ba_options.ceres->solver_options.parameter_tolerance = 0;
  ba_options.ceres->solver_options.min_trust_region_radius = 1e-64;
  ba_options.ceres->solver_options.max_num_consecutive_invalid_steps = 20;
  auto ba = colmap::CreateDefaultBundleAdjuster(ba_options, config, recon_copy);
  auto summary_ptr = ba->Solve();
  auto ceres_summary =
      std::static_pointer_cast<colmap::CeresBundleAdjustmentSummary>(
          summary_ptr);
  const auto &s = ceres_summary->ceres_summary;
  return s.num_successful_steps + s.num_unsuccessful_steps;
}
using namespace limap::estimators;

// BA modes — last arg in each benchmark case.
enum BAMode {
  kPointBA = 0,
  kPointLineBA = 1,
  kGroupBA = 2,
  kStructureBA = 3,
};

// ============================================================================
// Deep-copy helper
// ============================================================================
HolisticReconstruction CopyReconstruction(const HolisticReconstruction &src) {
  HolisticReconstruction dst(src.PointRecon());
  for (const auto &[id, s2d] : src.StructureRecon().Structures2d()) {
    dst.StructureRecon().AddStructure2D(id, s2d);
  }
  for (const auto &[id, line] : src.StructureRecon().Lines3D()) {
    dst.StructureRecon().AddLine3D(id, static_cast<const Line3d &>(line));
  }
  for (const auto &[id, group] : src.StructureRecon().Groups3D()) {
    dst.StructureRecon().AddGroup3D(id, static_cast<const Group3d &>(group));
  }
  for (const auto &edge : src.StructureRecon().Wireframe().GetAllEdges()) {
    dst.StructureRecon().Wireframe().AddEdge(edge);
  }
  return dst;
}

// ============================================================================
// Report custom counters.
// ============================================================================
void ReportCounters(benchmark::State &state,
                    const HolisticReconstruction &recon, int num_iterations,
                    int total_benchmark_iters) {
  state.counters["imgs"] = recon.PointRecon().NumRegImages();
  state.counters["pnts"] = recon.PointRecon().NumPoints3D();
  state.counters["lns"] = recon.StructureRecon().NumLines3D();
  state.counters["grps"] = recon.StructureRecon().NumGroups3D();
  state.counters["wf_edges"] = recon.StructureRecon().Wireframe().CountEdges();
  // Count junction points (points referenced by wireframe edges).
  FlatHashSet<point3D_t> junc_pts;
  for (const auto &edge : recon.StructureRecon().Wireframe().GetAllEdges()) {
    junc_pts.insert(edge.point_idx);
  }
  state.counters["junc_pnts"] = junc_pts.size();
  if (total_benchmark_iters > 0) {
    state.counters["avg_itrs"] =
        std::round(num_iterations * 10.0 / total_benchmark_iters) / 10.0;
  }
}

// ============================================================================
// Helper: configure BA configs.
// ============================================================================
PointLineBundleAdjustmentConfig
MakeFullConfig(const HolisticReconstruction &recon) {
  PointLineBundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon.PointRecon().RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto &[lid, _] : recon.StructureRecon().Lines3D()) {
    config.AddVariableLine(lid);
  }
  return config;
}

GroupBundleAdjustmentConfig
MakeFullGroupConfig(const HolisticReconstruction &recon) {
  GroupBundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon.PointRecon().RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto &[lid, _] : recon.StructureRecon().Lines3D()) {
    config.AddVariableLine(lid);
  }
  for (const auto &[gid, _] : recon.StructureRecon().Groups3D()) {
    config.AddVariableGroup(gid);
  }
  return config;
}

// ============================================================================
// All sweeps use the same arg layout:
// Args: {track, rigs, cams, frms, pnts, lns, planes, wf_edges, ppp, mode}
//
// Reference scene defined by kRef* constants below.
// Each sweep applies a scale factor to one variable; others at reference.
// Changing kRef* values automatically updates all sweeps.
// ============================================================================

static constexpr int kRefTrack = 6;
static constexpr int kRefPpp = 10;
static constexpr int kRefFrms = 1000;
static constexpr int kRefPnts = kRefFrms * 10;
static constexpr int kRefLns = kRefFrms * 1;
static constexpr int kRefPlanes = kRefFrms / 5;
static constexpr int kRefWf = kRefLns * 3;

// Scale helper: multiply reference by scale factor, clamp to >= 1.
static int Scale(double s, int ref) {
  return std::max(1, static_cast<int>(std::round(s * ref)));
}

// ============================================================================
// Image scaling: all scene parameters scale uniformly.
//   Scales: 0.1x .. 8x of reference.  All 5 modes.
// ============================================================================
void GenerateImageScalingArgs(benchmark::internal::Benchmark *b) {
  for (int track : {4, 6, 8}) {
    for (double s : {0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0}) {
      const int frms = Scale(s, kRefFrms);
      const int pnts = Scale(s, kRefPnts);
      const int lns = Scale(s, kRefLns);
      const int planes = Scale(s, kRefPlanes);
      const int wf_edges = Scale(s, kRefWf);
      for (int mode = kPointBA; mode <= kStructureBA; ++mode) {
        b->Args(
            {track, 1, 1, frms, pnts, lns, planes, wf_edges, kRefPpp, mode});
      }
    }
  }
}

// ============================================================================
// Feature scaling: sweep pnts & lns (proportionally), all else at reference.
//   Scales: 0.25x, 0.5x, 2.5x, 5x.  All 5 modes.
// ============================================================================
void GenerateFeatureScalingArgs(benchmark::internal::Benchmark *b) {
  for (double s : {0.25, 0.5, 2.5, 5.0}) {
    const int pnts = Scale(s, kRefPnts);
    const int lns = Scale(s, kRefLns);
    for (int mode = kPointBA; mode <= kStructureBA; ++mode) {
      b->Args({kRefTrack, 1, 1, kRefFrms, pnts, lns, kRefPlanes, kRefWf,
               kRefPpp, mode});
    }
  }
}

// ============================================================================
// Plane scaling: sweep planes, all else at reference.
//   Scales: 0.25x, 0.5x, 2.5x, 5x.  GroupBA & StructureBA only.
// ============================================================================
void GeneratePlaneScalingArgs(benchmark::internal::Benchmark *b) {
  for (double s : {0.25, 0.5, 2.5, 5.0}) {
    const int planes = Scale(s, kRefPlanes);
    b->Args({kRefTrack, 1, 1, kRefFrms, kRefPnts, kRefLns, planes, kRefWf,
             kRefPpp, kGroupBA});
    b->Args({kRefTrack, 1, 1, kRefFrms, kRefPnts, kRefLns, planes, kRefWf,
             kRefPpp, kStructureBA});
  }
}

// ============================================================================
// Benchmark fixture
// ============================================================================
// Static single-entry cache: synthesis is deterministic (same seed + same
// args → same data).  Caches one reconstruction at a time so different modes
// sharing the same scene parameters reuse the same synthesized data.
// Evicts when args change to avoid unbounded memory growth.
// ============================================================================
using CacheKey = std::array<int64_t, 9>;
static std::optional<std::pair<CacheKey, HolisticReconstruction>> g_synth_cache;

class BM_BA : public benchmark::Fixture {
public:
  void SetUp(::benchmark::State &state) {
    CacheKey key;
    for (int i = 0; i < 9; ++i)
      key[i] = state.range(i);

    if (g_synth_cache && g_synth_cache->first == key) {
      reconstruction_ = std::make_unique<HolisticReconstruction>(
          CopyReconstruction(g_synth_cache->second));
      return;
    }

    SyntheticStructuredDatasetOptions dataset_options;
    dataset_options.point_track_length = key[0];
    dataset_options.line_track_length = key[0];
    dataset_options.num_rigs = key[1];
    dataset_options.num_cameras_per_rig = key[2];
    dataset_options.num_frames_per_rig = key[3];
    dataset_options.num_points3D = key[4];
    dataset_options.num_lines3D = key[5];
    dataset_options.num_planes = key[6];
    dataset_options.num_points_per_plane = key[8];
    dataset_options.num_wireframe_edges = key[7];

    SyntheticStructuredNoiseOptions noise_options;
    noise_options.point2D_stddev = 1.0;
    noise_options.point3D_stddev = 0.1;
    noise_options.pose_translation_stddev = 0.05;
    noise_options.pose_rotation_stddev = 1.0;
    noise_options.line2D_endpoint_stddev = 1.0;
    noise_options.line3D_endpoint_stddev = 0.1;
    noise_options.plane_normal_stddev = 0.02;
    noise_options.plane_offset_stddev = 0.1;

    // Retry with different seeds until PointBA reaches 20 iterations
    // (i.e. does not converge early), so that per-iteration timing is fair.
    constexpr int kMinIters = 20;
    constexpr int kMaxRetries = 20;
    int seed = 42;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
      colmap::SetPRNGSeed(seed);
      // Construct fresh each time (HolisticReconstruction has a reference
      // member in StructureReconstruction, so assignment is not safe).
      g_synth_cache.emplace(key, HolisticReconstruction());
      SynthesizeStructuredDataset(dataset_options, &g_synth_cache->second);
      SynthesizeStructuredNoise(noise_options, &g_synth_cache->second);
      const int iters = QuickPointBAIterations(g_synth_cache->second);
      if (iters >= kMinIters) {
        break;
      }
      LOG(WARNING) << "PointBA converged in " << iters
                   << " iters (seed=" << seed << "), retrying...";
      g_synth_cache.reset();
      ++seed;
    }

    // Deep-copy for this run.
    reconstruction_ = std::make_unique<HolisticReconstruction>(
        CopyReconstruction(g_synth_cache->second));
  }

  void TearDown(::benchmark::State &) { reconstruction_.reset(); }

protected:
  void RunBA(benchmark::State &state);
  std::unique_ptr<HolisticReconstruction> reconstruction_;
};

// ============================================================================
// Shared benchmark body — dispatches on mode (arg index 9).
// ============================================================================
void BM_BA::RunBA(benchmark::State &state) {
  const int mode = state.range(9);
  int total_iters = 0;
  int bench_iters = 0;
  double total_first_iter_ms = 0.0;
  double total_rest_iter_ms = 0.0;
  int rest_iter_count = 0;

  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<colmap::BundleAdjustmentSummary> summary_ptr;

    if (mode == kPointBA) {
      colmap::Reconstruction recon_copy = reconstruction_->PointRecon();
      colmap::BundleAdjustmentConfig config;
      for (const colmap::image_t img_id : recon_copy.RegImageIds()) {
        config.AddImage(img_id);
      }
      config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
      colmap::BundleAdjustmentOptions ba_options;
      ba_options.print_summary = false;
      ba_options.ceres->auto_select_solver_type = false;
      ba_options.ceres->solver_options.linear_solver_type = g_solver_type;
      ba_options.ceres->solver_options.max_num_iterations = 20;
      ba_options.ceres->solver_options.function_tolerance = 0;
      ba_options.ceres->solver_options.gradient_tolerance = 0;
      ba_options.ceres->solver_options.parameter_tolerance = 0;
      ba_options.ceres->solver_options.min_trust_region_radius = 1e-64;
      ba_options.ceres->solver_options.max_num_consecutive_invalid_steps = 20;
      state.ResumeTiming();
      auto ba =
          colmap::CreateDefaultBundleAdjuster(ba_options, config, recon_copy);
      summary_ptr = ba->Solve();

    } else if (mode == kPointLineBA) {
      auto recon_copy = CopyReconstruction(*reconstruction_);
      auto config = MakeFullConfig(*reconstruction_);
      PointLineBundleAdjustmentOptions options;
      options.print_summary = false;
      options.ceres->solver_options.linear_solver_type = g_solver_type;

      options.custom_function_tolerance = 0.0;
      options.ceres->solver_options.max_num_iterations = 20;
      options.ceres->solver_options.function_tolerance = 0;
      options.ceres->solver_options.gradient_tolerance = 0;
      options.ceres->solver_options.parameter_tolerance = 0;
      options.ceres->solver_options.min_trust_region_radius = 1e-64;
      options.ceres->solver_options.max_num_consecutive_invalid_steps = 20;
      state.ResumeTiming();
      auto ba = CreatePointLineBundleAdjuster(options, config, recon_copy);
      summary_ptr = ba->Solve();

    } else if (mode == kGroupBA) {
      auto recon_copy = CopyReconstruction(*reconstruction_);
      auto config = MakeFullGroupConfig(*reconstruction_);
      GroupBundleAdjustmentOptions options;
      options.print_summary = false;
      options.ceres->solver_options.linear_solver_type = g_solver_type;

      options.custom_function_tolerance = 0.0;
      options.ceres->solver_options.max_num_iterations = 20;
      options.ceres->solver_options.function_tolerance = 0;
      options.ceres->solver_options.gradient_tolerance = 0;
      options.ceres->solver_options.parameter_tolerance = 0;
      options.ceres->solver_options.min_trust_region_radius = 1e-64;
      options.ceres->solver_options.max_num_consecutive_invalid_steps = 20;
      options.min_active_group_associations = 0;
      options.enforce_plane_angular_constraints = false;
      options.enforce_vp_angular_constraints = false;
      state.ResumeTiming();
      auto ba = CreateGroupBundleAdjuster(options, config, recon_copy);
      summary_ptr = ba->Solve();

    } else if (mode == kStructureBA) {
      auto recon_copy = CopyReconstruction(*reconstruction_);
      auto config = MakeFullGroupConfig(*reconstruction_);
      StructureBundleAdjustmentOptions options;
      options.print_summary = false;
      options.ceres->solver_options.linear_solver_type = g_solver_type;

      options.custom_function_tolerance = 0.0;
      options.ceres->solver_options.max_num_iterations = 20;
      options.ceres->solver_options.function_tolerance = 0;
      options.ceres->solver_options.gradient_tolerance = 0;
      options.ceres->solver_options.parameter_tolerance = 0;
      options.ceres->solver_options.min_trust_region_radius = 1e-64;
      options.ceres->solver_options.max_num_consecutive_invalid_steps = 20;
      options.min_active_group_associations = 0;
      options.enforce_plane_angular_constraints = false;
      options.enforce_vp_angular_constraints = false;
      options.force_reconstruct_wireframe3d = false;
      state.ResumeTiming();
      auto ba = CreateStructureBundleAdjuster(options, config, recon_copy);
      summary_ptr = ba->Solve();
    }

    state.PauseTiming();
    auto ceres_summary_ptr =
        std::static_pointer_cast<colmap::CeresBundleAdjustmentSummary>(
            summary_ptr);
    const auto &summary = ceres_summary_ptr->ceres_summary;
    const int iters =
        summary.num_successful_steps + summary.num_unsuccessful_steps;
    total_iters += iters;
    bench_iters++;
    if (iters > 0) {
      state.SetIterationTime(summary.total_time_in_seconds / iters);
    }
    // Accumulate first-iteration vs rest timing.
    const auto &iter_summaries = summary.iterations;
    if (!iter_summaries.empty()) {
      total_first_iter_ms += iter_summaries[0].iteration_time_in_seconds * 1e3;
      for (size_t i = 1; i < iter_summaries.size(); ++i) {
        total_rest_iter_ms += iter_summaries[i].iteration_time_in_seconds * 1e3;
        rest_iter_count++;
      }
    }
    state.ResumeTiming();
  }

  state.PauseTiming();
  ReportCounters(state, *reconstruction_, total_iters, bench_iters);
  state.counters["mode"] = mode;
  if (bench_iters > 0) {
    state.counters["1st_iter_ms"] = total_first_iter_ms / bench_iters;
  }
  if (rest_iter_count > 0) {
    state.counters["rest_iter_ms"] = total_rest_iter_ms / rest_iter_count;
  }
  state.ResumeTiming();
}

// ============================================================================
// Benchmark definitions — all delegate to RunBA.
// ============================================================================
BENCHMARK_DEFINE_F(BM_BA, ImageScaling)(benchmark::State &state) {
  RunBA(state);
}
BENCHMARK_DEFINE_F(BM_BA, FeatureScaling)(benchmark::State &state) {
  RunBA(state);
}
BENCHMARK_DEFINE_F(BM_BA, PlaneScaling)(benchmark::State &state) {
  RunBA(state);
}

// ============================================================================
// Register
// ============================================================================

BENCHMARK_REGISTER_F(BM_BA, ImageScaling)
    ->Apply(GenerateImageScalingArgs)
    ->ArgNames({"track", "rigs", "cams", "frms", "pnts", "lns", "planes",
                "wf_edges", "ppp", "mode"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_REGISTER_F(BM_BA, FeatureScaling)
    ->Apply(GenerateFeatureScalingArgs)
    ->ArgNames({"track", "rigs", "cams", "frms", "pnts", "lns", "planes",
                "wf_edges", "ppp", "mode"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_REGISTER_F(BM_BA, PlaneScaling)
    ->Apply(GeneratePlaneScalingArgs)
    ->ArgNames({"track", "rigs", "cams", "frms", "pnts", "lns", "planes",
                "wf_edges", "ppp", "mode"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

int main(int argc, char **argv) {
  // Parse --solver={sparse,dense} before benchmark::Initialize consumes flags.
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--solver=dense") {
      g_solver_type = ceres::DENSE_SCHUR;
      // Remove from argv so benchmark doesn't complain.
      for (int j = i; j < argc - 1; ++j)
        argv[j] = argv[j + 1];
      --argc;
      --i;
    } else if (arg == "--solver=sparse") {
      g_solver_type = ceres::SPARSE_SCHUR;
      for (int j = i; j < argc - 1; ++j)
        argv[j] = argv[j + 1];
      --argc;
      --i;
    }
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  const char *solver_name =
      g_solver_type == ceres::DENSE_SCHUR ? "DENSE_SCHUR" : "SPARSE_SCHUR";
  std::cerr << "\033[1mNote: Time column reports time (ms) per solver "
               "iteration.\033[0m"
            << std::endl;
  std::cerr << "Solver: " << solver_name << std::endl;
  std::cerr << "Mode: 0=PointBA, 1=PointLineBA, 2=GroupBA, 3=StructureBA"
            << std::endl;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
