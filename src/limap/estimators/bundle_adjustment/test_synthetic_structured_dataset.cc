// Parameterized tests for SynthesizeStructuredDataset:
//   Suite A (FeasibilityCheck): invalid options must throw.
//   Suite B (ExactCountMatch): output counts must exactly match requested.

#include <gtest/gtest.h>

#include <colmap/math/random.h>

#include "limap/estimators/bundle_adjustment/synthetic_structured_dataset.h"
#include "limap/scene/holistic_reconstruction.h"

namespace limap {
namespace estimators {
namespace {

// ============================================================================
// Suite A: FeasibilityCheck — invalid parameters must throw
// ============================================================================

struct FeasibilityCase {
  std::string name;
  SyntheticStructuredDatasetOptions opts;
};

class FeasibilityCheck : public ::testing::TestWithParam<FeasibilityCase> {};

TEST_P(FeasibilityCheck, Throws) {
  colmap::SetPRNGSeed(42);
  const auto &opts = GetParam().opts;
  HolisticReconstruction recon;
  EXPECT_ANY_THROW(SynthesizeStructuredDataset(opts, &recon));
}

// Helper to build options with a fluent style.
FeasibilityCase MakeFeasibilityCase(const std::string &name, int frames,
                                    int pts, int ppp, int lines, int planes,
                                    int wf, double radius = 5.0,
                                    double extent = 3.0) {
  SyntheticStructuredDatasetOptions opts;
  opts.num_frames_per_rig = frames;
  opts.num_points3D = pts;
  opts.num_points_per_plane = ppp;
  opts.num_lines3D = lines;
  opts.num_planes = planes;
  opts.num_wireframe_edges = wf;
  opts.camera_sphere_radius = radius;
  opts.cube_half_extent = extent;
  return {name, opts};
}

INSTANTIATE_TEST_SUITE_P(
    SyntheticDataset, FeasibilityCheck,
    ::testing::Values(
        // ppp * planes > num_points3D
        MakeFeasibilityCase("PlanePtsBudgetExceeded", 50, 2000, 50, 400, 200,
                            500),
        // wf > num_points3D (can't have more edges than points)
        MakeFeasibilityCase("WireframePtsBudgetExceeded", 50, 2000, 10, 400, 20,
                            6000),
        // wf slightly exceeds achievable limit within point budget
        MakeFeasibilityCase("WireframeTooTight", 500, 3000, 10, 400, 60, 4000),
        // wireframe > 0 but too few lines on planes (< 2)
        MakeFeasibilityCase("WireframeNoPlaneLines", 50, 2000, 10, 0, 0, 100),
        // num_frames = 0
        []() {
          SyntheticStructuredDatasetOptions opts;
          opts.num_frames_per_rig = 0;
          return FeasibilityCase{"ZeroFrames", opts};
        }(),
        // camera_sphere_radius <= cube_half_extent
        MakeFeasibilityCase("RadiusTooSmall", 50, 500, 10, 100, 5, 0, 3.0,
                            3.0)),
    [](const ::testing::TestParamInfo<FeasibilityCase> &info) {
      return info.param.name;
    });

// ============================================================================
// Suite B: ExactCountMatch — output counts must match requested parameters
// ============================================================================

struct CountCase {
  std::string name;
  SyntheticStructuredDatasetOptions opts;
};

class ExactCountMatch : public ::testing::TestWithParam<CountCase> {};

TEST_P(ExactCountMatch, MatchesCounts) {
  colmap::SetPRNGSeed(42);
  const auto &c = GetParam();
  const auto &opts = c.opts;

  HolisticReconstruction recon;
  ASSERT_NO_THROW(SynthesizeStructuredDataset(opts, &recon));

  const int expected_images =
      opts.num_rigs * opts.num_cameras_per_rig * opts.num_frames_per_rig;

  EXPECT_EQ(static_cast<int>(recon.PointRecon().NumRegImages()),
            expected_images)
      << "NumRegImages mismatch";
  EXPECT_EQ(static_cast<int>(recon.PointRecon().NumPoints3D()),
            opts.num_points3D)
      << "NumPoints3D mismatch";
  EXPECT_EQ(static_cast<int>(recon.StructureRecon().NumLines3D()),
            opts.num_lines3D)
      << "NumLines3D mismatch";
  EXPECT_EQ(static_cast<int>(recon.StructureRecon().NumGroups3D()),
            opts.num_planes)
      << "NumGroups3D mismatch";
  EXPECT_EQ(static_cast<int>(recon.StructureRecon().Wireframe().CountEdges()),
            opts.num_wireframe_edges)
      << "Wireframe edge count mismatch";
}

CountCase MakeCountCase(const std::string &name, int track, int frames, int pts,
                        int lines, int planes, int ppp, int wf) {
  SyntheticStructuredDatasetOptions opts;
  opts.line_track_length = track;
  opts.point_track_length = track;
  opts.num_frames_per_rig = frames;
  opts.num_points3D = pts;
  opts.num_lines3D = lines;
  opts.num_planes = planes;
  opts.num_points_per_plane = ppp;
  opts.num_wireframe_edges = wf;
  return {name, opts};
}

INSTANTIATE_TEST_SUITE_P(
    SyntheticDataset, ExactCountMatch,
    ::testing::Values(
        MakeCountCase("PointsOnly", 5, 50, 500, 0, 0, 0, 0),
        MakeCountCase("PointsAndLines", 5, 50, 500, 100, 0, 0, 0),
        MakeCountCase("SmallGroups", 5, 50, 500, 100, 10, 10, 0),
        MakeCountCase("LargeGroups", 5, 100, 2000, 400, 40, 50, 0),
        MakeCountCase("SmallWireframe", 5, 100, 2000, 400, 20, 10, 800),
        MakeCountCase("MediumWireframe", 5, 100, 2000, 400, 20, 10, 1600),
        MakeCountCase("LargeWireframe", 5, 100, 4000, 400, 80, 10, 2000),
        MakeCountCase("LongTracks", 20, 250, 2000, 400, 40, 50, 1600),
        // Tight wireframe budget: 1× point count.
        MakeCountCase("TightWireframeBudget", 10, 500, 3000, 400, 60, 10,
                      3000)),
    [](const ::testing::TestParamInfo<CountCase> &info) {
      return info.param.name;
    });

} // namespace
} // namespace estimators
} // namespace limap
