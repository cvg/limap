// Nominal tests for point-line bundle adjustment.
// Follows COLMAP's BA test strategy: synthesize dataset, add noise, run BA,
// verify the solver doesn't crash and returns a usable solution.

#include <gtest/gtest.h>

#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/estimators/cost_functions/reprojection_error.h>
#include <colmap/math/random.h>
#include <colmap/scene/synthetic.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/group_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/synthetic_structured_dataset.h"
#include "limap/scene/holistic_reconstruction.h"

namespace limap {
namespace estimators {
namespace {

// Deep-copy helper (same as benchmark).
HolisticReconstruction CopyRecon(const HolisticReconstruction &src) {
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

// ---------------------------------------------------------------------------
// Diagnostic: verify cost function parameter block sizes are correct.
// If this fails, the template instantiation is broken in our build.
// ---------------------------------------------------------------------------
TEST(CostFunctionDiagnostic, ParameterBlockSizes) {
  const Eigen::Vector2d pt(100.0, 200.0);

  // ReprojErrorCostFunctor<SimpleRadialCameraModel>
  // SimpleRadialCameraModel: num_params=4, has_img_from_cam_with_jac=true
  // Expected: SizedCostFunction<2, 3, 7, 4> → 3 parameter blocks
  std::unique_ptr<ceres::CostFunction> cf_reproj(
      colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
          colmap::SimpleRadialCameraModel::model_id, pt));
  ASSERT_NE(cf_reproj, nullptr);
  EXPECT_EQ(cf_reproj->num_residuals(), 2)
      << "ReprojErrorCostFunctor num_residuals";
  EXPECT_EQ(cf_reproj->parameter_block_sizes().size(), 3u)
      << "ReprojErrorCostFunctor parameter_block_sizes: expected {3,7,4}=" << 3
      << " blocks, got " << cf_reproj->parameter_block_sizes().size();
  if (cf_reproj->parameter_block_sizes().size() >= 1)
    EXPECT_EQ(cf_reproj->parameter_block_sizes()[0], 3);
  if (cf_reproj->parameter_block_sizes().size() >= 2)
    EXPECT_EQ(cf_reproj->parameter_block_sizes()[1], 7);
  if (cf_reproj->parameter_block_sizes().size() >= 3)
    EXPECT_EQ(cf_reproj->parameter_block_sizes()[2], 4);

  // ReprojErrorConstantPoseCostFunctor<SimpleRadialCameraModel>
  // Expected: SizedCostFunction<2, 3, 4> → 2 parameter blocks
  colmap::Rigid3d identity;
  std::unique_ptr<ceres::CostFunction> cf_const_pose(
      colmap::CreateCameraCostFunction<
          colmap::ReprojErrorConstantPoseCostFunctor>(
          colmap::SimpleRadialCameraModel::model_id, pt, identity));
  ASSERT_NE(cf_const_pose, nullptr);
  EXPECT_EQ(cf_const_pose->parameter_block_sizes().size(), 2u)
      << "ReprojErrorConstantPoseCostFunctor: expected 2 blocks, got "
      << cf_const_pose->parameter_block_sizes().size();

  // RigReprojErrorCostFunctor<SimpleRadialCameraModel>
  // Expected: SizedCostFunction<2, 3, 7, 7, 4> → 4 parameter blocks
  std::unique_ptr<ceres::CostFunction> cf_rig(
      colmap::CreateCameraCostFunction<colmap::RigReprojErrorCostFunctor>(
          colmap::SimpleRadialCameraModel::model_id, pt));
  ASSERT_NE(cf_rig, nullptr);
  EXPECT_EQ(cf_rig->parameter_block_sizes().size(), 4u)
      << "RigReprojErrorCostFunctor: expected 4 blocks, got "
      << cf_rig->parameter_block_sizes().size();

  // SimplePinholeCameraModel: num_params=3, has_img_from_cam_with_jac=false
  // Expected: SizedCostFunction<2, 3, 7, 3> → 3 parameter blocks
  std::unique_ptr<ceres::CostFunction> cf_pinhole(
      colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
          colmap::SimplePinholeCameraModel::model_id, pt));
  ASSERT_NE(cf_pinhole, nullptr);
  EXPECT_EQ(cf_pinhole->parameter_block_sizes().size(), 3u)
      << "ReprojErrorCostFunctor<SimplePinhole>: expected 3 blocks, got "
      << cf_pinhole->parameter_block_sizes().size();
}

// ---------------------------------------------------------------------------
// Manual BA: mimics DefaultBundleAdjuster step-by-step.
// If this passes but ColmapBaseline crashes, the bug is inside
// DefaultBundleAdjuster's specific code path.
// ---------------------------------------------------------------------------
TEST(ColmapBaseline, ManualPointBA) {
  colmap::SetPRNGSeed(0);

  colmap::Reconstruction recon;
  colmap::SyntheticDatasetOptions syn_opts;
  syn_opts.num_rigs = 1;
  syn_opts.num_cameras_per_rig = 1;
  syn_opts.num_frames_per_rig = 10;
  syn_opts.num_points3D = 200;
  colmap::SynthesizeDataset(syn_opts, &recon);

  colmap::SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 0.5;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  colmap::SynthesizeNoise(noise_opts, &recon);

  // Verify data properties.
  ASSERT_TRUE(recon.IsValid());
  for (const colmap::image_t img_id : recon.RegImageIds()) {
    ASSERT_TRUE(recon.Image(img_id).IsRefInFrame())
        << "Image " << img_id << " is NOT ref-in-frame";
  }

  // Manually create a Ceres problem and add residual blocks,
  // mimicking DefaultBundleAdjuster::AddImageWithTrivialFrame.
  ceres::Problem problem;
  int num_residuals_added = 0;

  for (const colmap::image_t img_id : recon.RegImageIds()) {
    colmap::Image &image = recon.Image(img_id);
    colmap::Camera &camera = *image.CameraPtr();
    colmap::Rigid3d &rig_from_world = image.FramePtr()->RigFromWorld();

    for (const colmap::Point2D &point2D : image.Points2D()) {
      if (!point2D.HasPoint3D())
        continue;
      colmap::Point3D &point3D = recon.Point3D(point2D.point3D_id);
      if (point3D.track.Length() < 2)
        continue;

      ceres::CostFunction *cost_function =
          colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
              camera.model_id, point2D.xy);

      // Sanity check before AddResidualBlock.
      ASSERT_EQ(cost_function->parameter_block_sizes().size(), 3u)
          << "img=" << img_id << " model=" << static_cast<int>(camera.model_id)
          << " cost_function has "
          << cost_function->parameter_block_sizes().size() << " blocks";

      problem.AddResidualBlock(cost_function, nullptr, point3D.xyz.data(),
                               rig_from_world.params.data(),
                               camera.params.data());
      ++num_residuals_added;
    }
  }

  EXPECT_GT(num_residuals_added, 0);
}

// ---------------------------------------------------------------------------
// Baseline: COLMAP's own SynthesizeDataset + COLMAP's BA.
// If this fails, our COLMAP build is broken.
// ---------------------------------------------------------------------------
TEST(ColmapBaseline, PointBA) {
  colmap::SetPRNGSeed(0);

  colmap::Reconstruction gt_recon;
  colmap::SyntheticDatasetOptions syn_opts;
  syn_opts.num_rigs = 1;
  syn_opts.num_cameras_per_rig = 1;
  syn_opts.num_frames_per_rig = 10;
  syn_opts.num_points3D = 200;
  colmap::SynthesizeDataset(syn_opts, &gt_recon);

  colmap::Reconstruction recon = gt_recon;
  colmap::SyntheticNoiseOptions noise_opts;
  noise_opts.point2D_stddev = 0.5;
  noise_opts.point3D_stddev = 0.1;
  noise_opts.rig_from_world_rotation_stddev = 0.5;
  noise_opts.rig_from_world_translation_stddev = 0.1;
  colmap::SynthesizeNoise(noise_opts, &recon);

  colmap::BundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon.RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  colmap::BundleAdjustmentOptions options;
  options.print_summary = false;
  auto ba = colmap::CreateDefaultBundleAdjuster(options, config, recon);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

// ---------------------------------------------------------------------------
// Diagnostic: verify synthetic data validity before running BA.
// ---------------------------------------------------------------------------
TEST(SyntheticDataDiagnostic, DataIsValid) {
  colmap::SetPRNGSeed(42);

  SyntheticStructuredDatasetOptions opts;
  opts.num_rigs = 1;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 10;
  opts.num_points3D = 100;
  opts.num_lines3D = 20;
  opts.num_planes = 3;
  opts.num_points_per_plane = 10;
  opts.num_wireframe_edges = 10;
  opts.point_track_length = 5;
  opts.line_track_length = 5;

  HolisticReconstruction recon;
  ASSERT_NO_THROW(SynthesizeStructuredDataset(opts, &recon));

  const auto &point_recon = recon.PointRecon();

  // Verify COLMAP's internal consistency checks pass.
  EXPECT_TRUE(point_recon.IsValid())
      << "Reconstruction::IsValid() failed — rig/frame/image structure broken";

  // Verify all images report as ref sensor.
  for (const colmap::image_t img_id : point_recon.RegImageIds()) {
    const auto &image = point_recon.Image(img_id);
    EXPECT_TRUE(image.HasFramePtr())
        << "Image " << img_id << " has no FramePtr";
    EXPECT_TRUE(image.HasCameraPtr())
        << "Image " << img_id << " has no CameraPtr";
    EXPECT_TRUE(image.IsRefInFrame())
        << "Image " << img_id
        << " is NOT ref sensor (camera_id=" << image.CameraId()
        << ", frame_id=" << image.FrameId() << ")";
  }

  // Verify track lengths >= 2 (required by COLMAP's AddResidualBlock check).
  for (const auto &[pt_id, pt] : point_recon.Points3D()) {
    EXPECT_GE(pt.track.Length(), 2u)
        << "Point3D " << pt_id << " has track length " << pt.track.Length();
  }
}

// ---------------------------------------------------------------------------
// Test our synthetic data with COLMAP's own BA.
// If ColmapBaseline passes but this fails, our data is wrong.
// ---------------------------------------------------------------------------
TEST(SyntheticDataDiagnostic, PointBAWithOurData) {
  colmap::SetPRNGSeed(42);

  SyntheticStructuredDatasetOptions opts;
  opts.num_rigs = 1;
  opts.num_cameras_per_rig = 1;
  opts.num_frames_per_rig = 10;
  opts.num_points3D = 100;
  opts.num_lines3D = 0;
  opts.num_planes = 0;
  opts.num_points_per_plane = 0;
  opts.num_wireframe_edges = 0;
  opts.point_track_length = 5;
  opts.line_track_length = 5;

  HolisticReconstruction recon;
  SynthesizeStructuredDataset(opts, &recon);

  SyntheticStructuredNoiseOptions noise;
  noise.point2D_stddev = 1.0;
  noise.point3D_stddev = 0.05;
  noise.pose_translation_stddev = 0.01;
  noise.pose_rotation_stddev = 1.0;
  SynthesizeStructuredNoise(noise, &recon);

  colmap::Reconstruction recon_copy = recon.PointRecon();

  // Verify data is valid.
  ASSERT_TRUE(recon_copy.IsValid());

  // Verify all images are ref sensors.
  for (const colmap::image_t img_id : recon_copy.RegImageIds()) {
    ASSERT_TRUE(recon_copy.Image(img_id).IsRefInFrame())
        << "Image " << img_id << " not ref sensor after copy";
  }

  colmap::BundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  colmap::BundleAdjustmentOptions options;
  options.print_summary = false;
  options.ceres->solver_options.max_num_iterations = 10;

  auto ba = colmap::CreateDefaultBundleAdjuster(options, config, recon_copy);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

// ---------------------------------------------------------------------------
// Shared fixture: synthesizes a full holistic reconstruction with points,
// lines, groups (planes), and wireframe edges.  All tests run on deep copies.
// ---------------------------------------------------------------------------
class BundleAdjustmentTest : public ::testing::Test {
protected:
  void SetUp() override {
    colmap::SetPRNGSeed(42);

    SyntheticStructuredDatasetOptions opts;
    opts.num_rigs = 1;
    opts.num_cameras_per_rig = 1;
    opts.num_frames_per_rig = 30;
    opts.num_points3D = 500;
    opts.num_lines3D = 100;
    opts.num_planes = 6;
    opts.num_points_per_plane = 20;
    opts.num_wireframe_edges = 50;
    opts.point_track_length = 5;
    opts.line_track_length = 5;

    recon_ = std::make_unique<HolisticReconstruction>();
    SynthesizeStructuredDataset(opts, recon_.get());

    SyntheticStructuredNoiseOptions noise;
    noise.point2D_stddev = 1.0;
    noise.point3D_stddev = 0.05;
    noise.pose_translation_stddev = 0.01;
    noise.pose_rotation_stddev = 1.0;
    noise.line2D_endpoint_stddev = 1.0;
    noise.line3D_endpoint_stddev = 0.05;
    noise.plane_normal_stddev = 0.02;
    noise.plane_offset_stddev = 0.05;
    SynthesizeStructuredNoise(noise, recon_.get());
  }

  std::unique_ptr<HolisticReconstruction> recon_;
};

// ---------------------------------------------------------------------------
// Mode 0: COLMAP's DefaultCeresBundleAdjuster (point-only BA)
// ---------------------------------------------------------------------------
TEST_F(BundleAdjustmentTest, PointBA) {
  colmap::Reconstruction recon_copy = recon_->PointRecon();

  colmap::BundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  colmap::BundleAdjustmentOptions options;
  options.print_summary = false;
  options.ceres->solver_options.max_num_iterations = 10;

  auto ba = colmap::CreateDefaultBundleAdjuster(options, config, recon_copy);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

// ---------------------------------------------------------------------------
// Mode 1: PointLineBA
// ---------------------------------------------------------------------------
TEST_F(BundleAdjustmentTest, PointLineBA) {
  auto recon_copy = CopyRecon(*recon_);

  PointLineBundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.PointRecon().RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto &[lid, _] : recon_copy.StructureRecon().Lines3D()) {
    config.AddVariableLine(lid);
  }

  PointLineBundleAdjustmentOptions options;
  options.print_summary = false;
  options.custom_function_tolerance = 0.0;
  options.ceres->solver_options.max_num_iterations = 10;

  auto ba = CreatePointLineBundleAdjuster(options, config, recon_copy);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

// ---------------------------------------------------------------------------
// Mode 2: GroupBA (points + lines + groups)
// ---------------------------------------------------------------------------
TEST_F(BundleAdjustmentTest, GroupBA) {
  auto recon_copy = CopyRecon(*recon_);

  GroupBundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.PointRecon().RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto &[lid, _] : recon_copy.StructureRecon().Lines3D()) {
    config.AddVariableLine(lid);
  }
  for (const auto &[gid, _] : recon_copy.StructureRecon().Groups3D()) {
    config.AddVariableGroup(gid);
  }

  GroupBundleAdjustmentOptions options;
  options.print_summary = false;
  options.custom_function_tolerance = 0.0;
  options.ceres->solver_options.max_num_iterations = 10;
  options.min_active_group_associations = 0;
  options.enforce_plane_angular_constraints = false;
  options.enforce_vp_angular_constraints = false;

  auto ba = CreateGroupBundleAdjuster(options, config, recon_copy);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

// ---------------------------------------------------------------------------
// Mode 3: StructureBA (points + lines + groups + wireframe)
// ---------------------------------------------------------------------------
TEST_F(BundleAdjustmentTest, StructureBA) {
  auto recon_copy = CopyRecon(*recon_);

  GroupBundleAdjustmentConfig config;
  for (const colmap::image_t img_id : recon_copy.PointRecon().RegImageIds()) {
    config.AddImage(img_id);
  }
  config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  for (const auto &[lid, _] : recon_copy.StructureRecon().Lines3D()) {
    config.AddVariableLine(lid);
  }
  for (const auto &[gid, _] : recon_copy.StructureRecon().Groups3D()) {
    config.AddVariableGroup(gid);
  }

  StructureBundleAdjustmentOptions options;
  options.print_summary = false;
  options.custom_function_tolerance = 0.0;
  options.ceres->solver_options.max_num_iterations = 10;
  options.min_active_group_associations = 0;
  options.enforce_plane_angular_constraints = false;
  options.enforce_vp_angular_constraints = false;
  options.force_reconstruct_wireframe3d = false;

  auto ba = CreateStructureBundleAdjuster(options, config, recon_copy);
  auto summary = ba->Solve();
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(summary->IsSolutionUsable());
  EXPECT_GT(summary->num_residuals, 0);
}

} // namespace
} // namespace estimators
} // namespace limap
