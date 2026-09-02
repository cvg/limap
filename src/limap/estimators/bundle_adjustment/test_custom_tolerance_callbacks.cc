// Tests for AngleCostAwareToleranceCallback: validates that the manual angle
// cost computation matches Ceres's Problem::Evaluate() for the same residuals,
// and that the angle-stripping convergence logic works correctly.

#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <colmap/math/random.h>

#include "limap/estimators/bundle_adjustment/custom_tolerance_callbacks.h"
#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/line3d.h"
#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {
namespace estimators {
namespace {

// Test that ComputeAngleCost() matches Ceres Problem::Evaluate() for
// VP line-to-VP (sine) + orthogonality + parallelism residuals.
TEST(AngleCostCallback, ManualCostMatchesCeresEvaluate) {
  colmap::SetPRNGSeed(42);

  // Create a Ceres problem to hold the same residuals
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  // Loss functions (matching typical BA settings)
  ceres::CauchyLoss angle_loss(0.2);
  ceres::CauchyLoss angular_constraint_loss(0.1);

  // Weights
  const double weight_vp = 5e3;
  const double weight_angular = 1e5;

  // Generate random VP directions (unit vectors)
  constexpr int kNumVPs = 4;
  double vp_params[kNumVPs][3];
  for (int i = 0; i < kNumVPs; ++i) {
    Eigen::Vector3d v = Eigen::Vector3d::Random().normalized();
    vp_params[i][0] = v.x();
    vp_params[i][1] = v.y();
    vp_params[i][2] = v.z();
  }

  // Generate random lines as MinimalPlucker params
  constexpr int kNumLines = 5;
  std::vector<MinimalInfiniteLine3d> line_minimals(kNumLines);
  for (int i = 0; i < kNumLines; ++i) {
    Eigen::Vector3d p1 = Eigen::Vector3d::Random() * 5.0;
    Eigen::Vector3d dir = Eigen::Vector3d::Random().normalized();
    Eigen::Vector3d p2 = p1 + dir * colmap::RandomUniformReal(1.0, 5.0);
    line_minimals[i] = MinimalInfiniteLine3d(Line3d(p1, p2));
  }

  // Generate random plane normals (4D: [nx, ny, nz, d])
  constexpr int kNumPlanes = 4;
  double plane_params[kNumPlanes][4];
  for (int i = 0; i < kNumPlanes; ++i) {
    Eigen::Vector3d n = Eigen::Vector3d::Random().normalized();
    plane_params[i][0] = n.x();
    plane_params[i][1] = n.y();
    plane_params[i][2] = n.z();
    plane_params[i][3] = colmap::RandomUniformReal(-5.0, 5.0);
  }

  // Collect angle residual infos and corresponding Ceres residual block IDs
  std::vector<AngleResidualInfo> angle_infos;
  std::vector<ceres::ResidualBlockId> angle_block_ids;

  // Storage for ScaledLoss instances
  std::vector<std::unique_ptr<ceres::LossFunction>> losses;

  // 1. Add VP line-to-VP (sine) residuals: lines 0-2 -> VP 0, lines 3-4 -> VP 1
  auto add_vp_line = [&](int line_idx, int vp_idx) {
    losses.push_back(std::make_unique<ceres::ScaledLoss>(
        &angle_loss, weight_vp, ceres::DO_NOT_TAKE_OWNERSHIP));
    ceres::CostFunction *cost = LineToVPSineCostFunctor::Create();
    auto block_id = problem.AddResidualBlock(
        cost, losses.back().get(), line_minimals[line_idx].data.data(),
        vp_params[vp_idx]);
    angle_block_ids.push_back(block_id);
    angle_infos.push_back({AngleResidualInfo::PARALLELISM,
                           line_minimals[line_idx].data.data(),
                           vp_params[vp_idx], weight_vp, &angle_loss});
  };

  add_vp_line(0, 0);
  add_vp_line(1, 0);
  add_vp_line(2, 0);
  add_vp_line(3, 1);
  add_vp_line(4, 1);

  // 2. Add VP orthogonality residuals: VP0-VP1, VP0-VP2
  auto add_vp_ortho = [&](int vp_i, int vp_j) {
    losses.push_back(std::make_unique<ceres::ScaledLoss>(
        &angular_constraint_loss, weight_angular,
        ceres::DO_NOT_TAKE_OWNERSHIP));
    ceres::CostFunction *cost = VPOrthogonalityCostFunctor::Create();
    auto block_id = problem.AddResidualBlock(cost, losses.back().get(),
                                             vp_params[vp_i], vp_params[vp_j]);
    angle_block_ids.push_back(block_id);
    angle_infos.push_back({AngleResidualInfo::ORTHOGONALITY, vp_params[vp_i],
                           vp_params[vp_j], weight_angular,
                           &angular_constraint_loss});
  };

  add_vp_ortho(0, 1);
  add_vp_ortho(0, 2);

  // 3. Add VP parallelism residual: VP2-VP3
  {
    losses.push_back(std::make_unique<ceres::ScaledLoss>(
        &angular_constraint_loss, weight_angular,
        ceres::DO_NOT_TAKE_OWNERSHIP));
    ceres::CostFunction *cost = VPParallelismCostFunctor::Create();
    auto block_id = problem.AddResidualBlock(cost, losses.back().get(),
                                             vp_params[2], vp_params[3]);
    angle_block_ids.push_back(block_id);
    angle_infos.push_back({AngleResidualInfo::VP_PARALLELISM, vp_params[2],
                           vp_params[3], weight_angular,
                           &angular_constraint_loss});
  }

  // 4. Add plane orthogonality residual: plane0-plane1
  {
    losses.push_back(std::make_unique<ceres::ScaledLoss>(
        &angular_constraint_loss, weight_angular,
        ceres::DO_NOT_TAKE_OWNERSHIP));
    ceres::CostFunction *cost = PlaneNormalOrthogonalityCostFunctor::Create();
    auto block_id = problem.AddResidualBlock(cost, losses.back().get(),
                                             plane_params[0], plane_params[1]);
    angle_block_ids.push_back(block_id);
    angle_infos.push_back({AngleResidualInfo::ORTHOGONALITY, plane_params[0],
                           plane_params[1], weight_angular,
                           &angular_constraint_loss});
  }

  // 5. Add plane parallelism residual: plane2-plane3
  {
    losses.push_back(std::make_unique<ceres::ScaledLoss>(
        &angular_constraint_loss, weight_angular,
        ceres::DO_NOT_TAKE_OWNERSHIP));
    ceres::CostFunction *cost = PlaneNormalParallelismCostFunctor::Create();
    auto block_id = problem.AddResidualBlock(cost, losses.back().get(),
                                             plane_params[2], plane_params[3]);
    angle_block_ids.push_back(block_id);
    angle_infos.push_back({AngleResidualInfo::PLANE_PARALLELISM,
                           plane_params[2], plane_params[3], weight_angular,
                           &angular_constraint_loss});
  }

  // Compute angle cost via our callback's ComputeAngleCost()
  AngleCostAwareToleranceCallback callback(1e-5, angle_infos);
  const double manual_cost = callback.ComputeAngleCost();

  // Compute angle cost via Ceres Problem::Evaluate()
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.residual_blocks = angle_block_ids;
  double ceres_cost = 0.0;
  problem.Evaluate(eval_options, &ceres_cost, nullptr, nullptr, nullptr);

  // Compare: should match within floating-point tolerance
  EXPECT_NEAR(manual_cost, ceres_cost, 1e-10)
      << "Manual angle cost = " << manual_cost
      << ", Ceres evaluate = " << ceres_cost;
}

// Test with cosine VP residual type
TEST(AngleCostCallback, CosineVPResidualType) {
  colmap::SetPRNGSeed(123);

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  ceres::CauchyLoss angle_loss(0.2);
  const double weight_vp = 5e3;

  // Generate random VP and line
  Eigen::Vector3d vp = Eigen::Vector3d::Random().normalized();
  double vp_params[3] = {vp.x(), vp.y(), vp.z()};

  Eigen::Vector3d p1 = Eigen::Vector3d::Random() * 5.0;
  Eigen::Vector3d dir = Eigen::Vector3d::Random().normalized();
  Eigen::Vector3d p2 = p1 + dir * 3.0;
  MinimalInfiniteLine3d line_minimal(Line3d(p1, p2));

  std::vector<std::unique_ptr<ceres::LossFunction>> losses;
  std::vector<AngleResidualInfo> angle_infos;
  std::vector<ceres::ResidualBlockId> angle_block_ids;

  // Add cosine VP residual
  losses.push_back(std::make_unique<ceres::ScaledLoss>(
      &angle_loss, weight_vp, ceres::DO_NOT_TAKE_OWNERSHIP));
  ceres::CostFunction *cost = LineToVPCosineCostFunctor::Create();
  auto block_id = problem.AddResidualBlock(cost, losses.back().get(),
                                           line_minimal.data.data(), vp_params);
  angle_block_ids.push_back(block_id);
  angle_infos.push_back({AngleResidualInfo::PARALLELISM,
                         line_minimal.data.data(), vp_params, weight_vp,
                         &angle_loss});

  // Compute with cosine mode
  AngleCostAwareToleranceCallback callback(1e-5, angle_infos,
                                           /*vp_use_cosine=*/true);
  const double manual_cost = callback.ComputeAngleCost();

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.residual_blocks = angle_block_ids;
  double ceres_cost = 0.0;
  problem.Evaluate(eval_options, &ceres_cost, nullptr, nullptr, nullptr);

  EXPECT_NEAR(manual_cost, ceres_cost, 1e-10)
      << "Manual (cosine) = " << manual_cost
      << ", Ceres evaluate = " << ceres_cost;
}

// Test that empty angle_infos produces zero cost
TEST(AngleCostCallback, EmptyInfosGivesZeroCost) {
  std::vector<AngleResidualInfo> empty;
  AngleCostAwareToleranceCallback callback(1e-5, empty);
  EXPECT_EQ(callback.ComputeAngleCost(), 0.0);
}

// Test convergence: with zero angle cost, convergence on total cost triggers.
TEST(AngleCostCallback, ConvergenceOnRelativeTolerance) {
  // Use ORTHOGONALITY with two orthogonal unit vectors -> residual = 0,
  // so angle_cost = 0 and non_angle_cost = summary.cost.
  std::vector<AngleResidualInfo> angle_infos;
  double dir1[3] = {1.0, 0.0, 0.0};
  double dir2[3] = {0.0, 1.0, 0.0};
  angle_infos.push_back(
      {AngleResidualInfo::ORTHOGONALITY, dir1, dir2, 1e5, nullptr});

  AngleCostAwareToleranceCallback callback(1e-5, angle_infos);

  // iteration 0: first successful step, stores baseline
  ceres::IterationSummary s0;
  s0.iteration = 0;
  s0.step_is_successful = true;
  s0.cost = 100.0;
  s0.cost_change = 0.0;
  EXPECT_EQ(callback(s0), ceres::SOLVER_CONTINUE);

  // iteration 1: large cost change, should continue
  ceres::IterationSummary s1;
  s1.iteration = 1;
  s1.step_is_successful = true;
  s1.cost = 90.0;
  s1.cost_change = -10.0;
  EXPECT_EQ(callback(s1), ceres::SOLVER_CONTINUE);

  // iteration 2: large cost change -- should continue
  ceres::IterationSummary s2;
  s2.iteration = 2;
  s2.step_is_successful = true;
  s2.cost = 50.0;
  s2.cost_change = -40.0;
  EXPECT_EQ(callback(s2), ceres::SOLVER_CONTINUE);

  // iteration 3: tiny cost change relative to cost -- should converge.
  // |non_angle_change| / non_angle_cost = 2e-9 < 1e-5 -> terminate
  ceres::IterationSummary s3;
  s3.iteration = 3;
  s3.step_is_successful = true;
  s3.cost = 50.0;
  s3.cost_change = -1e-7;
  EXPECT_EQ(callback(s3), ceres::SOLVER_TERMINATE_SUCCESSFULLY);
}

// Test that fast-dropping angle cost prevents premature termination.
// Core cost converges (change ~ 0), but total cost is still changing
// significantly due to angle cost -- the callback should strip angle and
// correctly detect that non_angle has converged.
TEST(AngleCostCallback, AngleStrippingAllowsEarlyTermination) {
  // Use two non-orthogonal directions so angle_cost > 0.
  // We'll modify dir2 between iterations to simulate angle cost dropping.
  double dir1[3] = {1.0, 0.0, 0.0};
  double dir2[3] = {0.6, 0.8, 0.0}; // dot = 0.6, angle_cost > 0

  std::vector<AngleResidualInfo> angle_infos;
  // No loss function, weight=1 for simplicity.
  // angle_cost = 0.5 * 1.0 * (dot(dir1,dir2))^2 = 0.5 * 0.36 = 0.18
  angle_infos.push_back(
      {AngleResidualInfo::ORTHOGONALITY, dir1, dir2, 1.0, nullptr});

  AngleCostAwareToleranceCallback callback(1e-5, angle_infos);

  ceres::IterationSummary s;
  s.step_is_successful = true;

  // Iteration 0: baseline
  s.iteration = 0;
  s.cost = 100.18; // core=100, angle=0.18
  s.cost_change = 0.0;
  EXPECT_EQ(callback(s), ceres::SOLVER_CONTINUE);

  // Iteration 1: large core drop
  s.iteration = 1;
  s.cost = 50.18;
  s.cost_change = -50.0;
  EXPECT_EQ(callback(s), ceres::SOLVER_CONTINUE);

  // Now simulate: core cost converges at 50 while angle cost drops.

  // Iteration 2: core stays 50, angle drops significantly
  dir2[0] = 0.4;
  dir2[1] = std::sqrt(1.0 - 0.16);
  double angle2 = callback.ComputeAngleCost(); // 0.5 * 0.16 = 0.08
  s.iteration = 2;
  s.cost = 50.0 + angle2;
  s.cost_change = s.cost - 50.18;
  // non_angle_change = cost_change - angle_change = (50.08-50.18) - (0.08-0.18)
  // = -0.1 + 0.1 = 0 non_angle = 50.0, |0|/50 < 1e-5 -> terminate! The key
  // insight: angle stripping allows early termination when core converges even
  // though total cost is still changing.
  EXPECT_EQ(callback(s), ceres::SOLVER_TERMINATE_SUCCESSFULLY);
}

// Test fallback to total-cost check when no angle residuals
TEST(AngleCostCallback, FallbackToTotalCostCheck) {
  std::vector<AngleResidualInfo> empty;
  AngleCostAwareToleranceCallback callback(1e-5, empty);

  ceres::IterationSummary s;
  s.step_is_successful = true;

  // Iteration 0: stores baseline
  s.iteration = 0;
  s.cost = 100.0;
  s.cost_change = 0.0;
  EXPECT_EQ(callback(s), ceres::SOLVER_CONTINUE);

  // Iteration 1: large change
  s.iteration = 1;
  s.cost = 50.0;
  s.cost_change = -50.0;
  EXPECT_EQ(callback(s), ceres::SOLVER_CONTINUE);

  // Iteration 2: tiny change -> terminate
  s.iteration = 2;
  s.cost = 50.0;
  s.cost_change = -1e-7;
  EXPECT_EQ(callback(s), ceres::SOLVER_TERMINATE_SUCCESSFULLY);
}

// Test that unsuccessful steps are skipped
TEST(AngleCostCallback, SkipsUnsuccessfulSteps) {
  std::vector<AngleResidualInfo> empty;
  AngleCostAwareToleranceCallback callback(1e-5, empty);

  ceres::IterationSummary s;
  s.iteration = 5;
  s.step_is_successful = false;
  s.cost = 100.0;
  s.cost_change = -1e-8;
  EXPECT_EQ(callback(s), ceres::SOLVER_CONTINUE);
}

} // namespace
} // namespace estimators
} // namespace limap
