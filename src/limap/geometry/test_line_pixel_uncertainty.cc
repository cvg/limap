// Tests for ComputeLinePixelUncertainty and the full Jacobian chain used
// in covariance-based line track pixel uncertainty filtering.

#include <gtest/gtest.h>

#include <ceres/ceres.h>
#include <colmap/math/random.h>

#include <Eigen/Eigenvalues>

#include "limap/estimators/bundle_adjustment/analytical_line_cost_functions.h"
#include "limap/estimators/bundle_adjustment/test_utils.h"
#include "limap/geometry/line_jacobians.h"
#include "limap/geometry/line_pixel_uncertainty.h"
#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {
namespace {

using estimators::test::RandomCameraParams;
using estimators::test::RandomVisibleLine;

// Helper: generate a multi-view synthetic scene for a 3D line.
// Only cameras 0 is guaranteed to see the line; cameras 1..N are random
// and may have the line behind them. Use GenerateVisible() for tests that
// need all cameras to see the line.
struct SyntheticLineScene {
  double line_params[6];
  Line3d line3d;
  std::vector<Eigen::Quaterniond> rotations;
  std::vector<Eigen::Vector3d> translations;
  std::vector<Eigen::Vector4d> kvecs;
  std::vector<Line2d> observations;

  // Generate scene with random cameras (some may not see the line).
  void Generate(int num_views, unsigned seed, double noise_sigma = 0.0) {
    colmap::SetPRNGSeed(seed);

    Eigen::Quaterniond q0 = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t0 = Eigen::Vector3d::Random();
    double cam_params0[4];
    RandomCameraParams<colmap::PinholeCameraModel>(cam_params0);
    double kvec0[4];
    ParamsToKvec<double>(colmap::CameraModelId::kPinhole, cam_params0, kvec0);

    Line2d obs0;
    RandomVisibleLine(q0, t0, kvec0[0], kvec0[1], kvec0[2], kvec0[3],
                      line_params, obs0);

    MinimalInfiniteLine3d minimal;
    for (int i = 0; i < 6; ++i)
      minimal.data[i] = line_params[i];
    InfiniteLine3d inf_line = minimal.GetInfiniteLine();

    Eigen::Vector3d p_on_line = inf_line.d.cross(inf_line.m);
    line3d = Line3d(p_on_line - 2.0 * inf_line.d, p_on_line + 2.0 * inf_line.d);

    rotations.resize(num_views);
    translations.resize(num_views);
    kvecs.resize(num_views);
    observations.resize(num_views);

    rotations[0] = q0;
    translations[0] = t0;
    kvecs[0] = Eigen::Vector4d(kvec0[0], kvec0[1], kvec0[2], kvec0[3]);

    for (int v = 1; v < num_views; ++v) {
      rotations[v] = Eigen::Quaterniond::UnitRandom();
      translations[v] = Eigen::Vector3d::Random();
      double cp[4];
      RandomCameraParams<colmap::PinholeCameraModel>(cp);
      double kv[4];
      ParamsToKvec<double>(colmap::CameraModelId::kPinhole, cp, kv);
      kvecs[v] = Eigen::Vector4d(kv[0], kv[1], kv[2], kv[3]);
    }

    ProjectObservations(noise_sigma);
  }

  // Generate scene where all cameras are guaranteed to see the line
  // (both endpoints in front of camera with z > 0.5).
  void GenerateVisible(int num_views, unsigned seed, double noise_sigma = 0.0) {
    colmap::SetPRNGSeed(seed);

    Eigen::Quaterniond q0 = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d t0 = Eigen::Vector3d::Random();
    double cam_params0[4];
    RandomCameraParams<colmap::PinholeCameraModel>(cam_params0);
    double kvec0[4];
    ParamsToKvec<double>(colmap::CameraModelId::kPinhole, cam_params0, kvec0);

    Line2d obs0;
    RandomVisibleLine(q0, t0, kvec0[0], kvec0[1], kvec0[2], kvec0[3],
                      line_params, obs0);

    MinimalInfiniteLine3d minimal;
    for (int i = 0; i < 6; ++i)
      minimal.data[i] = line_params[i];
    InfiniteLine3d inf_line = minimal.GetInfiniteLine();

    Eigen::Vector3d p_on_line = inf_line.d.cross(inf_line.m);
    line3d = Line3d(p_on_line - 2.0 * inf_line.d, p_on_line + 2.0 * inf_line.d);

    rotations.clear();
    translations.clear();
    kvecs.clear();

    rotations.push_back(q0);
    translations.push_back(t0);
    kvecs.push_back(Eigen::Vector4d(kvec0[0], kvec0[1], kvec0[2], kvec0[3]));

    int attempts = 0;
    while (static_cast<int>(rotations.size()) < num_views &&
           attempts < num_views * 100) {
      ++attempts;
      Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
      Eigen::Vector3d t = Eigen::Vector3d::Random();
      double cp[4];
      RandomCameraParams<colmap::PinholeCameraModel>(cp);
      double kv[4];
      ParamsToKvec<double>(colmap::CameraModelId::kPinhole, cp, kv);

      Eigen::Matrix3d R = q.toRotationMatrix();
      Eigen::Vector3d s_cam = R * line3d.start + t;
      Eigen::Vector3d e_cam = R * line3d.end + t;

      if (s_cam.z() <= 0.5 || e_cam.z() <= 0.5)
        continue;

      rotations.push_back(q);
      translations.push_back(t);
      kvecs.push_back(Eigen::Vector4d(kv[0], kv[1], kv[2], kv[3]));
    }

    ProjectObservations(noise_sigma);
  }

  // Build the 6x6 ambient information matrix H using the same Jacobian chain
  // as ComputeLinePixelUncertainty. Used for verifying rank properties in
  // tests.
  Eigen::Matrix<double, 6, 6> BuildAmbientH() const {
    double d[3], m[3];
    double J_d_params[18], J_m_params[18];
    MinimalPluckerToPluckerWithJac(line_params, d, m, J_d_params, J_m_params);

    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(J_d_params);
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(J_m_params);

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();

    for (size_t k = 0; k < rotations.size(); ++k) {
      const Eigen::Matrix3d R = rotations[k].toRotationMatrix();
      const Eigen::Vector3d &t = translations[k];
      const double *kvec = kvecs[k].data();

      const Eigen::Map<const Eigen::Vector3d> d_vec(d);
      const Eigen::Map<const Eigen::Vector3d> m_vec(m);
      const Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(t);
      const Eigen::Vector3d m_cam = R * m_vec + t_skew * R * d_vec;

      const Eigen::Matrix3d J_mcam_d = t_skew * R;
      const Eigen::Matrix3d J_mcam_m = R;

      double l[3], J_l_mcam[9];
      LineCamToImgWithJac(kvec, m_cam.data(), l, J_l_mcam, nullptr);

      double residuals[2], J_r_l[6];
      LineResidualWithJac(l, observations[k], residuals, J_r_l);

      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr(J_r_l);
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jl(J_l_mcam);
      Eigen::Matrix<double, 2, 6> J_k =
          Jr * Jl * (J_mcam_d * Jd + J_mcam_m * Jm);

      H.noalias() += J_k.transpose() * J_k;
    }
    return H;
  }

private:
  void ProjectObservations(double noise_sigma) {
    observations.resize(rotations.size());
    for (size_t v = 0; v < rotations.size(); ++v) {
      Eigen::Matrix3d R = rotations[v].toRotationMatrix();
      Eigen::Vector3d t = translations[v];
      double fx = kvecs[v][0], fy = kvecs[v][1];
      double cx = kvecs[v][2], cy = kvecs[v][3];

      Eigen::Vector3d s_cam = R * line3d.start + t;
      Eigen::Vector3d e_cam = R * line3d.end + t;

      double su = fx * s_cam.x() / s_cam.z() + cx;
      double sv = fy * s_cam.y() / s_cam.z() + cy;
      double eu = fx * e_cam.x() / e_cam.z() + cx;
      double ev = fy * e_cam.y() / e_cam.z() + cy;

      if (noise_sigma > 0) {
        su += colmap::RandomGaussian(0.0, noise_sigma);
        sv += colmap::RandomGaussian(0.0, noise_sigma);
        eu += colmap::RandomGaussian(0.0, noise_sigma);
        ev += colmap::RandomGaussian(0.0, noise_sigma);
      }

      observations[v] = Line2d(V2D(su, sv), V2D(eu, ev));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test 1: Well-conditioned geometry → finite positive variance
//
// The covariance measures geometric sensitivity (Cramér-Rao bound), not
// residual magnitude. Even with perfect observations, the variance reflects
// how much a unit perturbation in line parameters would move the reprojected
// endpoints in pixel space. For a well-conditioned geometry (all cameras
// see the line from diverse viewpoints), this should be finite and positive.
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, WellConditionedGeometryFiniteVariance) {
  for (unsigned seed : {100u, 101u, 102u, 103u, 104u}) {
    SyntheticLineScene scene;
    scene.GenerateVisible(6, seed, /*noise_sigma=*/0.0);

    ASSERT_EQ(scene.rotations.size(), 6u)
        << "Seed " << seed << ": failed to generate 6 visible cameras";

    double sigma = ComputeLinePixelUncertainty(
        scene.line_params, scene.line3d, scene.rotations, scene.translations,
        scene.kvecs, scene.observations, nullptr);

    // Must be finite and positive (regression: manifold bug returned infinity)
    EXPECT_FALSE(std::isinf(sigma))
        << "Seed " << seed
        << ": well-conditioned geometry should not yield infinity";
    EXPECT_GT(sigma, 0.0) << "Seed " << seed;
    EXPECT_LT(sigma, 50.0)
        << "Seed " << seed
        << ": 6-view visible line should have moderate variance, got " << sigma;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Test 2: More observations → smaller variance
//
// Since H = Σ_k J_k^T J_k, adding more cameras increases H, which
// decreases Σ = H⁻¹ and thus the pixel variance.
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, MoreObservationsReduceVariance) {
  // Generate a scene with 10 visible cameras
  SyntheticLineScene scene;
  scene.GenerateVisible(10, /*seed=*/200, /*noise_sigma=*/0.0);
  ASSERT_EQ(scene.rotations.size(), 10u);

  // Compute variance with first 4 cameras
  double sigma_4 = ComputeLinePixelUncertainty(
      scene.line_params, scene.line3d,
      {scene.rotations.begin(), scene.rotations.begin() + 4},
      {scene.translations.begin(), scene.translations.begin() + 4},
      {scene.kvecs.begin(), scene.kvecs.begin() + 4},
      {scene.observations.begin(), scene.observations.begin() + 4}, nullptr);

  // Compute variance with all 10 cameras
  double sigma_10 = ComputeLinePixelUncertainty(
      scene.line_params, scene.line3d, scene.rotations, scene.translations,
      scene.kvecs, scene.observations, nullptr);

  ASSERT_FALSE(std::isinf(sigma_4)) << "4-view should be finite";
  ASSERT_FALSE(std::isinf(sigma_10)) << "10-view should be finite";
  EXPECT_GT(sigma_4, 0.0);
  EXPECT_GT(sigma_10, 0.0);

  // More observations should reduce variance
  EXPECT_LT(sigma_10, sigma_4)
      << "10 cameras should give smaller variance than 4 cameras"
      << " (σ_4=" << sigma_4 << ", σ_10=" << sigma_10 << ")";
}

////////////////////////////////////////////////////////////////////////////////
// Test 3: Degenerate geometry → infinite/large variance
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, DegenerateGeometryLargeVariance) {
  colmap::SetPRNGSeed(300);

  // Create a line along Z-axis
  Line3d line3d(Eigen::Vector3d(0, 0, 5), Eigen::Vector3d(0, 0, 10));
  MinimalInfiniteLine3d minimal(line3d);
  double line_params[6];
  for (int i = 0; i < 6; ++i)
    line_params[i] = minimal.data[i];

  // Two cameras looking along +Z, nearly coplanar with the line
  // (line lies along the viewing direction → poorly constrained)
  std::vector<Eigen::Quaterniond> rotations(2);
  std::vector<Eigen::Vector3d> translations(2);
  std::vector<Eigen::Vector4d> kvecs(2);

  rotations[0] = Eigen::Quaterniond::Identity();
  translations[0] = Eigen::Vector3d(0, 0, 0);
  kvecs[0] = Eigen::Vector4d(500, 500, 320, 240);

  rotations[1] = Eigen::Quaterniond::Identity();
  translations[1] = Eigen::Vector3d(0.01, 0, 0); // barely different
  kvecs[1] = Eigen::Vector4d(500, 500, 320, 240);

  std::vector<Line2d> observations(2);
  for (int v = 0; v < 2; ++v) {
    Eigen::Matrix3d R = rotations[v].toRotationMatrix();
    Eigen::Vector3d t = translations[v];
    Eigen::Vector3d s_cam = R * line3d.start + t;
    Eigen::Vector3d e_cam = R * line3d.end + t;
    double su = 500 * s_cam.x() / s_cam.z() + 320;
    double sv = 500 * s_cam.y() / s_cam.z() + 240;
    double eu = 500 * e_cam.x() / e_cam.z() + 320;
    double ev = 500 * e_cam.y() / e_cam.z() + 240;
    observations[v] = Line2d(V2D(su, sv), V2D(eu, ev));
  }

  double sigma =
      ComputeLinePixelUncertainty(line_params, line3d, rotations, translations,
                                  kvecs, observations, nullptr);

  EXPECT_TRUE(std::isinf(sigma) || sigma > 100.0)
      << "Degenerate geometry should yield very large sigma, got " << sigma;
}

////////////////////////////////////////////////////////////////////////////////
// Test 4: Loss function effect on information
//
// Cauchy loss downweights the outlier's contribution to the information
// matrix H, reducing total information → larger covariance → larger σ.
// This is correct: without the outlier's (geometrically valid) Jacobian
// contribution, the line has less observational support.
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, LossDownweightsOutlierInformation) {
  SyntheticLineScene scene;
  scene.GenerateVisible(5, /*seed=*/400, /*noise_sigma=*/0.0);
  ASSERT_EQ(scene.rotations.size(), 5u);

  // Add one large outlier observation
  scene.observations[0] = Line2d(
      V2D(scene.observations[0].start.x() + 100,
          scene.observations[0].start.y()),
      V2D(scene.observations[0].end.x() + 100, scene.observations[0].end.y()));

  // Without loss: all 5 observations contribute equally to H
  double sigma_no_loss = ComputeLinePixelUncertainty(
      scene.line_params, scene.line3d, scene.rotations, scene.translations,
      scene.kvecs, scene.observations, nullptr);

  // With Cauchy loss: outlier has huge residual → ρ'(s) ≈ 0 → its
  // contribution is removed from H → less information → larger σ
  ceres::CauchyLoss cauchy_loss(0.25);
  double sigma_cauchy = ComputeLinePixelUncertainty(
      scene.line_params, scene.line3d, scene.rotations, scene.translations,
      scene.kvecs, scene.observations, &cauchy_loss);

  ASSERT_TRUE(std::isfinite(sigma_no_loss)) << "No-loss should be finite";
  ASSERT_TRUE(std::isfinite(sigma_cauchy)) << "Cauchy should be finite";
  EXPECT_GT(sigma_no_loss, 0.0);
  EXPECT_GT(sigma_cauchy, 0.0);

  // Cauchy gives LARGER variance (less information after downweighting)
  EXPECT_GT(sigma_cauchy, sigma_no_loss)
      << "Cauchy should increase variance by removing outlier info"
      << " (σ_no_loss=" << sigma_no_loss << ", σ_cauchy=" << sigma_cauchy
      << ")";
}

////////////////////////////////////////////////////////////////////////////////
// Test 5: Jacobian chain matches AnalyticalLineReprojConstantPoseCostFunction
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, JacobianChainMatchesAnalytical) {
  colmap::SetPRNGSeed(500);

  double camera_params[4];
  RandomCameraParams<colmap::PinholeCameraModel>(camera_params);

  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d tvec = Eigen::Vector3d::Random();
  colmap::Rigid3d cam_from_world(q, tvec);

  double kvec[4];
  ParamsToKvec<double>(colmap::CameraModelId::kPinhole, camera_params, kvec);
  double line_params[6];
  Line2d observed;
  RandomVisibleLine(q, tvec, kvec[0], kvec[1], kvec[2], kvec[3], line_params,
                    observed);

  // Get Jacobian from AnalyticalLineReprojConstantPoseCostFunction
  auto *analytical =
      new estimators::AnalyticalLineReprojConstantPoseCostFunction<
          colmap::PinholeCameraModel>(observed, cam_from_world);

  const double *params[] = {line_params, camera_params};
  double residuals_ref[2];
  std::vector<double> jac_line(2 * 6), jac_cam(2 * 4);
  double *jacobians[] = {jac_line.data(), jac_cam.data()};
  analytical->Evaluate(params, residuals_ref, jacobians);

  // Compute the same chain manually (matching line_pixel_uncertainty.cc)
  double d[3], m[3];
  double J_d_params[18], J_m_params[18];
  MinimalPluckerToPluckerWithJac(line_params, d, m, J_d_params, J_m_params);

  Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(J_d_params);
  Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(J_m_params);

  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(tvec);
  Eigen::Matrix3d J_mcam_d = t_skew * R;
  Eigen::Matrix3d J_mcam_m = R;

  Eigen::Map<const Eigen::Vector3d> d_vec(d);
  Eigen::Map<const Eigen::Vector3d> m_vec(m);
  Eigen::Vector3d m_cam = R * m_vec + t_skew * R * d_vec;

  double l[3];
  double J_l_mcam[9];
  LineCamToImgWithJac(kvec, m_cam.data(), l, J_l_mcam, nullptr);

  double residuals_manual[2];
  double J_r_l[6];
  LineResidualWithJac(l, observed, residuals_manual, J_r_l);

  Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr(J_r_l);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jl(J_l_mcam);
  Eigen::Matrix<double, 2, 3> J_r_mcam = Jr * Jl;
  Eigen::Matrix<double, 2, 6> J_manual =
      J_r_mcam * (J_mcam_d * Jd + J_mcam_m * Jm);

  Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_ref(
      jac_line.data());

  double max_err = (J_manual - J_ref).cwiseAbs().maxCoeff();
  EXPECT_LT(max_err, 1e-10)
      << "Manual Jacobian chain should match analytical cost function\n"
      << "Manual:\n"
      << J_manual << "\nReference:\n"
      << J_ref;

  EXPECT_NEAR(residuals_manual[0], residuals_ref[0], 1e-12);
  EXPECT_NEAR(residuals_manual[1], residuals_ref[1], 1e-12);

  delete analytical;
}

////////////////////////////////////////////////////////////////////////////////
// Test 6: Not enough observations returns -1
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, NotEnoughObservationsReturnsNegative) {
  double line_params[6] = {1, 0, 0, 0, 0, 1};
  Line3d line3d(Eigen::Vector3d(0, 0, 5), Eigen::Vector3d(1, 0, 5));
  std::vector<Eigen::Quaterniond> rotations;
  std::vector<Eigen::Vector3d> translations;
  std::vector<Eigen::Vector4d> kvecs;
  std::vector<Line2d> lines2d;

  double sigma = ComputeLinePixelUncertainty(
      line_params, line3d, rotations, translations, kvecs, lines2d, nullptr);
  EXPECT_EQ(sigma, -1.0);

  rotations.push_back(Eigen::Quaterniond::Identity());
  translations.push_back(Eigen::Vector3d::Zero());
  kvecs.push_back(Eigen::Vector4d(500, 500, 320, 240));
  lines2d.push_back(Line2d(V2D(100, 100), V2D(200, 200)));

  sigma = ComputeLinePixelUncertainty(line_params, line3d, rotations,
                                      translations, kvecs, lines2d, nullptr);
  EXPECT_EQ(sigma, -1.0);
}

////////////////////////////////////////////////////////////////////////////////
// Test 7: Manifold tangent-space projection (regression test)
//
// MinimalPlucker lives on SO(3) × S¹ (4 DOF in 6D ambient space).
// The ambient 6×6 information matrix H may have near-zero eigenvalues
// along the constraint-violating directions (quaternion radial, weight
// radial). The correct approach is to project H to the 4D tangent space
// via B = MinusJacobian, giving H_tangent = B^T H B.
//
// Regression: the original code inverted the ambient H directly, which
// returned infinity for many real-world lines when the smallest ambient
// eigenvalue dropped below the 1e-10 threshold.
////////////////////////////////////////////////////////////////////////////////
TEST(LinePixelVariance, ManifoldTangentSpaceProjection) {
  for (unsigned seed : {700u, 701u, 702u, 703u, 704u}) {
    SyntheticLineScene scene;
    scene.GenerateVisible(8, seed, /*noise_sigma=*/0.5);
    ASSERT_GE(scene.rotations.size(), 6u)
        << "Seed " << seed << ": need enough visible cameras";

    // Build the 6×6 ambient information matrix
    Eigen::Matrix<double, 6, 6> H_ambient = scene.BuildAmbientH();

    // Compute eigenvalues of ambient H
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> ambient_solver(
        H_ambient);
    ASSERT_EQ(ambient_solver.info(), Eigen::Success);

    // Compute tangent-space projection: H_tangent = B^T * H * B (4×4)
    MinimalInfiniteLine3dManifold manifold;
    Eigen::Matrix<double, 6, 4, Eigen::RowMajor> B;
    manifold.MinusJacobian(scene.line_params, B.data());

    Eigen::Matrix4d H_tangent = B.transpose() * H_ambient * B;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> tangent_solver(H_tangent);
    ASSERT_EQ(tangent_solver.info(), Eigen::Success);

    double min_tangent_eig = tangent_solver.eigenvalues().minCoeff();
    double max_tangent_eig = tangent_solver.eigenvalues().maxCoeff();

    // The tangent-space H (4×4) must be well-conditioned for a line
    // observed from 8+ cameras with diverse viewpoints
    EXPECT_GT(min_tangent_eig, 1e-10)
        << "Seed " << seed << ": tangent H should be well-conditioned"
        << "\n  tangent eigenvalues: "
        << tangent_solver.eigenvalues().transpose()
        << "\n  ambient eigenvalues: "
        << ambient_solver.eigenvalues().transpose();

    EXPECT_LT(max_tangent_eig / min_tangent_eig, 1e12)
        << "Seed " << seed << ": tangent H condition number too large";

    // The function should return a finite, positive result
    double sigma = ComputeLinePixelUncertainty(
        scene.line_params, scene.line3d, scene.rotations, scene.translations,
        scene.kvecs, scene.observations, nullptr);

    EXPECT_FALSE(std::isinf(sigma))
        << "Seed " << seed
        << ": should be finite (tangent H is well-conditioned)"
        << "\n  min tangent eig: " << min_tangent_eig
        << "\n  ambient eigenvalues: "
        << ambient_solver.eigenvalues().transpose();
    EXPECT_GT(sigma, 0.0) << "Seed " << seed;
  }
}

} // namespace
} // namespace limap
