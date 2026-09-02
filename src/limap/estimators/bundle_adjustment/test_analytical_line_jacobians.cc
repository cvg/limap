// Tests for analytical line reprojection Jacobians (ConstantPose +
// VariablePose). Compares analytical Jacobians against autodiff and numerical
// gradients.

#include <gtest/gtest.h>

#include "limap/estimators/bundle_adjustment/analytical_line_cost_functions.h"
#include "limap/estimators/bundle_adjustment/cost_functions.h"
#include "limap/estimators/bundle_adjustment/test_utils.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/line_jacobians.h"

namespace limap {
namespace estimators {
namespace {

using test::CheckGradient;
using test::CompareJacobians;
using test::CompareResiduals;
using test::kNumTrials;
using test::RandomCameraParams;
using test::RandomVisibleLine;

////////////////////////////////////////////////////////////////////////////////
// Full chain tests
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
void RunConstantPoseTest(const std::string &model_name) {
  const std::string test_name = "ConstantPose<" + model_name + ">";

  double camera_params[CameraModel::num_params];
  RandomCameraParams<CameraModel>(camera_params);

  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d tvec = Eigen::Vector3d::Random();
  colmap::Rigid3d cam_from_world(q, tvec);

  double kvec[4];
  ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);
  double line_params[6];
  Line2d observed;
  RandomVisibleLine(q, tvec, kvec[0], kvec[1], kvec[2], kvec[3], line_params,
                    observed);

  auto *analytical =
      new AnalyticalLineReprojConstantPoseCostFunction<CameraModel>(
          observed, cam_from_world);
  auto *autodiff = LineReprojectionConstantPoseCostFunctor<CameraModel>::Create(
      observed, cam_from_world);

  const double *params[] = {line_params, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  MinimalInfiniteLine3dManifold line_manifold;
  std::vector<const ceres::Manifold *> manifolds = {&line_manifold, nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

template <typename CameraModel>
void RunVariablePoseTest(const std::string &model_name) {
  const std::string test_name = "VariablePose<" + model_name + ">";

  double camera_params[CameraModel::num_params];
  RandomCameraParams<CameraModel>(camera_params);

  Eigen::Quaterniond q_eigen = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d t_eigen = Eigen::Vector3d::Random();

  // Unified 7-element pose block: [qx, qy, qz, qw, tx, ty, tz]
  double cam_from_world[7] = {q_eigen.x(), q_eigen.y(), q_eigen.z(),
                              q_eigen.w(), t_eigen.x(), t_eigen.y(),
                              t_eigen.z()};

  double kvec[4];
  ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);
  double line_params[6];
  Line2d observed;
  RandomVisibleLine(q_eigen, t_eigen, kvec[0], kvec[1], kvec[2], kvec[3],
                    line_params, observed);

  auto *analytical =
      new AnalyticalLineReprojCostFunction<CameraModel>(observed);
  auto *autodiff = LineReprojectionCostFunctor<CameraModel>::Create(observed);

  const double *params[] = {line_params, cam_from_world, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  ceres::ProductManifold<ceres::EigenQuaternionManifold,
                         ceres::EuclideanManifold<3>>
      pose_manifold;
  MinimalInfiniteLine3dManifold line_manifold;
  std::vector<const ceres::Manifold *> manifolds = {&line_manifold,
                                                    &pose_manifold, nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

TEST(AnalyticalLineJacobians, ConstantPoseSimplePinhole) {
  colmap::SetPRNGSeed(42);
  for (int i = 0; i < kNumTrials; ++i)
    RunConstantPoseTest<colmap::SimplePinholeCameraModel>("SimplePinhole");
}

TEST(AnalyticalLineJacobians, ConstantPosePinhole) {
  colmap::SetPRNGSeed(44);
  for (int i = 0; i < kNumTrials; ++i)
    RunConstantPoseTest<colmap::PinholeCameraModel>("Pinhole");
}

TEST(AnalyticalLineJacobians, VariablePoseSimplePinhole) {
  colmap::SetPRNGSeed(46);
  for (int i = 0; i < kNumTrials; ++i)
    RunVariablePoseTest<colmap::SimplePinholeCameraModel>("SimplePinhole");
}

TEST(AnalyticalLineJacobians, VariablePosePinhole) {
  colmap::SetPRNGSeed(48);
  for (int i = 0; i < kNumTrials; ++i)
    RunVariablePoseTest<colmap::PinholeCameraModel>("Pinhole");
}

////////////////////////////////////////////////////////////////////////////////
// PointOnLineWithJac tests
////////////////////////////////////////////////////////////////////////////////

// Wrap PointOnLineWithJac as a CostFunction for gradient checking.
// Params: [d(3), m(3)], Residuals: 3 (projected point)
class PointOnLineCostFunction : public ceres::SizedCostFunction<3, 3, 3> {
public:
  explicit PointOnLineCostFunction(const Eigen::Vector3d &point)
      : point_(point) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    PointOnLineWithJac(parameters[0], parameters[1], point_.data(), residuals,
                       jacobians ? jacobians[0] : nullptr,
                       jacobians ? jacobians[1] : nullptr);
    return true;
  }

private:
  Eigen::Vector3d point_;
};

TEST(AnalyticalLineJacobians, PointOnLineWithJac) {
  colmap::SetPRNGSeed(50);
  for (int i = 0; i < kNumTrials; ++i) {
    const std::string test_name = "PointOnLine";

    // Generate random Plücker line (unit d)
    Eigen::Vector3d d = Eigen::Vector3d::Random().normalized();
    // m must be perpendicular to d for a valid Plücker line
    Eigen::Vector3d arb = Eigen::Vector3d::Random();
    Eigen::Vector3d m = d.cross(arb);
    if (m.norm() < 1e-6) {
      m = d.cross(Eigen::Vector3d::UnitX());
    }

    Eigen::Vector3d point = Eigen::Vector3d::Random() * 5.0;

    auto *cost = new PointOnLineCostFunction(point);
    const double *params[] = {d.data(), m.data()};

    // Check Jacobians via Ridders numerical differentiation
    std::vector<const ceres::Manifold *> manifolds = {nullptr, nullptr};
    EXPECT_TRUE(CheckGradient(cost, manifolds, params, test_name));

    // Verify projected point lies on the line: (p_proj - d×m) should be
    // parallel to d
    double projected[3];
    PointOnLineWithJac(d.data(), m.data(), point.data(), projected, nullptr,
                       nullptr);
    Eigen::Map<Eigen::Vector3d> proj(projected);
    Eigen::Vector3d dxm = d.cross(m);
    Eigen::Vector3d diff = proj - dxm;
    // diff should be parallel to d: diff × d ≈ 0
    EXPECT_LT(diff.cross(d).norm(), 1e-10) << "Projected point not on line";

    delete cost;
  }
}

////////////////////////////////////////////////////////////////////////////////
// EndpointFromMinimalPluckerWithJac tests
////////////////////////////////////////////////////////////////////////////////

// Wrap EndpointFromMinimalPluckerWithJac as a CostFunction for gradient
// checking.
// Params: [line_params(6)], Residuals: 3 (projected endpoint)
class EndpointFromMinimalPluckerCostFunction
    : public ceres::SizedCostFunction<3, 6> {
public:
  explicit EndpointFromMinimalPluckerCostFunction(
      const Eigen::Vector3d &endpoint)
      : endpoint_(endpoint) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    EndpointFromMinimalPluckerWithJac(parameters[0], endpoint_.data(),
                                      residuals,
                                      jacobians ? jacobians[0] : nullptr);
    return true;
  }

private:
  Eigen::Vector3d endpoint_;
};

TEST(AnalyticalLineJacobians, EndpointFromMinimalPluckerWithJac) {
  colmap::SetPRNGSeed(52);
  for (int i = 0; i < kNumTrials; ++i) {
    const std::string test_name = "EndpointFromMinimalPlucker";

    // Generate random 3D line via two random points
    Eigen::Vector3d p1 = Eigen::Vector3d::Random() * 5.0;
    Eigen::Vector3d p2 = Eigen::Vector3d::Random() * 5.0;
    Line3d line3d(p1, p2);
    MinimalInfiniteLine3d minimal(line3d);
    double line_params[6];
    for (int j = 0; j < 6; ++j)
      line_params[j] = minimal.data[j];

    // Use one of the line's endpoints as the test point
    Eigen::Vector3d endpoint = p1;

    auto *cost = new EndpointFromMinimalPluckerCostFunction(endpoint);
    const double *params[] = {line_params};

    // Check gradient with manifold
    MinimalInfiniteLine3dManifold line_manifold;
    std::vector<const ceres::Manifold *> manifolds = {&line_manifold};
    EXPECT_TRUE(CheckGradient(cost, manifolds, params, test_name));

    // Verify consistency with PointOnLineWithJac
    double projected_chain[3];
    EndpointFromMinimalPluckerWithJac(line_params, endpoint.data(),
                                      projected_chain, nullptr);

    double d[3], m[3];
    MinimalPluckerToPluckerWithJac(line_params, d, m, nullptr, nullptr);
    double projected_direct[3];
    PointOnLineWithJac(d, m, endpoint.data(), projected_direct, nullptr,
                       nullptr);

    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(projected_chain[j], projected_direct[j], 1e-12)
          << "Mismatch at component " << j;
    }

    delete cost;
  }
}

} // namespace
} // namespace estimators
} // namespace limap
