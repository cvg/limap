// Tests for analytical group Jacobians (VP Sine/Cosine, Plane costs).

#include <gtest/gtest.h>

#include "limap/estimators/bundle_adjustment/analytical_group_cost_functions.h"
#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/estimators/bundle_adjustment/test_utils.h"
#include "limap/geometry/groups/projection.h"

namespace limap {
namespace estimators {
namespace {

using test::CheckGradient;
using test::CompareJacobians;
using test::CompareResiduals;
using test::CompareResidualsDynamic;
using test::kNumTrials;
using test::RandomCameraParams;
using test::RandomVisibleLine;
using test::RandomVisiblePoint;

////////////////////////////////////////////////////////////////////////////////
// VP Tests
////////////////////////////////////////////////////////////////////////////////

TEST(AnalyticalGroupJacobians, VPSine) {
  colmap::SetPRNGSeed(100);
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const std::string test_name = "LineToVPSine/trial=" + std::to_string(trial);

    Eigen::Quaterniond q_line = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d dir = q_line.toRotationMatrix().col(0);
    double line_params[6];
    Line3d line3d(Eigen::Vector3d::Zero(), dir);
    MinimalInfiniteLine3d minimal(line3d);
    for (int i = 0; i < 6; ++i)
      line_params[i] = minimal.data[i];

    Eigen::Vector3d vp = Eigen::Vector3d::Random().normalized();
    if (std::abs(dir.dot(vp)) > 0.99)
      vp = (dir.cross(Eigen::Vector3d::UnitX())).normalized();
    double vp_params[3] = {vp.x(), vp.y(), vp.z()};

    auto *analytical = new AnalyticalLineToVPSineCostFunction();
    auto *autodiff = LineToVPSineCostFunctor::Create();

    const double *params[] = {line_params, vp_params};
    EXPECT_TRUE(
        CompareResidualsDynamic(analytical, autodiff, params, 1, test_name));
    EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
    delete autodiff;

    ceres::SphereManifold<3> sphere_manifold;
    MinimalInfiniteLine3dManifold line_manifold;
    std::vector<const ceres::Manifold *> manifolds = {&line_manifold,
                                                      &sphere_manifold};
    EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
    delete analytical;
  }
}

TEST(AnalyticalGroupJacobians, VPCosine) {
  colmap::SetPRNGSeed(101);
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const std::string test_name =
        "LineToVPCosine/trial=" + std::to_string(trial);

    Eigen::Quaterniond q_line = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d dir = q_line.toRotationMatrix().col(0);
    double line_params[6];
    Line3d line3d(Eigen::Vector3d::Zero(), dir);
    MinimalInfiniteLine3d minimal(line3d);
    for (int i = 0; i < 6; ++i)
      line_params[i] = minimal.data[i];

    Eigen::Vector3d vp = Eigen::Vector3d::Random().normalized();
    double vp_params[3] = {vp.x(), vp.y(), vp.z()};

    auto *analytical = new AnalyticalLineToVPCosineCostFunction();
    auto *autodiff = LineToVPCosineCostFunctor::Create();

    const double *params[] = {line_params, vp_params};
    EXPECT_TRUE(
        CompareResidualsDynamic(analytical, autodiff, params, 1, test_name));
    EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
    delete autodiff;

    ceres::SphereManifold<3> sphere_manifold;
    MinimalInfiniteLine3dManifold line_manifold;
    std::vector<const ceres::Manifold *> manifolds = {&line_manifold,
                                                      &sphere_manifold};
    EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
    delete analytical;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Plane: PointToPlane2D ConstantPose
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
void RunPointToPlane2DConstantPoseTest(const std::string &model_name) {
  const std::string test_name =
      "PointToPlane2DConstantPose<" + model_name + ">";

  double camera_params[CameraModel::num_params];
  RandomCameraParams<CameraModel>(camera_params);

  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d tvec = Eigen::Vector3d::Random();
  colmap::Rigid3d cam_from_world(q, tvec);

  double kvec[4];
  ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

  double point3D[3];
  Eigen::Vector2d pixel;
  RandomVisiblePoint(q, tvec, kvec[0], kvec[1], kvec[2], kvec[3], point3D,
                     pixel);

  Eigen::Vector3d normal = Eigen::Vector3d::Random().normalized();
  Eigen::Map<const Eigen::Vector3d> p(point3D);
  double d = -(normal.dot(p)) + colmap::RandomGaussian(0.0, 0.5);
  double plane_params[4] = {normal.x(), normal.y(), normal.z(), d};

  auto *analytical =
      new AnalyticalPointToPlane2DConstantPoseCostFunction<CameraModel>(
          cam_from_world);

  using AutoDiffType =
      PointToSurface2DConstantPoseCostFunctor<CameraModel, PlaneProjection>;
  auto *autodiff = AutoDiffType::Create(cam_from_world);

  const double *params[] = {point3D, plane_params, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  ceres::ProductManifold<ceres::SphereManifold<3>, ceres::EuclideanManifold<1>>
      plane_manifold;
  std::vector<const ceres::Manifold *> manifolds = {nullptr, &plane_manifold,
                                                    nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

TEST(AnalyticalGroupJacobians, PointToPlane2DSimplePinhole) {
  colmap::SetPRNGSeed(102);
  for (int i = 0; i < kNumTrials; ++i)
    RunPointToPlane2DConstantPoseTest<colmap::SimplePinholeCameraModel>(
        "SimplePinhole");
}

TEST(AnalyticalGroupJacobians, PointToPlane2DPinhole) {
  colmap::SetPRNGSeed(103);
  for (int i = 0; i < kNumTrials; ++i)
    RunPointToPlane2DConstantPoseTest<colmap::PinholeCameraModel>("Pinhole");
}

////////////////////////////////////////////////////////////////////////////////
// Plane: LineToPlane2D ConstantPose
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
void RunLineToPlane2DConstantPoseTest(const std::string &model_name) {
  const std::string test_name = "LineToPlane2DConstantPose<" + model_name + ">";

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

  Eigen::Vector3d normal = Eigen::Vector3d::Random().normalized();
  double d = colmap::RandomUniformReal(-2.0, 2.0);
  double plane_params[4] = {normal.x(), normal.y(), normal.z(), d};

  auto *analytical =
      new AnalyticalLineToPlane2DConstantPoseCostFunction<CameraModel>(
          observed, cam_from_world);

  using AutoDiffType =
      LineToSurface2DConstantPoseCostFunctor<CameraModel, PlaneProjection>;
  auto *autodiff = AutoDiffType::Create(observed, cam_from_world);

  const double *params[] = {line_params, plane_params, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  MinimalInfiniteLine3dManifold line_manifold;
  ceres::ProductManifold<ceres::SphereManifold<3>, ceres::EuclideanManifold<1>>
      plane_manifold;
  std::vector<const ceres::Manifold *> manifolds = {&line_manifold,
                                                    &plane_manifold, nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

TEST(AnalyticalGroupJacobians, LineToPlane2DSimplePinhole) {
  colmap::SetPRNGSeed(104);
  for (int i = 0; i < kNumTrials; ++i)
    RunLineToPlane2DConstantPoseTest<colmap::SimplePinholeCameraModel>(
        "SimplePinhole");
}

TEST(AnalyticalGroupJacobians, LineToPlane2DPinhole) {
  colmap::SetPRNGSeed(105);
  for (int i = 0; i < kNumTrials; ++i)
    RunLineToPlane2DConstantPoseTest<colmap::PinholeCameraModel>("Pinhole");
}

} // namespace
} // namespace estimators
} // namespace limap
