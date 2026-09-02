// Tests for analytical wireframe Jacobians (LineToPoint, PointToLine).

#include <gtest/gtest.h>

#include "limap/estimators/bundle_adjustment/analytical_wireframe_cost_functions.h"
#include "limap/estimators/bundle_adjustment/test_utils.h"
#include "limap/estimators/bundle_adjustment/wireframe_cost_functions.h"

namespace limap {
namespace estimators {
namespace {

using test::CheckGradient;
using test::CompareJacobians;
using test::CompareResiduals;
using test::kNumTrials;
using test::RandomCameraParams;
using test::RandomVisibleLine;
using test::RandomVisiblePoint;

////////////////////////////////////////////////////////////////////////////////
// LineToPoint ConstantPose
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
void RunLineToPointConstantPoseTest(const std::string &model_name) {
  const std::string test_name = "LineToPointConstantPose<" + model_name + ">";

  double camera_params[CameraModel::num_params];
  RandomCameraParams<CameraModel>(camera_params);

  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d tvec = Eigen::Vector3d::Random();
  colmap::Rigid3d cam_from_world(q, tvec);

  double kvec[4];
  ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

  double line_params[6];
  Line2d observed_line;
  RandomVisibleLine(q, tvec, kvec[0], kvec[1], kvec[2], kvec[3], line_params,
                    observed_line);

  Eigen::Vector2d observed_point =
      0.5 * (observed_line.start + observed_line.end) +
      Eigen::Vector2d(colmap::RandomGaussian(0.0, 2.0),
                      colmap::RandomGaussian(0.0, 2.0));

  auto *analytical =
      new AnalyticalLineToPointConstantPoseCostFunction<CameraModel>(
          observed_point, cam_from_world);
  auto *autodiff = LineToPointConstantPoseCostFunctor<CameraModel>::Create(
      observed_point, cam_from_world);

  const double *params[] = {line_params, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  MinimalInfiniteLine3dManifold line_manifold;
  std::vector<const ceres::Manifold *> manifolds = {&line_manifold, nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

TEST(AnalyticalWireframeJacobians, LineToPointSimplePinhole) {
  colmap::SetPRNGSeed(200);
  for (int i = 0; i < kNumTrials; ++i)
    RunLineToPointConstantPoseTest<colmap::SimplePinholeCameraModel>(
        "SimplePinhole");
}

TEST(AnalyticalWireframeJacobians, LineToPointPinhole) {
  colmap::SetPRNGSeed(201);
  for (int i = 0; i < kNumTrials; ++i)
    RunLineToPointConstantPoseTest<colmap::PinholeCameraModel>("Pinhole");
}

////////////////////////////////////////////////////////////////////////////////
// PointToLine ConstantPose
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
void RunPointToLineConstantPoseTest(const std::string &model_name) {
  const std::string test_name = "PointToLineConstantPose<" + model_name + ">";

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

  double angle = colmap::RandomUniformReal(0.0, M_PI);
  double len = colmap::RandomUniformReal(30.0, 100.0);
  Eigen::Vector2d dir_2d(std::cos(angle), std::sin(angle));
  Eigen::Vector2d p1 = pixel - 0.5 * len * dir_2d +
                       Eigen::Vector2d(colmap::RandomGaussian(0.0, 2.0),
                                       colmap::RandomGaussian(0.0, 2.0));
  Eigen::Vector2d p2 = pixel + 0.5 * len * dir_2d +
                       Eigen::Vector2d(colmap::RandomGaussian(0.0, 2.0),
                                       colmap::RandomGaussian(0.0, 2.0));
  Line2d observed_line(p1, p2);

  auto *analytical =
      new AnalyticalPointToLineConstantPoseCostFunction<CameraModel>(
          observed_line, cam_from_world);
  auto *autodiff = PointToLineConstantPoseCostFunctor<CameraModel>::Create(
      observed_line, cam_from_world);

  const double *params[] = {point3D, camera_params};

  EXPECT_TRUE(CompareResiduals(analytical, autodiff, params, test_name));
  EXPECT_TRUE(CompareJacobians(analytical, autodiff, params, test_name));
  delete autodiff;

  std::vector<const ceres::Manifold *> manifolds = {nullptr, nullptr};
  EXPECT_TRUE(CheckGradient(analytical, manifolds, params, test_name));
  delete analytical;
}

TEST(AnalyticalWireframeJacobians, PointToLineSimplePinhole) {
  colmap::SetPRNGSeed(202);
  for (int i = 0; i < kNumTrials; ++i)
    RunPointToLineConstantPoseTest<colmap::SimplePinholeCameraModel>(
        "SimplePinhole");
}

TEST(AnalyticalWireframeJacobians, PointToLinePinhole) {
  colmap::SetPRNGSeed(203);
  for (int i = 0; i < kNumTrials; ++i)
    RunPointToLineConstantPoseTest<colmap::PinholeCameraModel>("Pinhole");
}

} // namespace
} // namespace estimators
} // namespace limap
