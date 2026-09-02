// Tests for PointTriangulationCovariance and LineTriangulationCovariance.
//
// Both functions propagate a 2D observation covariance to 3D by first-order
// linearization: they build the analytic Jacobian J of the corresponding
// triangulator w.r.t. the 2D observations and return J * Sigma * J^T.
//
// The tests below verify that J really is the Jacobian of TriangulatePoint /
// TriangulateLine, by comparing against central finite differences of those
// exact functions. Since the functions return J * Sigma * J^T rather than J,
// individual columns are probed by passing Sigma = e_i * e_i^T, which yields
// the rank-1 matrix (J.col(i)) * (J.col(i))^T -- free of sign ambiguity.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Eigenvalues>

#include <colmap/scene/camera.h>
#include <colmap/scene/frame.h>
#include <colmap/scene/image.h>
#include <colmap/sensor/rig.h>

#include "limap/estimators/triangulation/functions.h"

namespace limap {
namespace estimators {
namespace triangulation {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Test scaffolding
////////////////////////////////////////////////////////////////////////////////

// A standalone colmap::Image wired to its own camera / rig / frame, so that
// Image::CamFromWorld() and Image::CameraPtr() are usable without a full
// Reconstruction. Holds raw pointers into its own members, hence non-copyable.
class TestView {
public:
  TestView(colmap::camera_t id, const colmap::Rigid3d &cam_from_world) {
    camera_.camera_id = id;
    camera_.model_id = colmap::PinholeCameraModel::model_id;
    camera_.params = {800.0, 800.0, 320.0, 240.0};
    camera_.width = 640;
    camera_.height = 480;

    image_.SetImageId(id);
    image_.SetCameraId(camera_.camera_id);

    rig_.AddRefSensor(camera_.SensorId());
    rig_.SetRigId(id);

    frame_.SetFrameId(id);
    frame_.SetRigId(rig_.RigId());
    frame_.AddDataId(image_.DataId());
    frame_.SetRigPtr(&rig_);
    frame_.SetCamFromWorld(camera_.camera_id, cam_from_world);

    image_.SetFrameId(frame_.FrameId());
    image_.SetCameraPtr(&camera_);
    image_.SetFramePtr(&frame_);
  }

  TestView(const TestView &) = delete;
  TestView &operator=(const TestView &) = delete;

  const colmap::Image &Image() const { return image_; }

  // Pinhole projection, matching the K^-1 unprojection in ImageUtils.
  V2D Project(const V3D &point3d) const {
    const V3D p_cam = image_.CamFromWorld() * point3d;
    const V3D x = camera_.CalibrationMatrix() * p_cam;
    return V2D(x.x() / x.z(), x.y() / x.z());
  }

private:
  colmap::Camera camera_;
  colmap::Rig rig_;
  colmap::Frame frame_;
  colmap::Image image_;
};

colmap::Rigid3d PoseFromCenter(const Eigen::Matrix3d &R_cw, const V3D &center) {
  colmap::Rigid3d cam_from_world;
  cam_from_world.rotation() = Eigen::Quaterniond(R_cw);
  cam_from_world.translation() = -(R_cw * center);
  return cam_from_world;
}

// Scene center that both views are aimed at; also the point used below.
constexpr double kSceneX = 0.3;
constexpr double kSceneZ = 5.0;

// Yaw about +Y that points a camera at (kSceneX, *, kSceneZ) from center
// (center_x, *, 0.1), so that the scene lands near the principal point
// regardless of how far along the baseline the camera sits.
double YawTowardScene(double center_x) {
  return std::atan2(center_x - kSceneX, kSceneZ - 0.1);
}

// Two well-conditioned views of the same scene: view 1 at the origin looking
// down +Z, view 2 offset along the baseline and yawed back toward the scene.
struct TwoViewScene {
  TwoViewScene()
      : view1(1, PoseFromCenter(Eigen::Matrix3d::Identity(), V3D(0, 0, 0))),
        view2(2, PoseFromCenter(
                     Eigen::AngleAxisd(YawTowardScene(1.0), V3D::UnitY())
                         .toRotationMatrix(),
                     V3D(1.0, 0.05, 0.1))) {}

  TestView view1;
  TestView view2;
};

// Central-difference Jacobian of f: R^n -> R^dim_out.
template <typename Func>
Eigen::MatrixXd FiniteDiffJacobian(const Func &f, const Eigen::VectorXd &x,
                                   int dim_out, double h = 1e-3) {
  Eigen::MatrixXd J(dim_out, x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    Eigen::VectorXd xp = x, xm = x;
    xp[i] += h;
    xm[i] -= h;
    J.col(i) = (f(xp) - f(xm)) / (2.0 * h);
  }
  return J;
}

// Compare two matrices with a tolerance relative to their overall scale.
void ExpectMatrixNear(const Eigen::MatrixXd &actual,
                      const Eigen::MatrixXd &expected, double rel_tol,
                      const std::string &what) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  const double scale =
      std::max(1e-12, std::max(actual.cwiseAbs().maxCoeff(),
                               expected.cwiseAbs().maxCoeff()));
  const double max_err = (actual - expected).cwiseAbs().maxCoeff();
  EXPECT_LE(max_err, rel_tol * scale)
      << what << ": max abs error " << max_err << " exceeds " << rel_tol
      << " * " << scale << "\nactual:\n"
      << actual << "\nexpected:\n"
      << expected;
}

////////////////////////////////////////////////////////////////////////////////
// Point triangulation covariance
////////////////////////////////////////////////////////////////////////////////

// Parameter vector for the point case: [p1.x, p1.y, p2.x, p2.y].
struct PointSetup {
  TwoViewScene scene;
  V3D point3d = V3D(kSceneX, 0.1, kSceneZ);
  Eigen::VectorXd x = Eigen::VectorXd(4);

  PointSetup() {
    const V2D p1 = scene.view1.Project(point3d);
    const V2D p2 = scene.view2.Project(point3d);
    x << p1.x(), p1.y(), p2.x(), p2.y();
  }

  Eigen::VectorXd Triangulate(const Eigen::VectorXd &xv) const {
    auto res = TriangulatePoint(V2D(xv[0], xv[1]), scene.view1.Image(),
                                V2D(xv[2], xv[3]), scene.view2.Image());
    EXPECT_TRUE(res.has_value());
    return *res;
  }

  Eigen::MatrixXd FiniteDiff() const {
    return FiniteDiffJacobian(
        [this](const Eigen::VectorXd &xv) { return Triangulate(xv); }, x, 3);
  }

  M3D Covariance(const Eigen::Matrix4d &sigma) const {
    return PointTriangulationCovariance(V2D(x[0], x[1]), scene.view1.Image(),
                                        V2D(x[2], x[3]), scene.view2.Image(),
                                        sigma);
  }
};

TEST(PointTriangulationCovariance, SetupTriangulatesExactly) {
  PointSetup s;
  ExpectMatrixNear(s.Triangulate(s.x), s.point3d, 1e-9, "triangulated point");
}

// Probe each Jacobian column independently: Sigma = e_i e_i^T must yield
// (J.col(i)) (J.col(i))^T.
TEST(PointTriangulationCovariance, JacobianColumnsMatchFiniteDifference) {
  PointSetup s;
  const Eigen::MatrixXd J_fd = s.FiniteDiff();

  for (int i = 0; i < 4; ++i) {
    Eigen::Matrix4d sigma = Eigen::Matrix4d::Zero();
    sigma(i, i) = 1.0;
    const Eigen::MatrixXd expected = J_fd.col(i) * J_fd.col(i).transpose();
    ExpectMatrixNear(s.Covariance(sigma), expected, 1e-6,
                     "point Jacobian column " + std::to_string(i));
  }
}

TEST(PointTriangulationCovariance, IsotropicNoiseMatchesFiniteDifference) {
  PointSetup s;
  const Eigen::MatrixXd J_fd = s.FiniteDiff();
  const double var = 2.0 * 2.0;

  const M3D cov = s.Covariance(var * Eigen::Matrix4d::Identity());
  ExpectMatrixNear(cov, var * J_fd * J_fd.transpose(), 1e-6,
                   "point covariance");

  // Sanity: a covariance must be symmetric and positive semi-definite.
  ExpectMatrixNear(cov, cov.transpose(), 1e-12, "point covariance symmetry");
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<M3D>(cov).eigenvalues().minCoeff(),
            -1e-12);
}

////////////////////////////////////////////////////////////////////////////////
// Line triangulation covariance
////////////////////////////////////////////////////////////////////////////////

// Parameter vector for the line case, in the order the analytic Jacobian
// assumes: [l1.start(2), l1.end(2), l2.start(2), l2.end(2)].
struct LineSetup {
  TwoViewScene scene;
  V3D start3d = V3D(0.2, -0.3, 4.5);
  V3D end3d = V3D(0.5, 0.4, 5.5);
  Eigen::VectorXd x = Eigen::VectorXd(8);

  LineSetup() {
    const V2D s1 = scene.view1.Project(start3d);
    const V2D e1 = scene.view1.Project(end3d);
    const V2D s2 = scene.view2.Project(start3d);
    const V2D e2 = scene.view2.Project(end3d);
    x << s1.x(), s1.y(), e1.x(), e1.y(), s2.x(), s2.y(), e2.x(), e2.y();
  }

  static Line2d Line1(const Eigen::VectorXd &xv) {
    return Line2d(V2D(xv[0], xv[1]), V2D(xv[2], xv[3]));
  }
  static Line2d Line2(const Eigen::VectorXd &xv) {
    return Line2d(V2D(xv[4], xv[5]), V2D(xv[6], xv[7]));
  }

  Eigen::VectorXd Triangulate(const Eigen::VectorXd &xv) const {
    auto res = TriangulateLine(Line1(xv), scene.view1.Image(), Line2(xv),
                               scene.view2.Image());
    EXPECT_TRUE(res.has_value());
    Eigen::VectorXd out(6);
    out << res->start, res->end;
    return out;
  }

  Eigen::MatrixXd FiniteDiff() const {
    return FiniteDiffJacobian(
        [this](const Eigen::VectorXd &xv) { return Triangulate(xv); }, x, 6);
  }

  M6D Covariance(const M8D &sigma) const {
    return LineTriangulationCovariance(Line1(x), scene.view1.Image(), Line2(x),
                                       scene.view2.Image(), sigma);
  }
};

TEST(LineTriangulationCovariance, SetupTriangulatesExactly) {
  LineSetup s;
  Eigen::VectorXd expected(6);
  expected << s.start3d, s.end3d;
  ExpectMatrixNear(s.Triangulate(s.x), expected, 1e-9, "triangulated line");
}

// Probe each of the 8 Jacobian columns independently.
//
// Columns 6 and 7 (the second view's line endpoint) are the regression guard:
// they were previously written into the wrong column of dA/dx -- c2_end sits in
// column 2 of A = [c1, -c2_start, -c2_end], not column 1 -- which left those
// two columns of J entirely wrong while columns 0-5 stayed correct.
TEST(LineTriangulationCovariance, JacobianColumnsMatchFiniteDifference) {
  LineSetup s;
  const Eigen::MatrixXd J_fd = s.FiniteDiff();

  for (int i = 0; i < 8; ++i) {
    M8D sigma = M8D::Zero();
    sigma(i, i) = 1.0;
    const Eigen::MatrixXd expected = J_fd.col(i) * J_fd.col(i).transpose();
    ExpectMatrixNear(s.Covariance(sigma), expected, 1e-6,
                     "line Jacobian column " + std::to_string(i));
  }
}

// Same guard, stated as a property: noise confined to the second view's line
// endpoint must still move the triangulated segment by the amount finite
// differences predict.
TEST(LineTriangulationCovariance, SecondViewEndpointNoisePropagates) {
  LineSetup s;
  const Eigen::MatrixXd J_fd = s.FiniteDiff();

  M8D sigma = M8D::Zero();
  sigma(6, 6) = 4.0;
  sigma(7, 7) = 4.0;

  const Eigen::MatrixXd expected =
      4.0 * (J_fd.col(6) * J_fd.col(6).transpose() +
             J_fd.col(7) * J_fd.col(7).transpose());
  const M6D cov = s.Covariance(sigma);
  ExpectMatrixNear(cov, expected, 1e-6, "line covariance from l2 endpoint");

  // The contribution must be non-trivial, otherwise the check above is vacuous.
  EXPECT_GT(cov.cwiseAbs().maxCoeff(), 1e-9);
}

TEST(LineTriangulationCovariance, IsotropicNoiseMatchesFiniteDifference) {
  LineSetup s;
  const Eigen::MatrixXd J_fd = s.FiniteDiff();
  const double var = 2.0 * 2.0;

  const M6D cov = s.Covariance(var * M8D::Identity());
  ExpectMatrixNear(cov, var * J_fd * J_fd.transpose(), 1e-6, "line covariance");

  ExpectMatrixNear(cov, cov.transpose(), 1e-12, "line covariance symmetry");
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<M6D>(cov).eigenvalues().minCoeff(),
            -1e-12);
}

////////////////////////////////////////////////////////////////////////////////
// Geometric sanity: a wider baseline must reduce the propagated uncertainty.
////////////////////////////////////////////////////////////////////////////////

double MaxStdDev(const Eigen::MatrixXd &cov) {
  return std::sqrt(
      std::max(0.0, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(cov)
                        .eigenvalues()
                        .maxCoeff()));
}

TEST(PointTriangulationCovariance, WiderBaselineReducesUncertainty) {
  const V3D point3d(kSceneX, 0.1, kSceneZ);
  const TestView view1(
      1, PoseFromCenter(Eigen::Matrix3d::Identity(), V3D::Zero()));

  double prev_sigma = std::numeric_limits<double>::infinity();
  for (const double baseline : {0.25, 0.5, 1.0, 2.0}) {
    const TestView view2(
        2,
        PoseFromCenter(Eigen::AngleAxisd(YawTowardScene(baseline), V3D::UnitY())
                           .toRotationMatrix(),
                       V3D(baseline, 0.05, 0.1)));
    const V2D p1 = view1.Project(point3d);
    const V2D p2 = view2.Project(point3d);

    const M3D cov = PointTriangulationCovariance(
        p1, view1.Image(), p2, view2.Image(), Eigen::Matrix4d::Identity());
    const double sigma = MaxStdDev(cov);
    EXPECT_LT(sigma, prev_sigma) << "baseline " << baseline;
    prev_sigma = sigma;
  }
}

} // namespace
} // namespace triangulation
} // namespace estimators
} // namespace limap
