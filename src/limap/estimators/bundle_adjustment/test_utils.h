#pragma once

// Shared test utilities for analytical Jacobian tests.
// Provides random scene generation and gradient checking helpers.

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/math/random.h>
#include <colmap/sensor/models.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {
namespace estimators {
namespace test {

constexpr int kNumTrials = 5;
constexpr double kGradientTolerance = 1e-5;
constexpr double kAbsoluteTolerance = 1e-9;

// Generate random 3D line that is visible from the camera.
inline void RandomVisibleLine(const Eigen::Quaterniond &q_cfw,
                              const Eigen::Vector3d &t_cfw, double fx,
                              double fy, double cx, double cy,
                              double *line_params, Line2d &observed) {
  double u1 = cx + colmap::RandomUniformReal(-200.0, 200.0);
  double v1 = cy + colmap::RandomUniformReal(-200.0, 200.0);
  double u2 = cx + colmap::RandomUniformReal(-200.0, 200.0);
  double v2 = cy + colmap::RandomUniformReal(-200.0, 200.0);
  double z1 = colmap::RandomUniformReal(2.0, 10.0);
  double z2 = colmap::RandomUniformReal(2.0, 10.0);

  Eigen::Vector3d p1_cam((u1 - cx) / fx * z1, (v1 - cy) / fy * z1, z1);
  Eigen::Vector3d p2_cam((u2 - cx) / fx * z2, (v2 - cy) / fy * z2, z2);

  Eigen::Matrix3d R = q_cfw.toRotationMatrix();
  Eigen::Vector3d p1_world = R.transpose() * (p1_cam - t_cfw);
  Eigen::Vector3d p2_world = R.transpose() * (p2_cam - t_cfw);

  Line3d line3d(p1_world, p2_world);
  MinimalInfiniteLine3d minimal(line3d);
  for (int i = 0; i < 6; ++i)
    line_params[i] = minimal.data[i];

  observed = Line2d(V2D(u1 + colmap::RandomGaussian(0.0, 1.0),
                        v1 + colmap::RandomGaussian(0.0, 1.0)),
                    V2D(u2 + colmap::RandomGaussian(0.0, 1.0),
                        v2 + colmap::RandomGaussian(0.0, 1.0)));
}

// Generate random 3D point visible from camera.
inline void RandomVisiblePoint(const Eigen::Quaterniond &q_cfw,
                               const Eigen::Vector3d &t_cfw, double fx,
                               double fy, double cx, double cy, double *point3D,
                               Eigen::Vector2d &pixel) {
  double u = cx + colmap::RandomUniformReal(-200.0, 200.0);
  double v = cy + colmap::RandomUniformReal(-200.0, 200.0);
  double z = colmap::RandomUniformReal(2.0, 10.0);

  Eigen::Vector3d p_cam((u - cx) / fx * z, (v - cy) / fy * z, z);
  Eigen::Matrix3d R = q_cfw.toRotationMatrix();
  Eigen::Vector3d p_world = R.transpose() * (p_cam - t_cfw);

  point3D[0] = p_world.x();
  point3D[1] = p_world.y();
  point3D[2] = p_world.z();
  pixel = Eigen::Vector2d(u + colmap::RandomGaussian(0.0, 1.0),
                          v + colmap::RandomGaussian(0.0, 1.0));
}

// Generate random camera parameters for a given camera model.
template <typename CameraModel> void RandomCameraParams(double *camera_params) {
  if constexpr (std::is_same_v<CameraModel, colmap::SimplePinholeCameraModel>) {
    camera_params[0] = colmap::RandomUniformReal(400.0, 600.0);
    camera_params[1] = colmap::RandomUniformReal(300.0, 340.0);
    camera_params[2] = colmap::RandomUniformReal(220.0, 260.0);
  } else {
    camera_params[0] = colmap::RandomUniformReal(400.0, 600.0);
    camera_params[1] = colmap::RandomUniformReal(380.0, 580.0);
    camera_params[2] = colmap::RandomUniformReal(300.0, 340.0);
    camera_params[3] = colmap::RandomUniformReal(220.0, 260.0);
  }
}

// Compare residuals between analytical and AutoDiff cost functions (2D).
inline bool CompareResiduals(ceres::CostFunction *analytical,
                             ceres::CostFunction *autodiff,
                             double const *const *params,
                             const std::string &test_name) {
  std::vector<double> r_anal(2), r_auto(2);
  analytical->Evaluate(params, r_anal.data(), nullptr);
  autodiff->Evaluate(params, r_auto.data(), nullptr);

  double diff =
      std::abs(r_anal[0] - r_auto[0]) + std::abs(r_anal[1] - r_auto[1]);
  if (diff >= 1e-10) {
    std::cerr << test_name << ": residual mismatch = " << diff << std::endl;
    return false;
  }
  return true;
}

// Compare residuals with dynamic size.
inline bool CompareResidualsDynamic(ceres::CostFunction *analytical,
                                    ceres::CostFunction *autodiff,
                                    double const *const *params,
                                    int num_residuals,
                                    const std::string &test_name) {
  std::vector<double> r_anal(num_residuals), r_auto(num_residuals);
  analytical->Evaluate(params, r_anal.data(), nullptr);
  autodiff->Evaluate(params, r_auto.data(), nullptr);

  double diff = 0;
  for (int i = 0; i < num_residuals; ++i)
    diff += std::abs(r_anal[i] - r_auto[i]);
  if (diff >= 1e-10) {
    std::cerr << test_name << ": residual mismatch = " << diff << std::endl;
    return false;
  }
  return true;
}

// Compare Jacobians between analytical and AutoDiff cost functions directly.
// More reliable than GradientChecker/Ridders because AutoDiff computes exact
// derivatives through the same computational graph (no finite differencing).
inline bool CompareJacobians(ceres::CostFunction *analytical,
                             ceres::CostFunction *autodiff,
                             double const *const *params,
                             const std::string &test_name) {
  const auto &block_sizes = analytical->parameter_block_sizes();
  const int num_blocks = block_sizes.size();
  const int num_residuals = analytical->num_residuals();

  // Allocate Jacobian storage
  std::vector<std::vector<double>> anal_jac_data(num_blocks);
  std::vector<std::vector<double>> auto_jac_data(num_blocks);
  std::vector<double *> anal_jac_ptrs(num_blocks);
  std::vector<double *> auto_jac_ptrs(num_blocks);
  for (int i = 0; i < num_blocks; ++i) {
    anal_jac_data[i].resize(num_residuals * block_sizes[i], 0.0);
    auto_jac_data[i].resize(num_residuals * block_sizes[i], 0.0);
    anal_jac_ptrs[i] = anal_jac_data[i].data();
    auto_jac_ptrs[i] = auto_jac_data[i].data();
  }

  std::vector<double> r_anal(num_residuals), r_auto(num_residuals);
  analytical->Evaluate(params, r_anal.data(), anal_jac_ptrs.data());
  autodiff->Evaluate(params, r_auto.data(), auto_jac_ptrs.data());

  bool ok = true;
  for (int k = 0; k < num_blocks; ++k) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        J_anal(anal_jac_data[k].data(), num_residuals, block_sizes[k]);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        J_auto(auto_jac_data[k].data(), num_residuals, block_sizes[k]);

    const auto diff = (J_anal - J_auto).cwiseAbs();
    const double max_abs = diff.maxCoeff();
    if (max_abs < kAbsoluteTolerance)
      continue;

    // Check relative error for non-trivial entries
    for (int i = 0; i < num_residuals; ++i) {
      for (int j = 0; j < block_sizes[k]; ++j) {
        double a = J_anal(i, j), n = J_auto(i, j);
        double abs_err = std::abs(a - n);
        double scale = std::max(std::abs(a), std::abs(n));
        double rel_err = (scale > 0) ? abs_err / scale : 0;
        if (rel_err > kGradientTolerance && abs_err > kAbsoluteTolerance) {
          if (ok) {
            std::cerr << "FAIL: " << test_name
                      << ": Jacobian mismatch (analytical vs AutoDiff)"
                      << std::endl;
          }
          ok = false;
          std::cerr << "  block " << k << " (" << i << "," << j
                    << "): analytical=" << a << " autodiff=" << n
                    << " rel_err=" << rel_err << " abs_err=" << abs_err
                    << std::endl;
        }
      }
    }
    if (!ok) {
      std::cerr << "  block " << k << " analytical:\n" << J_anal << std::endl;
      std::cerr << "  block " << k << " autodiff:\n" << J_auto << std::endl;
    }
  }
  return ok;
}

// Run GradientChecker on analytical cost function.
// Uses relative tolerance (1e-5) from Ceres GradientChecker, with a fallback:
// if max absolute error is below kAbsoluteTolerance, the check passes even
// when relative error is high (which happens when Jacobian entries are near
// machine epsilon — the relative metric becomes meaningless).
//
// Uses ridders_relative_initial_step_size = 1e-4 (matching Ceres's own
// manifold_test_utils.h) to avoid large ambient-space perturbations that
// push manifold-constrained parameters (quaternions, sphere) far off the
// manifold surface, causing poor Ridders convergence.
inline bool CheckGradient(ceres::CostFunction *analytical,
                          const std::vector<const ceres::Manifold *> &manifolds,
                          double const *const *params,
                          const std::string &test_name) {
  ceres::NumericDiffOptions numeric_diff_options;
  numeric_diff_options.ridders_relative_initial_step_size = 1e-4;
  ceres::GradientChecker checker(analytical, &manifolds, numeric_diff_options);
  ceres::GradientChecker::ProbeResults results;
  bool ok = checker.Probe(params, kGradientTolerance, &results);
  if (!ok) {
    // Check if failure is due to near-zero entries (machine epsilon noise)
    double max_abs_error = 0;
    for (size_t i = 0; i < results.jacobians.size(); ++i) {
      if (results.jacobians[i].rows() == 0)
        continue;
      double block_err = (results.jacobians[i] - results.numeric_jacobians[i])
                             .cwiseAbs()
                             .maxCoeff();
      max_abs_error = std::max(max_abs_error, block_err);
    }
    if (max_abs_error < kAbsoluteTolerance) {
      // Both analytical and numeric are essentially zero — pass
      return true;
    }

    std::cerr << "FAIL: " << test_name << ": GradientChecker failed"
              << std::endl;
    std::cerr << "  max relative error: " << results.maximum_relative_error
              << std::endl;
    std::cerr << "  max absolute error: " << max_abs_error << std::endl;
    for (size_t i = 0; i < results.jacobians.size(); ++i) {
      if (results.jacobians[i].rows() == 0)
        continue;
      double block_err = (results.jacobians[i] - results.numeric_jacobians[i])
                             .cwiseAbs()
                             .maxCoeff();
      std::cerr << "  block " << i << " max abs error: " << block_err
                << std::endl;
      if (block_err > kAbsoluteTolerance) {
        std::cerr << "  analytical:\n" << results.jacobians[i] << std::endl;
        std::cerr << "  numeric:\n"
                  << results.numeric_jacobians[i] << std::endl;
      }
    }
  }
  return ok;
}

} // namespace test
} // namespace estimators
} // namespace limap
