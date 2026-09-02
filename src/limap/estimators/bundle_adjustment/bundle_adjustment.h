#pragma once

#include <ceres/ceres.h>
#include <cmath>
#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/estimators/bundle_adjustment_ceres.h>
#include <memory>
#include <set>

#include "limap/estimators/bundle_adjustment/custom_tolerance_callbacks.h"
#include "limap/geometry/minimal_inf_line3d.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/util/types.h"

namespace limap {
namespace estimators {

// Options for point-line bundle adjustment
// Inherits from colmap::BundleAdjustmentOptions and adds line-specific options
struct PointLineBundleAdjustmentOptions
    : public colmap::BundleAdjustmentOptions {
  // Re-export LossFunctionType from CeresBundleAdjustmentOptions so that
  // subclasses can use unqualified LossFunctionType in field declarations.
  using LossFunctionType =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType;

  PointLineBundleAdjustmentOptions() {
    // Use SPARSE_SCHUR: the Schur complement size depends only on
    // poses + group params (not residual count), so extra residuals from
    // wireframe edges or group associations don't affect factorization cost.
    // DENSE_SCHUR is still too slow for >50 images.
    ceres->auto_select_solver_type = false;
    ceres->solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  }

  // Structure refinement options
  bool refine_points = true;
  bool refine_lines = true;

  // Weights for different residual types
  double weight_point = 1.0;
  double weight_line = 1.0;

  // Length-based line weighting: each line observation is weighted by
  // line2d.Length() / line_weight_normalization_length.
  // Set to 0 to disable length-based weighting (uses weight_line uniformly).
  double line_weight_normalization_length = 60.0;

  // Separate loss function for line residuals (default: CAUCHY 1.0).
  // Lines are noisier than points, so a robust loss is always applied
  // regardless of the point loss function setting.
  LossFunctionType loss_function_type_line = LossFunctionType::CAUCHY;
  double loss_function_scale_line = 1.0;

  // Skip point reprojection residuals entirely. Used by
  // CreateLineBundleAdjuster for line-only BA where all point-related parameter
  // blocks are constant, making point residuals pure computational overhead.
  bool disable_point_residuals = false;

  // Explicitly set parameter block ordering: points + lines in group 0
  // (eliminated via Schur complement), poses + cameras + groups in group 1.
  // When false, Ceres auto-detects via greedy independent set heuristic.
  bool use_parameter_block_ordering = true;

  // Custom function tolerance for convergence, applied via
  // CustomToleranceCallback that only checks |cost_change|/cost on successful
  // steps. Ceres's built-in function_tolerance incorrectly fires on
  // unsuccessful steps, causing premature termination in problems with many
  // line/group residuals where gradient magnitudes are higher.
  // Set to 0.0 to disable (falls back to Ceres's built-in tolerances).
  double custom_function_tolerance = 1e-5;
};

// Configuration for point-line bundle adjustment
// Inherits from colmap::BundleAdjustmentConfig and adds line management
class PointLineBundleAdjustmentConfig : public colmap::BundleAdjustmentConfig {
public:
  PointLineBundleAdjustmentConfig() = default;

  size_t NumLines() const {
    return variable_line3D_ids_.size() + constant_line3D_ids_.size();
  }

  // Lines
  void AddVariableLine(line3D_t line3D_id);
  void AddConstantLine(line3D_t line3D_id);
  bool HasLine(line3D_t line3D_id) const;
  bool HasVariableLine(line3D_t line3D_id) const;
  bool HasConstantLine(line3D_t line3D_id) const;
  void RemoveVariableLine(line3D_t line3D_id);
  void RemoveConstantLine(line3D_t line3D_id);
  const FlatHashSet<line3D_t> &VariableLines() const {
    return variable_line3D_ids_;
  }
  const FlatHashSet<line3D_t> &ConstantLines() const {
    return constant_line3D_ids_;
  }

private:
  FlatHashSet<line3D_t> variable_line3D_ids_;
  FlatHashSet<line3D_t> constant_line3D_ids_;
};

// Base class for point-line bundle adjustment
// Optimizes HolisticReconstruction in-place following COLMAP's BundleAdjuster
// pattern. Can be extended to add more constraints (e.g., vanishing points,
// planes, coplanarity).
class BasePointLineBundleAdjuster : public colmap::CeresBundleAdjuster {
public:
  BasePointLineBundleAdjuster(PointLineBundleAdjustmentOptions options,
                              PointLineBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction);

  ~BasePointLineBundleAdjuster() override = default;

  // Run the optimization (override from base class)
  std::shared_ptr<colmap::BundleAdjustmentSummary> Solve() override;

  // Access the Ceres problem (override from CeresBundleAdjuster)
  std::shared_ptr<ceres::Problem> &Problem() override { return problem_; }

  // Structure-specific options (base class options_ has camera options)
  bool refine_points() const { return base_options_.refine_points; }
  bool refine_lines() const { return base_options_.refine_lines; }
  double weight_point() const { return base_options_.weight_point; }
  double weight_line() const { return base_options_.weight_line; }

  // Line-specific config access
  const FlatHashSet<line3D_t> &VariableLines() const {
    return variable_line3D_ids_;
  }
  const FlatHashSet<line3D_t> &ConstantLines() const {
    return constant_line3D_ids_;
  }
  bool HasVariableLine(line3D_t line3D_id) const {
    return variable_line3D_ids_.count(line3D_id) > 0;
  }
  bool HasConstantLine(line3D_t line3D_id) const {
    return constant_line3D_ids_.count(line3D_id) > 0;
  }

protected:
  // Methods for adding residuals - can be overridden or extended
  virtual void AddImageToProblem(colmap::image_t image_id,
                                 HolisticReconstruction &reconstruction);
  virtual void AddPointToProblem(colmap::point3D_t point3D_id,
                                 HolisticReconstruction &reconstruction);
  virtual void AddLineToProblem(line3D_t line3D_id,
                                HolisticReconstruction &reconstruction);

  // Methods for parameterization - can be overridden or extended
  virtual void ParameterizeCameras(HolisticReconstruction &reconstruction);
  virtual void ParameterizeImages(HolisticReconstruction &reconstruction);
  virtual void ParameterizePoints(HolisticReconstruction &reconstruction);
  virtual void ParameterizeLines();

  // Write back optimized parameters - can be overridden or extended
  virtual void WriteBackLines(HolisticReconstruction &reconstruction);

  // Create a ScaledLoss for a line observation based on its 2D length.
  // Returns loss_function_.get() if length-based weighting is disabled.
  ceres::LossFunction *CreateLineLoss(double line2d_length);

  // Set explicit parameter block ordering: points + lines in group 0
  // (eliminated via Schur complement), everything else in group 1.
  // No-op if use_parameter_block_ordering is false or solver is not Schur.
  void SetParameterBlockOrdering(ceres::Solver::Options &solver_options);

  // Point-line options (stored as typed struct; base options_ is sliced to
  // colmap::BundleAdjustmentOptions and lacks our fields)
  PointLineBundleAdjustmentOptions base_options_;

  // Line-specific config (extracted from PointLineBundleAdjustmentConfig)
  FlatHashSet<line3D_t> variable_line3D_ids_;
  FlatHashSet<line3D_t> constant_line3D_ids_;

  // Reference to reconstruction
  HolisticReconstruction &reconstruction_;

  // Ceres problem and loss functions
  std::shared_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::unique_ptr<ceres::LossFunction> line_loss_function_;

  // Line parameter storage (Line3d uses endpoints, need Plucker for
  // optimization)
  NodeHashMap<line3D_t, MinimalInfiniteLine3d> line_params_;

  // Tracking what's parameterized
  std::set<colmap::camera_t> parameterized_camera_ids_;
  std::set<colmap::image_t> parameterized_image_ids_;
  FlatHashMap<colmap::point3D_t, size_t> point3D_num_observations_;
  FlatHashMap<line3D_t, size_t> line3D_num_observations_;

  // Storage for per-observation scaled loss functions (memory management)
  std::vector<std::unique_ptr<ceres::LossFunction>> line_scaled_losses_;
};

// Factory function to create a default point-line bundle adjuster
std::unique_ptr<BasePointLineBundleAdjuster>
CreatePointLineBundleAdjuster(PointLineBundleAdjustmentOptions options,
                              PointLineBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction);

// Factory function to create a line-only bundle adjuster (no point residuals).
// Used for 2-step BA Step 2 where all point/pose/camera params are constant.
std::unique_ptr<BasePointLineBundleAdjuster>
CreateLineBundleAdjuster(PointLineBundleAdjustmentOptions options,
                         PointLineBundleAdjustmentConfig config,
                         HolisticReconstruction &reconstruction);

// Helper function to create loss function from type and scale
std::unique_ptr<ceres::LossFunction>
CreateLossFunction(colmap::CeresBundleAdjustmentOptions::LossFunctionType type,
                   double scale);

} // namespace estimators
} // namespace limap
