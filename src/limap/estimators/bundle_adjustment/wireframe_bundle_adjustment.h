#pragma once

#include <ceres/ceres.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/util/types.h"

namespace limap {
namespace estimators {

// Options for wireframe bundle adjustment
// Extends PointLineBundleAdjustmentOptions with wireframe-specific parameters
struct WireframeBundleAdjustmentOptions
    : public PointLineBundleAdjustmentOptions {
  // Whether to add wireframe constraints
  bool refine_wireframe = true;

  // Weight for wireframe residuals
  double weight_wireframe = 0.1;

  // Loss function for wireframe constraints (TRIVIAL, HUBER, CAUCHY, SOFT_L1)
  LossFunctionType loss_function_type_wireframe = LossFunctionType::CAUCHY;
  double loss_function_scale_wireframe = 3.0;

  // Whether to force reconstruct wireframe3d from 2D associations before BA
  // If true: clears existing wireframe and reconstructs using wireframe_voting
  // If false: only constructs if wireframe is empty
  bool force_reconstruct_wireframe3d = false;

  // Options for constructing wireframe from 2D
  WireframeVotingOptions wireframe_voting;
};

// Wireframe bundle adjuster extending BasePointLineBundleAdjuster
// Adds cross-feature reprojection constraints between points and lines
class WireframeBundleAdjuster : public BasePointLineBundleAdjuster {
public:
  WireframeBundleAdjuster(WireframeBundleAdjustmentOptions options,
                          PointLineBundleAdjustmentConfig config,
                          HolisticReconstruction &reconstruction);

  ~WireframeBundleAdjuster() override = default;

  // Run the optimization (override from base class)
  std::shared_ptr<colmap::BundleAdjustmentSummary> Solve() override;

  // Wireframe-specific options
  bool refine_wireframe() const { return wireframe_options_.refine_wireframe; }
  double weight_wireframe() const {
    return wireframe_options_.weight_wireframe;
  }

  // Get number of wireframe observations
  size_t NumWireframeObservations() const {
    return wireframe_num_observations_;
  }

protected:
  // Add wireframe edge residuals to the problem
  // For each edge, adds:
  // - Point-to-line residuals for each line observation
  // - Line-to-point residuals for each point observation
  void AddWireframeEdgeToProblem(point3D_t point3D_id, line3D_t line3D_id,
                                 double edge_weight,
                                 HolisticReconstruction &reconstruction);

  // Wireframe options (stored as typed struct; base options_ is sliced)
  WireframeBundleAdjustmentOptions wireframe_options_;

  // Wireframe loss function (created from options)
  std::unique_ptr<ceres::LossFunction> wireframe_loss_function_;

  // Track number of wireframe observations
  size_t wireframe_num_observations_ = 0;

  // Storage for scaled loss functions (for proper memory management)
  std::vector<std::unique_ptr<ceres::LossFunction>> wireframe_losses_;
};

// Factory function to create a wireframe bundle adjuster
std::unique_ptr<WireframeBundleAdjuster>
CreateWireframeBundleAdjuster(WireframeBundleAdjustmentOptions options,
                              PointLineBundleAdjustmentConfig config,
                              HolisticReconstruction &reconstruction);

} // namespace estimators
} // namespace limap
