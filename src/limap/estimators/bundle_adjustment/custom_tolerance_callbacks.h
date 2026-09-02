#pragma once

#include <ceres/ceres.h>
#include <cmath>
#include <vector>

#include "limap/geometry/ceres_angle_utils.h"
#include "limap/geometry/ceres_line_functions.h"
#include "limap/geometry/minimal_inf_line3d.h"
#include "limap/util/types.h"

namespace limap {
namespace estimators {

// Custom convergence callback for bundle adjustment.
// Ceres's built-in function_tolerance checks |cost_change|/cost but incorrectly
// fires on unsuccessful steps (checked before IsStepSuccessful in
// trust_region_minimizer.cc). This callback applies the same criterion only on
// successful steps, preventing premature termination on rejected steps.
class CustomToleranceCallback : public ceres::IterationCallback {
public:
  explicit CustomToleranceCallback(double function_tolerance)
      : function_tolerance_(function_tolerance) {}

  ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) override {
    if (!summary.step_is_successful) {
      return ceres::SOLVER_CONTINUE;
    }
    if (prev_cost_ < 0.0) {
      prev_cost_ = summary.cost;
      return ceres::SOLVER_CONTINUE;
    }
    if (summary.cost > 0.0 &&
        std::abs(summary.cost_change) / summary.cost < function_tolerance_) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }
    prev_cost_ = summary.cost;
    return ceres::SOLVER_CONTINUE;
  }

private:
  double function_tolerance_;
  double prev_cost_ = -1.0;
};

// Info needed to manually evaluate one angle-type residual in the convergence
// callback. Types based on the geometric constraint:
//   PARALLELISM: line direction || VP direction
//     param1 = 6D MinimalPlucker (line), param2 = 3D VP direction
//     residual depends on vp_residual_type (SINE or COSINE)
//   ORTHOGONALITY: two directions perpendicular
//     param1/param2 = 3D VP or 4D plane (first 3 = normal)
//     residual = |dot(n1, n2)| (= |cos(angle)|)
//   VP_PARALLELISM: two near-parallel VPs
//     param1/param2 = 3D VP directions
//     residual = |sin(angle)| — 0 when parallel
//   PLANE_PARALLELISM: two near-parallel plane normals
//     param1/param2 = 4D plane params (first 3 = normal)
//     residual = |sin(angle)| — 0 when parallel
struct AngleResidualInfo {
  enum Type { PARALLELISM, ORTHOGONALITY, VP_PARALLELISM, PLANE_PARALLELISM };
  Type type;
  const double *param1;
  const double *param2;
  double weight; // ScaledLoss weight (e.g., 5000 for VP, 1e5 for constraints)
  const ceres::LossFunction *loss; // Cauchy/etc (owned by BA class)
};

// Convergence callback that strips angle cost (VP + orthogonality) from the
// convergence check. Angle residuals have different convergence dynamics than
// pixel residuals and can delay termination when checked jointly.
//
// Checks: |non_angle_change| / non_angle_cost < tolerance
// where non_angle = total - angle (i.e., core pixel + structure plane costs).
//
// Falls back to total-cost check when no angle residuals are registered.
class AngleCostAwareToleranceCallback : public ceres::IterationCallback {
public:
  // vp_use_cosine: if true, PARALLELISM residual = 1 - |cos(angle)|;
  //                if false, PARALLELISM residual = sin(angle) (default "sin")
  AngleCostAwareToleranceCallback(
      double function_tolerance,
      const std::vector<AngleResidualInfo> &angle_infos,
      bool vp_use_cosine = false)
      : function_tolerance_(function_tolerance), angle_infos_(angle_infos),
        vp_use_cosine_(vp_use_cosine) {}

  // Compute total angle cost by evaluating all registered angle residuals.
  // Each residual is evaluated, passed through its loss function, scaled by
  // weight, and summed. Returns 0.5 * sum (matching Ceres's 0.5 * rho(s)
  // convention).
  inline double ComputeAngleCost() const {
    double total = 0.0;
    for (const auto &info : angle_infos_) {
      double residual = 0.0;
      if (info.type == AngleResidualInfo::PARALLELISM) {
        // param1 = 6D MinimalPlucker line, param2 = 3D VP direction
        double d[3], m[3];
        MinimalPluckerToPlucker<double>(info.param1, d, m);
        if (vp_use_cosine_) {
          residual = 1.0 - AngleCosine3D<double>(d, info.param2);
        } else {
          residual = AngleSine3D<double>(d, info.param2);
        }
      } else if (info.type == AngleResidualInfo::ORTHOGONALITY) {
        // param1/param2 = 3D VP or 4D plane (first 3 = normal)
        residual = AngleCosine3D<double>(info.param1, info.param2);
      } else {
        // VP_PARALLELISM or PLANE_PARALLELISM:
        // param1/param2 = 3D VP or 4D plane (first 3 = direction/normal)
        residual = AngleSine3D<double>(info.param1, info.param2);
      }

      // Apply loss function: loss->Evaluate(sq_residual, rho)
      // rho[0] = loss value, so cost contribution = 0.5 * weight * rho[0]
      const double sq_residual = residual * residual;
      double rho[3];
      if (info.loss) {
        info.loss->Evaluate(sq_residual, rho);
      } else {
        // Trivial loss: rho[0] = s
        rho[0] = sq_residual;
      }
      total += info.weight * rho[0];
    }
    return 0.5 * total;
  }

  inline ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) override {
    if (!summary.step_is_successful) {
      return ceres::SOLVER_CONTINUE;
    }

    // No angle residuals — fall back to total-cost check
    if (angle_infos_.empty()) {
      if (prev_angle_cost_ >= 0.0 && summary.cost > 0.0 &&
          std::abs(summary.cost_change) / summary.cost < function_tolerance_) {
        return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
      }
      prev_angle_cost_ = 0.0;
      return ceres::SOLVER_CONTINUE;
    }

    const double angle_cost = ComputeAngleCost();

    if (prev_angle_cost_ < 0.0) {
      // First successful step — store baseline
      prev_angle_cost_ = angle_cost;
      return ceres::SOLVER_CONTINUE;
    }

    const double angle_cost_change = angle_cost - prev_angle_cost_;
    prev_angle_cost_ = angle_cost;

    const double non_angle_cost = summary.cost - angle_cost;
    const double non_angle_cost_change =
        summary.cost_change - angle_cost_change;

    if (non_angle_cost > 0.0 &&
        std::abs(non_angle_cost_change) / non_angle_cost <
            function_tolerance_) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }

    return ceres::SOLVER_CONTINUE;
  }

private:
  double function_tolerance_;
  const std::vector<AngleResidualInfo> &angle_infos_;
  bool vp_use_cosine_;
  double prev_angle_cost_ = -1.0;
};

} // namespace estimators
} // namespace limap
