#ifndef LIMAP_OPTIMIZE_HYBRID_LOCALIZATION_HYBRID_LOCALIZATION_CONFIG_H_
#define LIMAP_OPTIMIZE_HYBRID_LOCALIZATION_HYBRID_LOCALIZATION_CONFIG_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "limap/ceresbase/ceres_extensions.h"
#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace hybrid_localization {

enum LineLocCostFunction {
  E2DMidpointDist2,
  E2DMidpointAngleDist3,
  E2DPerpendicularDist2,
  E2DPerpendicularDist4,
  E3DLineLineDist2,
  E3DPlaneLineDist2,
};

enum LineLocCostFunctionWeight {
  ENoneWeight,
  ECosineWeight,
  ELine3dppWeight,
  ELengthWeight,
  EInvLengthWeight
};

class LineLocConfig {
public:
  LineLocConfig() {
    loss_function.reset(new ceres::TrivialLoss());
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
    solver_options.logging_type = ceres::SILENT;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif // CERES_VERSION_MAJOR
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    cost_function = E2DPerpendicularDist2;
    cost_function_weight = ENoneWeight;
  }
  LineLocConfig(py::dict dict) : LineLocConfig() {
    ASSIGN_PYDICT_ITEM(dict, weight_line, double);
    ASSIGN_PYDICT_ITEM(dict, weight_point, double);
    ASSIGN_PYDICT_ITEM(dict, print_summary, bool);
    ASSIGN_PYDICT_ITEM(dict, loss_function,
                       std::shared_ptr<ceres::LossFunction>);
    if (dict.contains("solver_options"))
      AssignSolverOptionsFromDict(solver_options, dict["solver_options"]);
  }
  double weight_line = 1.0;
  double weight_point = 1.0;
  bool points_3d_dist = false;

  ceres::Solver::Options solver_options;
  bool print_summary = true;

  // These are not set from py::dict;
  ceres::Problem::Options problem_options;
  LineLocCostFunction cost_function;
  LineLocCostFunctionWeight cost_function_weight;
  std::shared_ptr<ceres::LossFunction> loss_function;
};

} // namespace hybrid_localization

} // namespace optimize

} // namespace limap

#endif
