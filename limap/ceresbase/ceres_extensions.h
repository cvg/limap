#ifndef LIMAP_CERESBASE_CERES_EXTENSIONS_H
#define LIMAP_CERESBASE_CERES_EXTENSIONS_H

// Modified from the pixel-perfect-sfm project

#include "_limap/helpers.h"
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

class PyIterationCallback : public ceres::IterationCallback {
public:
  using ceres::IterationCallback::IterationCallback;

  ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        ceres::CallbackReturnType, // Return type (ret_type)
        ceres::IterationCallback,  // Parent class (cname)
        "__call__",                // Name of method in Python (name)
        operator(),                // Name of function in C++ (fn)
        summary);
  }
};

class PyLossFunction : public ceres::LossFunction {
public:
  /* Inherit the constructors */
  using ceres::LossFunction::LossFunction;

  void Evaluate(double sq_norm, double out[3]) const override {}
};

inline void AssignSolverOptionsFromDict(ceres::Solver::Options &solver_options,
                                        py::dict dict) {
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, function_tolerance, double)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, gradient_tolerance, double)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, parameter_tolerance,
                               double)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict,
                               minimizer_progress_to_stdout, bool)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict,
                               max_linear_solver_iterations, int)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, max_num_iterations, int)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict,
                               max_num_consecutive_invalid_steps, int)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict,
                               max_consecutive_nonmonotonic_steps, int)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, use_inner_iterations, bool)
  ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, inner_iteration_tolerance,
                               double)
}

#endif
