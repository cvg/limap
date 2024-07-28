#ifndef LIMAP_OPTIMIZE_LINE_REFINEMENT_REFINEMENT_CONFIG_H_
#define LIMAP_OPTIMIZE_LINE_REFINEMENT_REFINEMENT_CONFIG_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ceresbase/interpolation.h"
#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace line_refinement {

class RefinementConfig {
public:
  RefinementConfig() {
    line_geometric_loss_function.reset(new ceres::CauchyLoss(0.25));
    vp_loss_function.reset(new ceres::TrivialLoss());
    heatmap_loss_function.reset(new ceres::HuberLoss(0.001));
    fconsis_loss_function.reset(new ceres::CauchyLoss(0.25));

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
  }
  RefinementConfig(py::dict dict) : RefinementConfig() {
    ASSIGN_PYDICT_ITEM(dict, use_geometric, bool);
    ASSIGN_PYDICT_ITEM(dict, min_num_images, int);
    ASSIGN_PYDICT_ITEM(dict, num_outliers_aggregate, int);
    ASSIGN_PYDICT_ITEM(dict, print_summary, bool);
    ASSIGN_PYDICT_ITEM(dict, geometric_alpha, double);
    ASSIGN_PYDICT_ITEM(dict, vp_multiplier, double);

    ASSIGN_PYDICT_ITEM(dict, sample_range_min, double);
    ASSIGN_PYDICT_ITEM(dict, sample_range_max, double);
    ASSIGN_PYDICT_ITEM(dict, n_samples_heatmap, int);
    ASSIGN_PYDICT_ITEM(dict, heatmap_multiplier, double);
    ASSIGN_PYDICT_ITEM(dict, n_samples_feature, int);
    ASSIGN_PYDICT_ITEM(dict, use_ref_descriptor, bool);
    ASSIGN_PYDICT_ITEM(dict, ref_multiplier, double);
    ASSIGN_PYDICT_ITEM(dict, fconsis_multiplier, double);
  }
  bool use_geometric = true;
  int min_num_images = 4;
  int num_outliers_aggregate = 2;

  // solver config
  ceres::Solver::Options solver_options;
  ceres::Problem::Options problem_options;
  bool print_summary = true;

  // geometric config
  std::shared_ptr<ceres::LossFunction> line_geometric_loss_function;
  double geometric_alpha = 10.0;

  // vp config
  std::shared_ptr<ceres::LossFunction> vp_loss_function;
  double vp_multiplier = 1.0;

  // heatmap config
  double sample_range_min = 0.05;
  double sample_range_max = 0.95;
  int n_samples_heatmap = 10;
  InterpolationConfig heatmap_interpolation_config;
  std::shared_ptr<ceres::LossFunction> heatmap_loss_function;
  double heatmap_multiplier = 1.0;

  // feature config
  int n_samples_feature = 100;
  bool use_ref_descriptor = false;
  double ref_multiplier = 5.0;
  InterpolationConfig feature_interpolation_config;
  std::shared_ptr<ceres::LossFunction> fconsis_loss_function;
  double fconsis_multiplier = 1.0;
};

} // namespace line_refinement

} // namespace optimize

} // namespace limap

#endif
