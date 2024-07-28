// Modified from the pixel-perfect-sfm project

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_limap/helpers.h"

#include "ceresbase/ceres_extensions.h"
#include "ceresbase/interpolation.h"
#include <ceres/ceres.h>

PYBIND11_MAKE_OPAQUE(std::vector<ceres::IterationCallback *>);

namespace limap {

void bind_ceres(py::module &m) {
  py::class_<ceres::LossFunction, PyLossFunction,
             std::shared_ptr<ceres::LossFunction>>(m, "LossFunction")
      .def(py::init<>())
      .def("evaluate", [](ceres::LossFunction &self, float v) {
        Eigen::Vector3d rho;
        self.Evaluate(v, rho.data());
        return rho;
      });

  py::class_<ceres::TrivialLoss, ceres::LossFunction,
             std::shared_ptr<ceres::TrivialLoss>>(m, "TrivialLoss")
      .def(py::init<>());

  py::class_<ceres::HuberLoss, ceres::LossFunction,
             std::shared_ptr<ceres::HuberLoss>>(m, "HuberLoss")
      .def(py::init<double>());

  py::class_<ceres::SoftLOneLoss, ceres::LossFunction,
             std::shared_ptr<ceres::SoftLOneLoss>>(m, "SoftLOneLoss")
      .def(py::init<double>());

  py::class_<ceres::CauchyLoss, ceres::LossFunction,
             std::shared_ptr<ceres::CauchyLoss>>(m, "CauchyLoss")
      .def(py::init<double>());

  py::enum_<ceres::CallbackReturnType>(m, "CallbackReturnType")
      .value("SOLVER_CONTINUE", ceres::CallbackReturnType::SOLVER_CONTINUE)
      .value("SOLVER_ABORT", ceres::CallbackReturnType::SOLVER_ABORT)
      .value("SOLVER_TERMINATE_SUCCESSFULLY",
             ceres::CallbackReturnType::SOLVER_TERMINATE_SUCCESSFULLY);
  py::class_<ceres::IterationSummary>(m, "IterationSummary")
      .def_readonly("step_is_valid", &ceres::IterationSummary::step_is_valid)
      .def_readonly("iteration", &ceres::IterationSummary::iteration)
      .def_readonly("step_is_nonmonotonic",
                    &ceres::IterationSummary::step_is_nonmonotonic)
      .def_readonly("step_is_successful",
                    &ceres::IterationSummary::step_is_successful)
      .def_readonly("cost", &ceres::IterationSummary::cost)
      .def_readonly("cost_change", &ceres::IterationSummary::cost_change)
      .def_readonly("gradient_max_norm",
                    &ceres::IterationSummary::gradient_max_norm)
      .def_readonly("gradient_norm", &ceres::IterationSummary::gradient_norm)
      .def_readonly("step_norm", &ceres::IterationSummary::step_norm)
      .def_readonly("relative_decrease",
                    &ceres::IterationSummary::relative_decrease)
      .def_readonly("trust_region_radius",
                    &ceres::IterationSummary::trust_region_radius)
      .def_readonly("eta", &ceres::IterationSummary::eta)
      .def_readonly("step_size", &ceres::IterationSummary::step_size)
      .def_readonly("line_search_function_evaluations",
                    &ceres::IterationSummary::line_search_function_evaluations)
      .def_readonly("line_search_gradient_evaluations",
                    &ceres::IterationSummary::line_search_gradient_evaluations)
      .def_readonly("line_search_iterations",
                    &ceres::IterationSummary::line_search_iterations)
      .def_readonly("linear_solver_iterations",
                    &ceres::IterationSummary::linear_solver_iterations)
      .def_readonly("iteration_time_in_seconds",
                    &ceres::IterationSummary::iteration_time_in_seconds)
      .def_readonly("step_solver_time_in_seconds",
                    &ceres::IterationSummary::step_solver_time_in_seconds)
      .def_readonly("cumulative_time_in_seconds",
                    &ceres::IterationSummary::cumulative_time_in_seconds);
  py::class_<ceres::IterationCallback, PyIterationCallback /*<--- trampoline*/>(
      m, "IterationCallback")
      .def(py::init<>())
      .def("__call__", &ceres::IterationCallback::operator());

  py::bind_vector<std::vector<ceres::IterationCallback *>>(
      m, "ListIterationCallbackPtr");

  py::enum_<ceres::LoggingType>(m, "LoggingType")
      .value("SILENT", ceres::LoggingType::SILENT)
      .value("STDOUT", ceres::LoggingType::PER_MINIMIZER_ITERATION);

  py::class_<ceres::Solver::Options>(m, "SolverOptions")
      .def(py::init<>())
      .def_readwrite("max_num_iterations",
                     &ceres::Solver::Options::max_num_iterations)
      .def_readwrite("use_inner_iterations",
                     &ceres::Solver::Options::use_inner_iterations)
      .def_readwrite("inner_iteration_tolerance",
                     &ceres::Solver::Options::inner_iteration_tolerance)
      .def_readwrite("max_linear_solver_iterations",
                     &ceres::Solver::Options::max_linear_solver_iterations)
      .def_readwrite("minimizer_progress_to_stdout",
                     &ceres::Solver::Options::minimizer_progress_to_stdout)
      .def_readwrite("callbacks", &ceres::Solver::Options::callbacks)
      .def_readwrite("update_state_every_iteration",
                     &ceres::Solver::Options::update_state_every_iteration)
      .def_readwrite("minimizer_type", &ceres::Solver::Options::minimizer_type)
      .def_readwrite("line_search_direction_type",
                     &ceres::Solver::Options::line_search_direction_type)
      .def_readwrite("line_search_type",
                     &ceres::Solver::Options::line_search_type)
      .def_readwrite("nonlinear_conjugate_gradient_type",
                     &ceres::Solver::Options::nonlinear_conjugate_gradient_type)
      .def_readwrite("max_lbfgs_rank", &ceres::Solver::Options::max_lbfgs_rank)
      .def_readwrite(
          "use_approximate_eigenvalue_bfgs_scaling",
          &ceres::Solver::Options::use_approximate_eigenvalue_bfgs_scaling)
      .def_readwrite("line_search_interpolation_type",
                     &ceres::Solver::Options::line_search_interpolation_type)
      .def_readwrite("min_line_search_step_size",
                     &ceres::Solver::Options::min_line_search_step_size)
      .def_readwrite(
          "line_search_sufficient_function_decrease",
          &ceres::Solver::Options::line_search_sufficient_function_decrease)
      .def_readwrite("max_line_search_step_contraction",
                     &ceres::Solver::Options::max_line_search_step_contraction)
      .def_readwrite("min_line_search_step_contraction",
                     &ceres::Solver::Options::min_line_search_step_contraction)
      .def_readwrite(
          "max_num_line_search_step_size_iterations",
          &ceres::Solver::Options::max_num_line_search_step_size_iterations)
      .def_readwrite(
          "max_num_line_search_direction_restarts",
          &ceres::Solver::Options::max_num_line_search_direction_restarts)
      .def_readwrite(
          "line_search_sufficient_curvature_decrease",
          &ceres::Solver::Options::line_search_sufficient_curvature_decrease)
      .def_readwrite("max_line_search_step_expansion",
                     &ceres::Solver::Options::max_line_search_step_expansion)
      .def_readwrite("trust_region_strategy_type",
                     &ceres::Solver::Options::trust_region_strategy_type)
      .def_readwrite("dogleg_type", &ceres::Solver::Options::dogleg_type)
      .def_readwrite("use_nonmonotonic_steps",
                     &ceres::Solver::Options::use_nonmonotonic_steps)
      .def_readwrite(
          "max_consecutive_nonmonotonic_steps",
          &ceres::Solver::Options::max_consecutive_nonmonotonic_steps)
      .def_readwrite("max_solver_time_in_seconds",
                     &ceres::Solver::Options::max_solver_time_in_seconds)
      .def_readwrite("num_threads", &ceres::Solver::Options::num_threads)
      .def_readwrite("initial_trust_region_radius",
                     &ceres::Solver::Options::initial_trust_region_radius)
      .def_readwrite("max_trust_region_radius",
                     &ceres::Solver::Options::max_trust_region_radius)
      .def_readwrite("min_trust_region_radius",
                     &ceres::Solver::Options::min_trust_region_radius)
      .def_readwrite("min_relative_decrease",
                     &ceres::Solver::Options::min_relative_decrease)
      .def_readwrite("min_lm_diagonal",
                     &ceres::Solver::Options::min_lm_diagonal)
      .def_readwrite("max_lm_diagonal",
                     &ceres::Solver::Options::max_lm_diagonal)
      .def_readwrite("max_num_consecutive_invalid_steps",
                     &ceres::Solver::Options::max_num_consecutive_invalid_steps)
      .def_readwrite("function_tolerance",
                     &ceres::Solver::Options::function_tolerance)
      .def_readwrite("gradient_tolerance",
                     &ceres::Solver::Options::gradient_tolerance)
      .def_readwrite("parameter_tolerance",
                     &ceres::Solver::Options::parameter_tolerance)
      .def_readwrite("linear_solver_type",
                     &ceres::Solver::Options::linear_solver_type)
      .def_readwrite("preconditioner_type",
                     &ceres::Solver::Options::preconditioner_type)
      .def_readwrite("visibility_clustering_type",
                     &ceres::Solver::Options::visibility_clustering_type)
      .def_readwrite("dense_linear_algebra_library_type",
                     &ceres::Solver::Options::dense_linear_algebra_library_type)
      .def_readwrite(
          "sparse_linear_algebra_library_type",
          &ceres::Solver::Options::sparse_linear_algebra_library_type)
      .def_readwrite("use_explicit_schur_complement",
                     &ceres::Solver::Options::use_explicit_schur_complement)
      .def_readwrite("dynamic_sparsity",
                     &ceres::Solver::Options::dynamic_sparsity)
      .def_readwrite("logging_type", &ceres::Solver::Options::logging_type);
}

template <typename dtype>
void BindBaseTemplate(py::module &m, std::string type_suffix) {
  using InterpQuery = InterpolationQuery<dtype>;
  py::class_<InterpQuery>(m, ("InterpolationQuery" + type_suffix).c_str())
      .def_readwrite("scale", &InterpQuery::scale)
      .def_readwrite("shape", &InterpQuery::shape)
      .def_readwrite("corner", &InterpQuery::corner);
}

void bind_ceresbase(py::module &m) {
  bind_ceres(m);

  py::enum_<InterpolatorType>(m, "InterpolatorType")
      .value("BICUBIC", InterpolatorType::BICUBIC)
      .value("BILINEAR", InterpolatorType::BILINEAR)
      .value("POLYGRADIENTFIELD", InterpolatorType::POLYGRADIENTFIELD)
      .value("BICUBICGRADIENTFIELD", InterpolatorType::BICUBICGRADIENTFIELD)
      .value("BICUBICCHAIN", InterpolatorType::BICUBICCHAIN)
      .value("CERES_BICUBIC", InterpolatorType::CERES_BICUBIC);

  py::class_<InterpolationConfig>(m, "InterpolationConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("l2_normalize", &InterpolationConfig::l2_normalize)
      .def_readwrite("ncc_normalize", &InterpolationConfig::ncc_normalize)
      .def_readwrite("nodes", &InterpolationConfig::nodes)
      .def_readwrite("n_nodes", &InterpolationConfig::n_nodes)
      .def_readwrite("interpolation", &InterpolationConfig::interpolation)
      .def_readwrite("check_bounds", &InterpolationConfig::check_bounds)
      .def_readwrite("condition_gradient",
                     &InterpolationConfig::condition_gradient)
      .def_readwrite("min_gradient", &InterpolationConfig::min_gradient);

  BindBaseTemplate<float16>(m, "_f16");
  BindBaseTemplate<double>(m, "_f64");
  BindBaseTemplate<float>(m, "_f32");

  m.def("fit_cubic_polynomial", &FitCubicPolynomial);
  m.def("eval_cubic_polynomial", [](Eigen::Vector4d &coeffs, double x) {
    double res;
    EvalCubicPolynomial(coeffs, x, &res, NULL);
    return res;
  });
}

} // namespace limap
