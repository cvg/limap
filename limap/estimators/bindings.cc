#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include <RansacLib/ransac.h>

#include "estimators/absolute_pose/bindings.cc"
#include "estimators/extended_hybrid_ransac.h"

namespace py = pybind11;

namespace limap {

void bind_pose(py::module &m);
void bind_ransaclib(py::module &m) {
  using ExtendedHybridLORansacOptions =
      estimators::ExtendedHybridLORansacOptions;
  py::class_<ransac_lib::RansacStatistics>(m, "RansacStatistics")
      .def(py::init<>())
      .def_readwrite("num_iterations",
                     &ransac_lib::RansacStatistics::num_iterations)
      .def_readwrite("best_num_inliers",
                     &ransac_lib::RansacStatistics::best_num_inliers)
      .def_readwrite("best_model_score",
                     &ransac_lib::RansacStatistics::best_model_score)
      .def_readwrite("inlier_ratio",
                     &ransac_lib::RansacStatistics::inlier_ratio)
      .def_readwrite("inlier_indices",
                     &ransac_lib::RansacStatistics::inlier_indices)
      .def_readwrite("number_lo_iterations",
                     &ransac_lib::RansacStatistics::number_lo_iterations);

  py::class_<ransac_lib::RansacOptions>(m, "RansacOptions")
      .def(py::init<>())
      .def_readwrite("min_num_iterations_",
                     &ransac_lib::RansacOptions::min_num_iterations_)
      .def_readwrite("max_num_iterations_",
                     &ransac_lib::RansacOptions::max_num_iterations_)
      .def_readwrite("success_probability_",
                     &ransac_lib::RansacOptions::success_probability_)
      .def_readwrite("squared_inlier_threshold_",
                     &ransac_lib::RansacOptions::squared_inlier_threshold_)
      .def_readwrite("random_seed_", &ransac_lib::RansacOptions::random_seed_);

  py::class_<ransac_lib::LORansacOptions>(
      m, "LORansacOptions", "Inherits :class:`~limap.estimators.RansacOptions`")
      .def(py::init<>())
      .def_readwrite("min_num_iterations_",
                     &ransac_lib::LORansacOptions::min_num_iterations_)
      .def_readwrite("max_num_iterations_",
                     &ransac_lib::LORansacOptions::max_num_iterations_)
      .def_readwrite("success_probability_",
                     &ransac_lib::LORansacOptions::success_probability_)
      .def_readwrite("squared_inlier_threshold_",
                     &ransac_lib::LORansacOptions::squared_inlier_threshold_)
      .def_readwrite("random_seed_", &ransac_lib::LORansacOptions::random_seed_)
      .def_readwrite("num_lo_steps_",
                     &ransac_lib::LORansacOptions::num_lo_steps_)
      .def_readwrite("threshold_multiplier_",
                     &ransac_lib::LORansacOptions::threshold_multiplier_)
      .def_readwrite("num_lsq_iterations_",
                     &ransac_lib::LORansacOptions::num_lsq_iterations_)
      .def_readwrite("min_sample_multiplicator_",
                     &ransac_lib::LORansacOptions::min_sample_multiplicator_)
      .def_readwrite("non_min_sample_multiplier_",
                     &ransac_lib::LORansacOptions::non_min_sample_multiplier_)
      .def_readwrite("lo_starting_iterations_",
                     &ransac_lib::LORansacOptions::lo_starting_iterations_)
      .def_readwrite("final_least_squares_",
                     &ransac_lib::LORansacOptions::final_least_squares_);

  // hybrid ransac
  py::class_<ransac_lib::HybridRansacStatistics>(
      m, "HybridRansacStatistics",
      "Inherits :class:`~limap.estimators.RansacStatistics`")
      .def(py::init<>())
      .def_readwrite("num_iterations_total",
                     &ransac_lib::HybridRansacStatistics::num_iterations_total)
      .def_readwrite(
          "num_iterations_per_solver",
          &ransac_lib::HybridRansacStatistics::num_iterations_per_solver)
      .def_readwrite("best_num_inliers",
                     &ransac_lib::HybridRansacStatistics::best_num_inliers)
      .def_readwrite("best_solver_type",
                     &ransac_lib::HybridRansacStatistics::best_solver_type)
      .def_readwrite("best_model_score",
                     &ransac_lib::HybridRansacStatistics::best_model_score)
      .def_readwrite("inlier_ratios",
                     &ransac_lib::HybridRansacStatistics::inlier_ratios)
      .def_readwrite("inlier_indices",
                     &ransac_lib::HybridRansacStatistics::inlier_indices)
      .def_readwrite("number_lo_iterations",
                     &ransac_lib::HybridRansacStatistics::number_lo_iterations);

  py::class_<ExtendedHybridLORansacOptions>(
      m, "HybridLORansacOptions",
      "Inherits :class:`~limap.estimators.LORansacOptions`")
      .def(py::init<>())
      .def_readwrite("min_num_iterations_",
                     &ExtendedHybridLORansacOptions::min_num_iterations_)
      .def_readwrite("max_num_iterations_",
                     &ExtendedHybridLORansacOptions::max_num_iterations_)
      .def_readwrite(
          "max_num_iterations_per_solver_",
          &ExtendedHybridLORansacOptions::max_num_iterations_per_solver_)
      .def_readwrite("success_probability_",
                     &ExtendedHybridLORansacOptions::success_probability_)
      .def_readwrite("squared_inlier_thresholds_",
                     &ExtendedHybridLORansacOptions::squared_inlier_thresholds_)
      .def_readwrite("data_type_weights_",
                     &ExtendedHybridLORansacOptions::data_type_weights_)
      .def_readwrite("random_seed_",
                     &ExtendedHybridLORansacOptions::random_seed_)
      .def_readwrite("num_lo_steps_",
                     &ExtendedHybridLORansacOptions::num_lo_steps_)
      .def_readwrite("threshold_multiplier_",
                     &ExtendedHybridLORansacOptions::threshold_multiplier_)
      .def_readwrite("num_lsq_iterations_",
                     &ExtendedHybridLORansacOptions::num_lsq_iterations_)
      .def_readwrite("min_sample_multiplicator_",
                     &ExtendedHybridLORansacOptions::min_sample_multiplicator_)
      .def_readwrite("non_min_sample_multiplier_",
                     &ExtendedHybridLORansacOptions::non_min_sample_multiplier_)
      .def_readwrite("lo_starting_iterations_",
                     &ExtendedHybridLORansacOptions::lo_starting_iterations_)
      .def_readwrite("final_least_squares_",
                     &ExtendedHybridLORansacOptions::final_least_squares_);
}

void bind_estimators(py::module &m) {
  using namespace estimators;

  bind_ransaclib(m);
  bind_absolute_pose(m);
}

} // namespace limap
