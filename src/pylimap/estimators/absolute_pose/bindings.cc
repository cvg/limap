#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <thirdparty/pycolmap/helpers.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "limap/estimators/absolute_pose/point_line_absolute_pose.h"

namespace py = pybind11;
using namespace py::literals;

namespace limap {

void bind_absolute_pose(py::module &m) {
  using namespace estimators::absolute_pose;

  // PointLineAbsolutePoseOptions
  auto PyPointLineAbsolutePoseOptions =
      py::classh<PointLineAbsolutePoseOptions>(m,
                                               "PointLineAbsolutePoseOptions")
          .def(py::init<>())
          .def_readwrite("max_error_point",
                         &PointLineAbsolutePoseOptions::max_error_point)
          .def_readwrite("max_error_line",
                         &PointLineAbsolutePoseOptions::max_error_line)
          .def_readwrite("weight_point",
                         &PointLineAbsolutePoseOptions::weight_point)
          .def_readwrite("weight_line",
                         &PointLineAbsolutePoseOptions::weight_line)
          .def_readwrite("max_iterations",
                         &PointLineAbsolutePoseOptions::max_iterations)
          .def_readwrite("min_iterations",
                         &PointLineAbsolutePoseOptions::min_iterations)
          .def_readwrite("success_prob",
                         &PointLineAbsolutePoseOptions::success_prob)
          .def_readwrite("random_seed",
                         &PointLineAbsolutePoseOptions::random_seed)
          .def_readwrite("seed", &PointLineAbsolutePoseOptions::seed)
          .def_readwrite("estimate_focal_length",
                         &PointLineAbsolutePoseOptions::estimate_focal_length);
  MakeDataclass(PyPointLineAbsolutePoseOptions);

  // PointLineAbsolutePoseResult
  auto PyPointLineAbsolutePoseResult =
      py::classh<PointLineAbsolutePoseResult>(m, "PointLineAbsolutePoseResult")
          .def(py::init<>())
          .def_readonly("pose", &PointLineAbsolutePoseResult::pose)
          .def_readonly("camera", &PointLineAbsolutePoseResult::camera)
          .def_readonly("num_inliers",
                        &PointLineAbsolutePoseResult::num_inliers)
          .def_readonly("num_inliers_points",
                        &PointLineAbsolutePoseResult::num_inliers_points)
          .def_readonly("num_inliers_lines",
                        &PointLineAbsolutePoseResult::num_inliers_lines)
          .def_readonly("iterations", &PointLineAbsolutePoseResult::iterations)
          .def_readonly("model_score",
                        &PointLineAbsolutePoseResult::model_score)
          .def_readonly("inliers_points",
                        &PointLineAbsolutePoseResult::inliers_points)
          .def_readonly("inliers_lines",
                        &PointLineAbsolutePoseResult::inliers_lines)
          .def_readonly("success", &PointLineAbsolutePoseResult::success);
  MakeDataclass(PyPointLineAbsolutePoseResult);

  m.def("estimate_point_line_absolute_pose", &EstimatePointLineAbsolutePose,
        "l3ds"_a, "l2ds"_a, "p3ds"_a, "p2ds"_a, "cam"_a, "options"_a);
}

} // namespace limap
