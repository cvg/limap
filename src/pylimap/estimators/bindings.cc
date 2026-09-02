#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <thirdparty/pycolmap/helpers.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include <PoseLib/types.h>

namespace py = pybind11;

namespace limap {

void bind_absolute_pose(py::module &);
void bind_bundle_adjustment(py::module &);
void bind_estimators_group3d(py::module &);
void bind_line3d_fitting(py::module &);
void bind_triangulation(py::module &);

void bind_poselib(py::module &m) {
  auto PyPoseLibRansacOptions =
      py::classh<poselib::RansacOptions>(m, "PoseLibRansacOptions")
          .def(py::init<>())
          .def_readwrite("success_prob", &poselib::RansacOptions::success_prob)
          .def_readwrite("min_iterations",
                         &poselib::RansacOptions::min_iterations)
          .def_readwrite("max_iterations",
                         &poselib::RansacOptions::max_iterations)
          .def_readwrite("progressive_sampling",
                         &poselib::RansacOptions::progressive_sampling)
          .def_readwrite("seed", &poselib::RansacOptions::seed);
  MakeDataclass(PyPoseLibRansacOptions);

  auto PyPoseLibRansacStats =
      py::classh<poselib::RansacStats>(m, "PoseLibRansacStats")
          .def(py::init<>())
          .def_readwrite("refinements", &poselib::RansacStats::refinements)
          .def_readwrite("iterations", &poselib::RansacStats::iterations)
          .def_readwrite("num_inliers", &poselib::RansacStats::num_inliers)
          .def_readwrite("inlier_ratio", &poselib::RansacStats::inlier_ratio)
          .def_readwrite("model_score", &poselib::RansacStats::model_score);
  MakeDataclass(PyPoseLibRansacStats);
}

void bind_estimators(py::module &m) {
  bind_poselib(m);

  // absolute pose
  py::module_ _absolute_pose = m.def_submodule("_absolute_pose");
  bind_absolute_pose(_absolute_pose);

  // bundle adjustment
  py::module_ _bundle_adjustment = m.def_submodule("_bundle_adjustment");
  bind_bundle_adjustment(_bundle_adjustment);

  // group3d fitting
  py::module_ _group3d = m.def_submodule("_group3d");
  bind_estimators_group3d(_group3d);

  // line3d fitting
  py::module_ _line3d = m.def_submodule("_line3d");
  bind_line3d_fitting(_line3d);

  // triangulation
  py::module_ _triangulation = m.def_submodule("_triangulation");
  bind_triangulation(_triangulation);
}

} // namespace limap
