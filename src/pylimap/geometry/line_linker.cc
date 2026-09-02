#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thirdparty/pycolmap/helpers.h>

#include "limap/geometry/line_linker.h"

namespace py = pybind11;

namespace limap {

void BindLineLinker(py::module &m) {
  py::classh<LineLinker2dOptions> PyLineLinker2dOptions(m,
                                                        "LineLinker2dOptions");
  PyLineLinker2dOptions.def(py::init<>())
      .def_readwrite("score_th", &LineLinker2dOptions::score_th)
      .def_readwrite("th_angle", &LineLinker2dOptions::th_angle)
      .def_readwrite("th_overlap", &LineLinker2dOptions::th_overlap)
      .def_readwrite("th_smartangle", &LineLinker2dOptions::th_smartangle)
      .def_readwrite("th_smartoverlap", &LineLinker2dOptions::th_smartoverlap)
      .def_readwrite("th_perp", &LineLinker2dOptions::th_perp)
      .def_readwrite("th_innerseg", &LineLinker2dOptions::th_innerseg)
      .def_readwrite("use_angle", &LineLinker2dOptions::use_angle)
      .def_readwrite("use_overlap", &LineLinker2dOptions::use_overlap)
      .def_readwrite("use_smartangle", &LineLinker2dOptions::use_smartangle)
      .def_readwrite("use_perp", &LineLinker2dOptions::use_perp)
      .def_readwrite("use_innerseg", &LineLinker2dOptions::use_innerseg)
      .def_static("build_default_options",
                  &LineLinker2dOptions::BuildDefaultOptions);
  MakeDataclass(PyLineLinker2dOptions);

  py::classh<LineLinker3dOptions> PyLineLinker3dOptions(m,
                                                        "LineLinker3dOptions");
  PyLineLinker3dOptions.def(py::init<>())
      .def_readwrite("score_th", &LineLinker3dOptions::score_th)
      .def_readwrite("th_angle", &LineLinker3dOptions::th_angle)
      .def_readwrite("th_overlap", &LineLinker3dOptions::th_overlap)
      .def_readwrite("th_smartangle", &LineLinker3dOptions::th_smartangle)
      .def_readwrite("th_smartoverlap", &LineLinker3dOptions::th_smartoverlap)
      .def_readwrite("th_perp", &LineLinker3dOptions::th_perp)
      .def_readwrite("th_innerseg", &LineLinker3dOptions::th_innerseg)
      .def_readwrite("use_angle", &LineLinker3dOptions::use_angle)
      .def_readwrite("use_overlap", &LineLinker3dOptions::use_overlap)
      .def_readwrite("use_smartangle", &LineLinker3dOptions::use_smartangle)
      .def_readwrite("use_perp", &LineLinker3dOptions::use_perp)
      .def_readwrite("use_innerseg", &LineLinker3dOptions::use_innerseg)
      .def_static("build_default_options",
                  &LineLinker3dOptions::BuildDefaultOptions)
      .def_static("build_shared_parent_scoring",
                  &LineLinker3dOptions::BuildSharedParentScoring)
      .def_static("build_spatial_merging",
                  &LineLinker3dOptions::BuildSpatialMerging);
  MakeDataclass(PyLineLinker3dOptions);

  py::classh<LineLinker2d>(m, "LineLinker2d")
      .def(py::init<>())
      .def(py::init<const LineLinker2dOptions &>())
      .def("check_connection", &LineLinker2d::CheckConnection)
      .def("compute_score", &LineLinker2d::ComputeScore)
      .def("add_check",
           [](LineLinker2d &self, py::function fn) {
             self.AddCheck([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("prepend_check",
           [](LineLinker2d &self, py::function fn) {
             self.PrependCheck([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("add_scorer",
           [](LineLinker2d &self, py::function fn) {
             self.AddScorer([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<double>();
             });
           })
      .def("prepend_scorer",
           [](LineLinker2d &self, py::function fn) {
             self.PrependScorer([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<double>();
             });
           })
      .def("clear_checks", &LineLinker2d::ClearChecks)
      .def("clear_scorers", &LineLinker2d::ClearScorers);

  py::classh<LineLinker3d>(m, "LineLinker3d")
      .def(py::init<>())
      .def(py::init<const LineLinker3dOptions &>())
      .def("check_connection", &LineLinker3d::CheckConnection)
      .def("compute_score", &LineLinker3d::ComputeScore)
      .def("add_check",
           [](LineLinker3d &self, py::function fn) {
             self.AddCheck([fn](const Line3d &a, const Line3d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("prepend_check",
           [](LineLinker3d &self, py::function fn) {
             self.PrependCheck([fn](const Line3d &a, const Line3d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("add_scorer",
           [](LineLinker3d &self, py::function fn) {
             self.AddScorer([fn](const Line3d &a, const Line3d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<double>();
             });
           })
      .def("prepend_scorer",
           [](LineLinker3d &self, py::function fn) {
             self.PrependScorer([fn](const Line3d &a, const Line3d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<double>();
             });
           })
      .def("clear_checks", &LineLinker3d::ClearChecks)
      .def("clear_scorers", &LineLinker3d::ClearScorers);

  py::classh<LineLinker>(m, "LineLinker")
      .def(py::init<>())
      .def(py::init<const LineLinker2d &, const LineLinker3d &>())
      .def(py::init<const LineLinker2dOptions &, const LineLinker3dOptions &>())
      .def("linker2d",
           static_cast<LineLinker2d &(LineLinker::*)()>(&LineLinker::Linker2d),
           py::return_value_policy::reference_internal)
      .def("linker3d",
           static_cast<LineLinker3d &(LineLinker::*)()>(&LineLinker::Linker3d),
           py::return_value_policy::reference_internal)
      .def("check_connection_2d", &LineLinker::CheckConnection2d)
      .def("compute_score_2d", &LineLinker::ComputeScore2d)
      .def("check_connection_3d", &LineLinker::CheckConnection3d)
      .def("compute_score_3d", &LineLinker::ComputeScore3d)
      .def("add_check_2d",
           [](LineLinker &self, py::function fn) {
             self.AddCheck2d([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("add_scorer_2d",
           [](LineLinker &self, py::function fn) {
             self.AddScorer2d([fn](const Line2d &a, const Line2d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<double>();
             });
           })
      .def("add_check_3d",
           [](LineLinker &self, py::function fn) {
             self.AddCheck3d([fn](const Line3d &a, const Line3d &b) {
               py::gil_scoped_acquire gil;
               return fn(a, b).cast<bool>();
             });
           })
      .def("add_scorer_3d", [](LineLinker &self, py::function fn) {
        self.AddScorer3d([fn](const Line3d &a, const Line3d &b) {
          py::gil_scoped_acquire gil;
          return fn(a, b).cast<double>();
        });
      });
}

} // namespace limap
