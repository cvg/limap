#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/group2d.h"
#include <thirdparty/pycolmap/helpers.h>

namespace py = pybind11;

namespace limap {

void BindGroup2d(py::module &m) {
  auto PyAssociatedFeature2d =
      py::classh<AssociatedFeature2d>(m, "AssociatedFeature2d")
          .def(py::init<>())
          .def(py::init<feature2D_t, double>(), py::arg("idx"),
               py::arg("w") = 1.0)
          .def_readwrite("idx", &AssociatedFeature2d::idx)
          .def_readwrite("w", &AssociatedFeature2d::w);
  MakeDataclass(PyAssociatedFeature2d);

  auto PyGroup2d =
      py::classh<Group2d>(m, "Group2d")
          .def(py::init<>())
          .def(py::init<GroupType>())
          .def(py::init<GroupType, const std::vector<AssociatedFeature2d> &,
                        const std::vector<AssociatedFeature2d> &>(),
               py::arg("type"), py::arg("points"),
               py::arg("lines") = std::vector<AssociatedFeature2d>{})
          .def_readwrite("type", &Group2d::type)
          .def_property("params", &Group2d::GetParams, &Group2d::SetParams)
          .def_readwrite("points", &Group2d::points)
          .def_readwrite("lines", &Group2d::lines)
          .def_readwrite("group3D_id", &Group2d::group3D_id)
          .def("get_max_point_id", &Group2d::GetMaxPointId)
          .def("get_max_line_id", &Group2d::GetMaxLineId)
          .def("add_point", py::overload_cast<const AssociatedFeature2d &>(
                                &Group2d::AddPoint))
          .def("add_line", py::overload_cast<const AssociatedFeature2d &>(
                               &Group2d::AddLine))
          .def("add_points",
               py::overload_cast<const std::vector<AssociatedFeature2d> &>(
                   &Group2d::AddPoints))
          .def("add_lines",
               py::overload_cast<const std::vector<AssociatedFeature2d> &>(
                   &Group2d::AddLines))
          .def("add_point",
               py::overload_cast<const feature2D_t &>(&Group2d::AddPoint))
          .def("add_line",
               py::overload_cast<const feature2D_t &>(&Group2d::AddLine))
          .def("add_points",
               py::overload_cast<const std::vector<feature2D_t> &,
                                 const std::vector<double> &>(
                   &Group2d::AddPoints),
               py::arg("feats"), py::arg("weights") = std::vector<double>{})
          .def("add_lines",
               py::overload_cast<const std::vector<feature2D_t> &,
                                 const std::vector<double> &>(
                   &Group2d::AddLines),
               py::arg("feats"), py::arg("weights") = std::vector<double>{});
  MakeDataclass(PyGroup2d);
}

} // namespace limap
