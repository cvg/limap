#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/structure2d.h"

namespace py = pybind11;

namespace limap {

void BindStructure2d(py::module &m) {
  py::classh<Structure2d>(m, "Structure2d")
      .def(py::init<>())
      .def(py::init<const std::vector<Line2d> &, const std::vector<Group2d> &,
                    const Wireframe2d &, point2D_t>(),
           py::arg("lines"), py::arg("groups"), py::arg("wireframe"),
           py::arg("num_points") = 0)
      .def_property_readonly("lines",
                             (std::vector<Line2d> & (Structure2d::*)()) &
                                 Structure2d::Lines,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("groups",
                             (std::vector<Group2d> & (Structure2d::*)()) &
                                 Structure2d::Groups,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("wireframe",
                             (Wireframe2d & (Structure2d::*)()) &
                                 Structure2d::Wireframe,
                             py::return_value_policy::reference_internal)
      .def("num_points", &Structure2d::NumPoints)
      .def("set_lines", &Structure2d::SetLines)
      .def("add_group", &Structure2d::AddGroup)
      .def("add_groups", &Structure2d::AddGroups)
      .def("set_groups", &Structure2d::SetGroups)
      .def("set_wireframe", &Structure2d::SetWireframe)
      .def("set_num_points", &Structure2d::SetNumPoints)
      .def("line", (Line2d & (Structure2d::*)(size_t)) & Structure2d::Line,
           py::return_value_policy::reference_internal)
      .def("num_lines", &Structure2d::NumLines)
      .def("group", (Group2d & (Structure2d::*)(size_t)) & Structure2d::Group,
           py::return_value_policy::reference_internal)
      .def("num_groups", &Structure2d::NumGroups)
      .def("__repr__", [](const Structure2d &s) {
        std::ostringstream oss;
        oss << "Structure2d(num_points=" << s.NumPoints()
            << ", num_lines=" << s.NumLines()
            << ", num_groups=" << s.NumGroups()
            << ", wf_edges=" << s.Wireframe().CountEdges() << ")";
        return oss.str();
      });
}

} // namespace limap
