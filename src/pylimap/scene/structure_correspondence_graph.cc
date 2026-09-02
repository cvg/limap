#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/structure_correspondence_graph.h"

namespace py = pybind11;

namespace limap {

void BindStructureCorrespondenceGraph(py::module &m) {
  using SCG = StructureCorrespondenceGraph;
  py::classh<SCG>(m, "StructureCorrespondenceGraph")
      .def(py::init<>())
      .def("add_line_matches", &SCG::AddLineMatches, py::arg("image_id1"),
           py::arg("image_id2"), py::arg("matches"))
      .def("add_group_matches", &SCG::AddGroupMatches, py::arg("image_id1"),
           py::arg("image_id2"), py::arg("matches"))
      .def_property_readonly("line_graph",
                             py::overload_cast<>(&SCG::LineGraph, py::const_),
                             py::return_value_policy::reference_internal)
      .def_property_readonly("group_graph",
                             py::overload_cast<>(&SCG::GroupGraph, py::const_),
                             py::return_value_policy::reference_internal);
}

} // namespace limap
