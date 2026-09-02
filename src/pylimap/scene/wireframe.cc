#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/wireframe.h"

namespace py = pybind11;

namespace limap {

void BindWireframe(py::module &m) {
  py::classh<WireframeConnection2d>(m, "WireframeConnection2d")
      .def(py::init<>())
      .def_readwrite("point_idx", &WireframeConnection2d::point_idx)
      .def_readwrite("line_idx", &WireframeConnection2d::line_idx)
      .def_readwrite("w", &WireframeConnection2d::w);

  py::classh<Wireframe2d>(m, "Wireframe2d")
      .def(py::init<>())
      .def("add_edge", py::overload_cast<point2D_t, line2D_t, double>(
                           &Wireframe2d::AddEdge))
      .def("add_edge", py::overload_cast<const WireframeConnection2d &>(
                           &Wireframe2d::AddEdge))
      .def("remove_edge",
           py::overload_cast<point2D_t, line2D_t>(&Wireframe2d::RemoveEdge))
      .def("remove_edge", py::overload_cast<const WireframeConnection2d &>(
                              &Wireframe2d::RemoveEdge))
      .def("clear", &Wireframe2d::Clear)
      .def("get_neighboring_line_ids", &Wireframe2d::GetNeighboringLineIds)
      .def("count_neighboring_lines", &Wireframe2d::CountNeighboringLines)
      .def("get_neighboring_point_ids", &Wireframe2d::GetNeighboringPointIds)
      .def("count_neighboring_points", &Wireframe2d::CountNeighboringPoints)
      .def("count_points", &Wireframe2d::CountPoints)
      .def("count_lines", &Wireframe2d::CountLines)
      .def("count_edges", &Wireframe2d::CountEdges)
      .def("get_edge_weight", &Wireframe2d::GetEdgeWeight)
      .def("get_edge", &Wireframe2d::GetEdge)
      .def("get_all_edges", &Wireframe2d::GetAllEdges)
      .def("__repr__", [](const Wireframe2d &s) {
        std::ostringstream oss;
        oss << "Wireframe2d(num_points=" << s.CountPoints()
            << ", num_lines=" << s.CountLines()
            << ", num_edges=" << s.CountEdges() << ")";
        return oss.str();
      });

  py::classh<WireframeConnection3d>(m, "WireframeConnection3d")
      .def(py::init<>())
      .def_readwrite("point_idx", &WireframeConnection3d::point_idx)
      .def_readwrite("line_idx", &WireframeConnection3d::line_idx)
      .def_readwrite("w", &WireframeConnection3d::w);

  py::classh<Wireframe3d>(m, "Wireframe3d")
      .def(py::init<>())
      .def("add_edge", py::overload_cast<point3D_t, line3D_t, double>(
                           &Wireframe3d::AddEdge))
      .def("add_edge", py::overload_cast<const WireframeConnection3d &>(
                           &Wireframe3d::AddEdge))
      .def("remove_edge",
           py::overload_cast<point3D_t, line3D_t>(&Wireframe3d::RemoveEdge))
      .def("remove_edge", py::overload_cast<const WireframeConnection3d &>(
                              &Wireframe3d::RemoveEdge))
      .def("clear", &Wireframe3d::Clear)
      .def("get_neighboring_line_ids", &Wireframe3d::GetNeighboringLineIds)
      .def("count_neighboring_lines", &Wireframe3d::CountNeighboringLines)
      .def("get_neighboring_point_ids", &Wireframe3d::GetNeighboringPointIds)
      .def("count_neighboring_points", &Wireframe3d::CountNeighboringPoints)
      .def("count_points", &Wireframe3d::CountPoints)
      .def("count_lines", &Wireframe3d::CountLines)
      .def("count_edges", &Wireframe3d::CountEdges)
      .def("get_edge_weight", &Wireframe3d::GetEdgeWeight)
      .def("get_edge", &Wireframe3d::GetEdge)
      .def("get_all_edges", &Wireframe3d::GetAllEdges)
      .def("__repr__", [](const Wireframe3d &s) {
        std::ostringstream oss;
        oss << "Wireframe3d(num_points=" << s.CountPoints()
            << ", num_lines=" << s.CountLines()
            << ", num_edges=" << s.CountEdges() << ")";
        return oss.str();
      });

  m.def("create_wireframe2d", &CreateWireframe2d, py::arg("points"),
        py::arg("lines"), py::arg("threshold") = 2.0);
}

} // namespace limap
