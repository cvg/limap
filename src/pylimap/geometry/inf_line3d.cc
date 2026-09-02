#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/inf_line3d.h"

namespace py = pybind11;

namespace limap {

void BindInfiniteLine3d(py::module &m) {
  py::classh<InfiniteLine3d>(m, "InfiniteLine3d")
      .def(py::init<>())
      .def(py::init<const V3D &, const V3D &, bool>(), py::arg("a"),
           py::arg("b"), py::arg("use_normal") = true)
      .def(py::init<const Line3d &>(), py::arg("line"))
      .def("point_projection", &InfiniteLine3d::PointProjection, py::arg("q"))
      .def("point_distance", &InfiniteLine3d::PointDistance, py::arg("q"))
      .def("projection", &InfiniteLine3d::Projection, py::arg("image"))
      .def("unprojection", &InfiniteLine3d::Unprojection, py::arg("p2d"),
           py::arg("image"))
      .def("project_from_infinite_line",
           &InfiniteLine3d::ProjectFromInfiniteLine, py::arg("line"))
      .def("project_to_infinite_line", &InfiniteLine3d::ProjectToInfiniteLine,
           py::arg("line"))
      .def("point", &InfiniteLine3d::Point)
      .def("direction", &InfiniteLine3d::Direction)
      .def("matrix", &InfiniteLine3d::Matrix)
      .def_readwrite("d", &InfiniteLine3d::d)
      .def_readwrite("m", &InfiniteLine3d::m);

  m.def("get_line_segment_from_infinite_line3d",
        py::overload_cast<const InfiniteLine3d &,
                          const std::vector<colmap::Image> &,
                          const std::vector<Line2d> &, const int>(
            &GetLineSegmentFromInfiniteLine3d),
        py::arg("inf_line"), py::arg("images"), py::arg("line2ds"),
        py::arg("num_outliers") = 2);

  m.def("get_line_segment_from_infinite_line3d",
        py::overload_cast<const InfiniteLine3d &, const std::vector<Line3d> &,
                          const int>(&GetLineSegmentFromInfiniteLine3d),
        py::arg("inf_line"), py::arg("line3ds"), py::arg("num_outliers") = 2);
}

} // namespace limap
