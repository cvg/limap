#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/inf_line2d.h"

namespace py = pybind11;

namespace limap {

void BindInfiniteLine2d(py::module &m) {

  py::classh<InfiniteLine2d>(m, "InfiniteLine2d")
      .def(py::init<>())
      .def(py::init<const V3D &>(), py::arg("coords"))
      .def(py::init<const V2D &, const V2D &>(), py::arg("point"),
           py::arg("direction"))
      .def(py::init<const Line2d &>(), py::arg("line"))
      .def("point_projection", &InfiniteLine2d::PointProjection, py::arg("q"))
      .def("point_distance", &InfiniteLine2d::PointDistance, py::arg("q"))
      .def("point", &InfiniteLine2d::Point)
      .def("direction", &InfiniteLine2d::Direction)
      .def_readwrite("coords", &InfiniteLine2d::coords);

  m.def("intersect_infinite_line2d", &IntersectInfiniteLine2d, py::arg("l1"),
        py::arg("l2"));
}

} // namespace limap
