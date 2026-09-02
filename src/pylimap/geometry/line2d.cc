#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/line2d.h"

namespace py = pybind11;

namespace limap {

namespace {
static void CheckSeg2x2(const Eigen::Ref<const Eigen::MatrixXd> &seg) {
  if (seg.rows() != 2 || seg.cols() != 2) {
    throw std::invalid_argument("Expected 2x2 matrix [[x0,y0],[x1,y1]]");
  }
}
} // namespace

void BindLine2d(py::module &m) {
  py::classh<Line2d>(m, "Line2d")
      // Default constructor (creates an uninitialized line)
      .def(py::init<>())

      // Construct from a 2x2 matrix: row 0 = start (x,y), row 1 = end (x,y)
      .def(py::init([](const Eigen::MatrixXd &seg2d) {
             CheckSeg2x2(seg2d);
             return Line2d(seg2d);
           }),
           py::arg("seg2d"))

      // Construct from two endpoints
      .def(py::init<V2D, V2D>(), py::arg("start"), py::arg("end"))

      // Left (first) endpoint of the line segment
      .def_readwrite("start", &Line2d::start)

      // Right (second) endpoint of the line segment
      .def_readwrite("end", &Line2d::end)

      // Optional score associated with this segment (default: -1)
      .def_readwrite("score", &Line2d::score)

      // ID of the corresponding 3D line (if any)
      .def_readwrite("line3D_id", &Line2d::line3D_id)

      // @brief Check if this 2D line has a corresponding 3D line
      .def("has_line3D", &Line2d::HasLine3D)

      // @brief Euclidean length of the segment
      .def("length", &Line2d::Length)

      // @brief Midpoint of the segment
      .def("midpoint", &Line2d::Midpoint)

      // @brief Unit direction vector from start to end
      .def("direction", &Line2d::Direction)

      // @brief Unit perpendicular direction (+90° rotation)
      .def("perp_direction", &Line2d::PerpDirection)

      // @brief Homogeneous line coordinates (a, b, c) satisfying ax + by + c =
      // 0
      .def("coords", &Line2d::Coords)

      /**
       * @brief Orthogonal projection of a point onto the segment.
       *
       * Projects the given point p onto the infinite line, then clamps to
       * the segment endpoints.
       * @param p Query point.
       * @return Closest point on the segment to p.
       */
      .def("point_projection", &Line2d::PointProjection, py::arg("p"))

      /**
       * @brief Euclidean distance from a point to the segment.
       * @param p Query point.
       * @return Distance between p and its projection on the segment.
       */
      .def("point_distance", &Line2d::PointDistance, py::arg("p"))

      // @brief Represent the segment as a 2×2 matrix [start; end]
      .def("as_array", &Line2d::AsArray)

      // Pickle support
      .def(py::pickle(
          [](const Line2d &l) {
            return py::make_tuple(l.start, l.end, l.score, l.line3D_id);
          },
          [](py::tuple t) {
            if (t.size() != 4) {
              throw std::runtime_error("Invalid Line2d pickle state");
            }
            Line2d l(t[0].cast<V2D>(), t[1].cast<V2D>());
            l.score = t[2].cast<double>();
            l.line3D_id = t[3].cast<line3D_t>();
            return l;
          }));

  // Vectorized builder from a matrix of stacked segments
  m.def("get_line2d_vector_from_array", &GetLine2dVectorFromArray,
        py::arg("segs2d"));
}

} // namespace limap
