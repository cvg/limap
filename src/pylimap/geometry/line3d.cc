#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/line3d.h"

namespace py = pybind11;

namespace limap {

namespace {
static void CheckSegShape(const Eigen::Ref<const Eigen::MatrixXd> &seg) {
  if (seg.rows() != 2 || seg.cols() != 3) {
    throw std::invalid_argument("Expected 2x3 matrix [[x0,y0,z0],[x1,y1,z1]]");
  }
}
} // namespace

void BindLine3d(py::module &m) {
  py::classh<Line3d>(m, "Line3d")
      // Default constructor
      .def(py::init<>())

      // Construct from a 2x3 matrix: row 0 = start (x,y,z), row 1 = end
      // (x,y,z)
      .def(py::init([](const Eigen::MatrixXd &seg3d) {
             CheckSegShape(seg3d);
             return Line3d(seg3d);
           }),
           py::arg("seg3d"))

      // Construct from endpoints + optional metadata
      .def(py::init<V3D, V3D, double>(), py::arg("start"), py::arg("end"),
           py::arg("uncertainty") = -1.0)

      // Endpoints of the 3D segment
      .def_readwrite("start", &Line3d::start)
      .def_readwrite("end", &Line3d::end)

      // Associated feature track (from COLMAP)
      .def_readwrite("track", &Line3d::track) // optional if bound

      // Optional per-line uncertainty (default: -1)
      .def_readwrite("uncertainty", &Line3d::uncertainty)

      // Set the uncertainty value
      .def("set_uncertainty", &Line3d::SetUncertainty, py::arg("val"))

      // Euclidean length of the segment
      .def("length", &Line3d::Length)

      // Midpoint of the segment
      .def("midpoint", &Line3d::Midpoint)

      // Unit direction vector from start to end
      .def("direction", &Line3d::Direction)

      // Orthogonal projection of a 3D point onto the segment
      .def("point_projection", &Line3d::PointProjection, py::arg("p"))

      // Euclidean distance from a 3D point to the segment
      .def("point_distance", &Line3d::PointDistance, py::arg("p"))

      // Represent as a 2×3 matrix [start; end]
      .def("as_array", &Line3d::AsArray)

      // Project this 3D segment into the given camera view
      .def("projection", &Line3d::Projection, py::arg("image"))

      /**
       * @brief Sensitivity of this segment in a given view (degrees).
       * 0°   → best (line direction ⟂ viewing ray),
       * 90°  → worst / collapsing (line direction ∥ viewing ray).
       */
      .def("sensitivity", &Line3d::Sensitivity, py::arg("image"))

      /**
       * @brief Estimate uncertainty based on depth and 2D variance model.
       * @param view  Camera view containing pose/camera model.
       * @param var2d Variance of 2D detection (default 5.0).
       */
      .def("compute_uncertainty", &Line3d::ComputeUncertainty, py::arg("image"),
           py::arg("var2d") = 5.0)

      // Pickle support
      .def(py::pickle(
          [](const Line3d &l) {
            return py::make_tuple(l.start, l.end, l.uncertainty, l.track);
          },
          [](py::tuple t) {
            if (t.size() != 4) {
              throw std::runtime_error("Invalid Line3d pickle state");
            }
            Line3d l(t[0].cast<V3D>(), t[1].cast<V3D>(), t[2].cast<double>());
            l.track = t[3].cast<colmap::Track>();
            return l;
          }));

  // Get vector<Line3d> from vector of 2x3 matrices
  m.def(
      "get_line3d_vector_from_array",
      [](const std::vector<Eigen::MatrixXd> &segs3d) {
        for (const auto &M : segs3d)
          CheckSegShape(M);
        return GetLine3dVectorFromArray(segs3d);
      },
      py::arg("segs3d"));

  // Project 3D line into 2D image
  m.def("projection_line3d", &ProjectionLine3d, py::arg("line3d"),
        py::arg("image"));

  // Unproject 2D line into 3D with depth pair
  m.def("unprojection_line2d", &UnprojectionLine2d, py::arg("line2d"),
        py::arg("image"), py::arg("depths"));
}

} // namespace limap
