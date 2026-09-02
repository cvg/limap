#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/minimal_inf_line3d.h"

namespace py = pybind11;

namespace limap {

void BindMinimalInfiniteLine3d(py::module &m) {
  py::classh<MinimalInfiniteLine3d>(m, "MinimalInfiniteLine3d")
      .def(py::init<>())
      .def(py::init<const Line3d &>(), py::arg("line"))
      .def(py::init<const InfiniteLine3d &>(), py::arg("inf_line"))
      .def(py::init<const std::vector<double> &>(), py::arg("values"))
      .def("get_infinite_line", &MinimalInfiniteLine3d::GetInfiniteLine)
      .def_readwrite("data", &MinimalInfiniteLine3d::data)
      .def_property(
          "uvec",
          [](const MinimalInfiniteLine3d &self) {
            Eigen::Vector4d v;
            v << self.data[0], self.data[1], self.data[2], self.data[3]; // xyzw
            return v;
          },
          [](MinimalInfiniteLine3d &self, const Eigen::Vector4d &v) {
            self.data[0] = v[0];
            self.data[1] = v[1];
            self.data[2] = v[2];
            self.data[3] = v[3];
          })
      .def_property(
          "wvec",
          [](const MinimalInfiniteLine3d &self) {
            Eigen::Vector2d v;
            v << self.data[4], self.data[5];
            return v;
          },
          [](MinimalInfiniteLine3d &self, const Eigen::Vector2d &v) {
            self.data[4] = v[0];
            self.data[5] = v[1];
          });
}

} // namespace limap
