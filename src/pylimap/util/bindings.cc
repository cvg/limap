#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "pylimap/helpers.h"

#include "limap/util/eigen_types.h"
#include "limap/util/kd_tree.h"

namespace limap {

void bind_kdtree(py::module &m) {
  py::classh<KDTree>(m, "KDTree")
      .def(py::init<>())
      .def(py::init<const std::vector<V3D> &>())
      .def(py::init<const M3D &>())
      .def("point_distance", &KDTree::point_distance)
      .def("query_nearest", &KDTree::query_nearest)
      .def("save", &KDTree::save)
      .def("load", &KDTree::load);
}

void bind_util(py::module &m) { bind_kdtree(m); }

} // namespace limap
