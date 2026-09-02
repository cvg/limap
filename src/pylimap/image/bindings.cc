#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

namespace limap {

void bind_groups(py::module &);

void bind_image(py::module &m) {
  pybind11::module_ _groups = m.def_submodule("_groups");
  bind_groups(_groups);
}

} // namespace limap
