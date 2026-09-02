#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "pylimap/helpers.h"

namespace limap {

void BindGroups(py::module &);
void BindLine2d(py::module &);
void BindLine3d(py::module &);
void BindInfiniteLine2d(py::module &);
void BindInfiniteLine3d(py::module &);
void BindMinimalInfiniteLine3d(py::module &);
void BindLineMetrics(py::module &);
void BindLineLinker(py::module &);

void bind_geometry(py::module &m) {
  BindGroups(m);
  BindLine2d(m);
  BindLine3d(m);
  BindInfiniteLine2d(m);
  BindInfiniteLine3d(m);
  BindMinimalInfiniteLine3d(m);
  BindLineMetrics(m);
  BindLineLinker(m);
}

} // namespace limap
