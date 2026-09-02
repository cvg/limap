#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "pylimap/helpers.h"

namespace limap {

void BindStructureObservationManager(py::module &m);
void BindFittedLineMerging(py::module &m);
void BindGlobalLineTriangulation(py::module &m);
void BindGlobalGroupTriangulation(py::module &m);
void BindGlobalStructureTriangulation(py::module &m);
void BindGroupVerification(py::module &m);
void BindIncrementalLineTriangulator(py::module &m);
void BindIncrementalGroupTriangulator(py::module &m);
void BindIncrementalStructureTriangulator(py::module &m);
void BindStructureIncrementalMapper(py::module &m);
void BindStructureIncrementalPipeline(py::module &m);

void bind_sfm(py::module &m) {
  BindFittedLineMerging(m);
  BindGlobalLineTriangulation(m);
  BindGlobalGroupTriangulation(m);
  BindGlobalStructureTriangulation(m);
  BindGroupVerification(m);
  BindIncrementalLineTriangulator(m);
  BindIncrementalGroupTriangulator(m);
  BindIncrementalStructureTriangulator(m);
  BindStructureObservationManager(m);
  BindStructureIncrementalMapper(m);
  BindStructureIncrementalPipeline(m);
}

} // namespace limap
