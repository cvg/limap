#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "optimize/global_pl_association/bindings.cc"
#include "optimize/hybrid_bundle_adjustment/bindings.cc"
#include "optimize/hybrid_localization/bindings.cc"
#include "optimize/line_refinement/bindings.cc"

namespace py = pybind11;

namespace limap {

void bind_line_refinement(py::module &m);
void bind_hybrid_bundle_adjustment(py::module &m);
void bind_global_pl_association(py::module &m);
void bind_hybrid_localization(py::module &m);

void bind_optimize(py::module &m) {
  bind_line_refinement(m);
  bind_hybrid_bundle_adjustment(m);
  bind_global_pl_association(m);
  bind_hybrid_localization(m);
}

} // namespace limap
