#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "optimize/line_refinement/bindings.cc"
#include "optimize/line_bundle_adjustment/bindings.cc"

namespace py = pybind11;

namespace limap {

void bind_line_refinement(py::module& m);
void bind_line_bundle_adjustment(py::module& m);

void bind_optimize(py::module &m) {
    bind_line_refinement(m);
    bind_line_bundle_adjustment(m);
}

} // namespace limap


