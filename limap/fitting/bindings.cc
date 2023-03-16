#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "fitting/line3d_estimator.h"

namespace py = pybind11;

namespace limap {

void bind_fitting(py::module& m) {
    using namespace fitting;

    m.def("Fit3DPoints", &Fit3DPoints);
}

} // namespace limap

