#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include <colmap/util/bitmap.h>
#include "undistortion/undistort.h"

namespace py = pybind11;

namespace limap {

void bind_undistortion(py::module &m) {
    using namespace undistortion;

    py::class_<colmap::Bitmap>(m, "COLMAP_Bitmap")
        .def(py::init<>())
        .def("Read", &colmap::Bitmap::Read)
        .def("Write", &colmap::Bitmap::Write)
        .def("Width", &colmap::Bitmap::Width)
        .def("Height", &colmap::Bitmap::Height)
        .def("Channels", &colmap::Bitmap::Channels);

    m.def("_Undistort", &Undistort);
}

} // namespace limap

