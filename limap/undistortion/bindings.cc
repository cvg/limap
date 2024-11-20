#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "undistortion/undistort.h"
#include <colmap/sensor/bitmap.h>

namespace py = pybind11;

namespace limap {

void bind_undistortion(py::module &m) {
  using namespace undistortion;

  py::class_<colmap::Bitmap>(m, "COLMAP_Bitmap")
      .def(py::init<>())
      .def("Read", &colmap::Bitmap::Read, py::arg("imname"),
           py::arg("as_rgb") = true)
      .def("Write", &colmap::Bitmap::Write, py::arg("path"),
           py::arg("flag") = 0)
      .def("Width", &colmap::Bitmap::Width)
      .def("Height", &colmap::Bitmap::Height)
      .def("Channels", &colmap::Bitmap::Channels);

  m.def("_UndistortCamera", &UndistortCamera);
  m.def("_UndistortCameraView", &UndistortCameraView);
  m.def("_UndistortPoint", &UndistortPoint);
  m.def("_UndistortPoints", &UndistortPoints);
}

} // namespace limap
