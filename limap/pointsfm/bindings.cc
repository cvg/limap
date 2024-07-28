#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "pointsfm/sfm_model.h"

namespace py = pybind11;

namespace limap {

void bind_pointsfm(py::module &m) {
  using namespace pointsfm;

  // bind the colmap mvs image
  py::class_<colmap::mvs::Image>(m, "SfmImage")
      .def(py::init<>())
      .def(py::init(&CreateSfmImage))
      .def("GetR", &colmap::mvs::Image::GetR)
      .def("GetT", &colmap::mvs::Image::GetT)
      .def("GetK", &colmap::mvs::Image::GetK)
      .def("GetP", &colmap::mvs::Image::GetP)
      .def("GetInvP", &colmap::mvs::Image::GetInvP);

  // bind the new sfm model.
  py::class_<SfmModel>(m, "SfmModel")
      .def(py::init<>())
      .def("addImage", &SfmModel::addImage, py::arg("image"),
           py::arg("img_id") = -1)
      .def("addPoint", &SfmModel::addPoint)
      .def("ReadFromCOLMAP", &SfmModel::ReadFromCOLMAP)
      .def("GetImageNames", &SfmModel::GetImageNames)
      .def("ComputeNumPoints", &SfmModel::ComputeNumPoints)
      .def("ComputeSharedPoints", &SfmModel::ComputeSharedPoints)
      .def("GetMaxOverlapImages", &SfmModel::GetMaxOverlapImages)
      .def("GetMaxIoUImages", &SfmModel::GetMaxIoUImages)
      .def("GetMaxDiceCoeffImages", &SfmModel::GetMaxDiceCoeffImages)
      .def("ComputeNumPoints", &SfmModel::ComputeNumPoints)
      .def("ComputeRanges", &SfmModel::ComputeRanges);
}

} // namespace limap
