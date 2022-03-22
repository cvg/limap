#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "sfm/sfm_model.h"

namespace py = pybind11;

namespace limap {

void bind_sfm(py::module &m) {
    using namespace sfm;

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
        .def("addImage", &SfmModel::addImage)
        .def("addPoint", &SfmModel::addPoint)
        .def("ReadFromCOLMAP", &SfmModel::ReadFromCOLMAP)
        .def("GetImageNames", &SfmModel::GetImageNames)
        .def("GetMaxOverlappingImages", &SfmModel::GetMaxOverlappingImages)
        .def("GetMaxIoUImages", &SfmModel::GetMaxIoUImages)
        .def("GetMaxDiceCoeffImages", &SfmModel::GetMaxDiceCoeffImages)
        .def("ComputeNumPoints", &SfmModel::ComputeNumPoints)
        .def("ComputeRanges", &SfmModel::ComputeRanges);
}

} // namespace limap

