#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <thirdparty/pycolmap/pybind11_extension.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "limap/scene/colmap_mvs_model.h"

namespace py = pybind11;

namespace limap {

void BindCOLMAPMVSModel(py::module &m) {
  using namespace scene;

  py::classh<colmap::mvs::Image>(m, "COLMAPMVSImage", py::module_local())
      .def(py::init<>())
      .def(py::init(&CreateCOLMAPMVSImage))
      .def("get_R", &colmap::mvs::Image::GetR)
      .def("get_T", &colmap::mvs::Image::GetT)
      .def("get_K", &colmap::mvs::Image::GetK)
      .def("get_P", &colmap::mvs::Image::GetP)
      .def("get_inv_P", &colmap::mvs::Image::GetInvP);

  py::classh<COLMAPMVSModel>(m, "COLMAPMVSModel", py::module_local())
      .def(py::init<>())
      .def("add_image", &COLMAPMVSModel::AddImage, py::arg("image"),
           py::arg("img_id") = -1)
      .def("add_point", &COLMAPMVSModel::AddPoint, py::arg("x"), py::arg("y"),
           py::arg("z"), py::arg("image_ids"))
      .def("read_from_colmap", &COLMAPMVSModel::ReadFromCOLMAP, py::arg("path"),
           py::arg("sparse_path"), py::arg("image_path"))
      .def("get_image_names", &COLMAPMVSModel::GetImageNames)
      .def("get_max_overlap_images", &COLMAPMVSModel::GetMaxOverlapImages,
           py::arg("num_images"), py::arg("min_tri_angle"))
      .def("get_max_iou_images", &COLMAPMVSModel::GetMaxIoUImages,
           py::arg("num_images"), py::arg("min_tri_angle"))
      .def("get_max_dice_coeff_images", &COLMAPMVSModel::GetMaxDiceCoeffImages,
           py::arg("num_images"), py::arg("min_tri_angle"))
      .def("compute_ranges", &COLMAPMVSModel::ComputeRanges,
           py::arg("range_robust"), py::arg("kstretch"));
}

} // namespace limap
