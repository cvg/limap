#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/image/groups/vplib/JLinkage/JLinkage.h"
#include "limap/image/groups/vplib/base_vp_detector.h"
#include "limap/image/groups/vplib/convert_to_groups.h"

namespace limap {

void bind_vpbase(py::module &m) {
  using namespace image::groups::vplib;

  py::classh<BaseVPDetectorOptions> PyBaseVPDetectorOptions(
      m, "BaseVPDetectorOptions");
  PyBaseVPDetectorOptions.def(py::init<>())
      .def_readwrite("min_length", &BaseVPDetectorOptions::min_length)
      .def_readwrite("inlier_threshold",
                     &BaseVPDetectorOptions::inlier_threshold)
      .def_readwrite("min_num_supports",
                     &BaseVPDetectorOptions::min_num_supports)
      .def_readwrite("n_jobs", &BaseVPDetectorOptions::n_jobs);
  MakeDataclass(PyBaseVPDetectorOptions);

  py::classh<VPResult> PyVPResult(m, "VPResult");
  PyVPResult.def(py::init<>())
      .def(py::init<const std::vector<int> &, const std::vector<V3D> &>())
      .def(py::init<const VPResult &>())
      .def_readwrite("labels", &VPResult::labels)
      .def_readwrite("vps", &VPResult::vps)
      .def("count_lines", &VPResult::CountLines)
      .def("count_vps", &VPResult::CountVPs)
      .def("get_vp_label", &VPResult::GetVPLabel, py::arg("line_id"))
      .def("get_vp_params", &VPResult::GetVPParams, py::arg("vp_id"))
      .def("has_vp", &VPResult::HasVP, py::arg("line_id"))
      .def("get_vp", &VPResult::GetVP, py::arg("line_id"));
  MakeDataclass(PyVPResult);
}

void bind_jlinkage(py::module &m);

void bind_vplib(py::module &m) {
  bind_vpbase(m);
  bind_jlinkage(m);

  m.def("convert_vpresult_to_groups2d",
        &image::groups::vplib::ConvertVPResultToGroups2d, py::arg("vpresult"));
  m.def("convert_vpresults_to_groups2d",
        &image::groups::vplib::ConvertVPResultsToGroups2d,
        py::arg("vpresults"));
}

} // namespace limap
