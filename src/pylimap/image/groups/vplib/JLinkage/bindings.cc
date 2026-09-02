#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <vector>

#include <thirdparty/pycolmap/helpers.h>

#include "limap/image/groups/vplib/JLinkage/JLinkage.h"

namespace limap {

void bind_jlinkage(py::module &m) {
  using namespace image::groups::vplib::JLinkage;

  py::classh<JLinkageOptions> PyJLinkageOptions(m, "JLinkageOptions");
  PyJLinkageOptions.def(py::init<>())
      .def_readwrite("base_options", &JLinkageOptions::base_options)
      .def_readwrite("th_perp_supports", &JLinkageOptions::th_perp_supports);
  MakeDataclass(PyJLinkageOptions);

  py::classh<JLinkage> PyJLinkage(m, "JLinkage");
  PyJLinkage.def(py::init<>())
      .def(py::init<const JLinkageOptions &>())
      .def("compute_vp_labels", &JLinkage::ComputeVPLabels)
      .def("associate_vps", &JLinkage::AssociateVPs)
      .def("associate_vps_parallel", &JLinkage::AssociateVPsParallel);
  MakeDataclass(PyJLinkage);
}

} // namespace limap
