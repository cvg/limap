#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "vplib/JLinkage/JLinkage.h"

namespace limap {

void bind_jlinkage(py::module &m) {
  using namespace vplib::JLinkage;
  py::class_<JLinkageConfig>(m, "JLinkageConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def(py::pickle(
          [](const JLinkageConfig &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return JLinkageConfig(dict);
          }))
      .def_readwrite("min_length", &JLinkageConfig::min_length)
      .def_readwrite("inlier_threshold", &JLinkageConfig::inlier_threshold)
      .def_readwrite("min_num_supports", &JLinkageConfig::min_num_supports)
      .def_readwrite("th_perp_supports", &JLinkageConfig::th_perp_supports);

  py::class_<JLinkage>(m, "JLinkage")
      .def(py::init<>())
      .def(py::init<const JLinkageConfig &>())
      .def(py::init<py::dict>())
      .def(py::pickle(
          [](const JLinkage &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return JLinkage(dict);
          }))
      .def("ComputeVPLabels", &JLinkage::ComputeVPLabels)
      .def("AssociateVPs", &JLinkage::AssociateVPs)
      .def("AssociateVPsParallel", &JLinkage::AssociateVPsParallel);
}

} // namespace limap
