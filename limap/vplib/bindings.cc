#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "vplib/vpbase.h"
#include "vplib/jlinkage.h"

namespace limap {

void bind_vplib(py::module &m) {
    using namespace vplib;

    py::class_<VPResult>(m, "VPResult")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::vector<V3D>&>())
        .def(py::init<const VPResult&>())
        .def_readonly("labels", &VPResult::labels)
        .def_readonly("vps", &VPResult::vps)
        .def("count_lines", &VPResult::count_lines)
        .def("count_vps", &VPResult::count_vps)
        .def("HasVP", &VPResult::HasVP)
        .def("GetVP", &VPResult::GetVP);
    
    py::class_<JLinkageConfig>(m, "JLinkageConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("min_length", &JLinkageConfig::min_length)
        .def_readwrite("inlier_threshold", &JLinkageConfig::inlier_threshold)
        .def_readwrite("min_num_supports", &JLinkageConfig::min_num_supports);

    py::class_<JLinkage>(m, "JLinkage")
        .def(py::init<>())
        .def(py::init<const JLinkageConfig&>())
        .def(py::init<py::dict>())
        .def("ComputeVPLabels", &JLinkage::ComputeVPLabels)
        .def("AssociateVPs", &JLinkage::AssociateVPs)
        .def("AssociateVPsParallel", &JLinkage::AssociateVPsParallel); 
}

} // namespace limap

