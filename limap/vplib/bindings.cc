#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "vplib/JLinkage/JLinkage.h"
#include "vplib/global_vptrack_constructor.h"
#include "vplib/vpbase.h"
#include "vplib/vptrack.h"

namespace limap {

void bind_vpdetector(py::module &m) {
  using namespace vplib;

  py::class_<VPResult>(m, "VPResult")
      .def(py::init<>())
      .def(py::init<const std::vector<int> &, const std::vector<V3D> &>())
      .def(py::init<const VPResult &>())
      .def(py::init<py::dict>())
      .def(py::pickle(
          [](const VPResult &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return VPResult(dict);
          }))
      .def_readonly("labels", &VPResult::labels)
      .def_readonly("vps", &VPResult::vps)
      .def("count_lines", &VPResult::count_lines)
      .def("count_vps", &VPResult::count_vps)
      .def("HasVP", &VPResult::HasVP)
      .def("GetVP", &VPResult::GetVP);
}

void bind_jlinkage(py::module &m);

void bind_vptrack(py::module &m) {
  using namespace vplib;

  py::class_<VP2d>(m, "VP2d")
      .def(py::init<>())
      .def(py::init<V3D, int>(), py::arg("p"), py::arg("point3D_id") = -1)
      .def("as_dict", &VP2d::as_dict)
      .def(py::pickle(
          [](const VP2d &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return VP2d(dict);
          }))
      .def_readwrite("p", &VP2d::p)
      .def_readwrite("point3D_id", &VP2d::point3D_id);

  py::class_<VPTrack>(m, "VPTrack")
      .def(py::init<>())
      .def(py::init<const V3D &, const std::vector<Node2d> &>())
      .def(py::init<const VPTrack &>())
      .def(py::pickle(
          [](const VPTrack &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return VPTrack(dict);
          }))
      .def_readwrite("direction", &VPTrack::direction)
      .def_readwrite("supports", &VPTrack::supports)
      .def("length", &VPTrack::length);

  m.def("MergeVPTracksByDirection", &MergeVPTracksByDirection,
        py::arg("tracks"), py::arg("th_angle_merge") = 1.0);
}

void bind_global_vptrack_constructor(py::module &m) {
  using namespace vplib;

  py::class_<GlobalVPTrackConstructorConfig>(m,
                                             "GlobalVPTrackConstructorConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("min_common_lines",
                     &GlobalVPTrackConstructorConfig::min_common_lines)
      .def_readwrite("th_angle_verify",
                     &GlobalVPTrackConstructorConfig::th_angle_verify)
      .def_readwrite("min_track_length",
                     &GlobalVPTrackConstructorConfig::min_track_length);

  py::class_<GlobalVPTrackConstructor>(m, "GlobalVPTrackConstructor")
      .def(py::init<>())
      .def(py::init<const GlobalVPTrackConstructorConfig &>())
      .def(py::init<py::dict>())
      .def("Init", &GlobalVPTrackConstructor::Init)
      .def("ClusterLineTracks", &GlobalVPTrackConstructor::ClusterLineTracks);
}

void bind_vplib(py::module &m) {
  bind_vpdetector(m);
  bind_jlinkage(m);
  bind_vptrack(m);
  bind_global_vptrack_constructor(m);
}

} // namespace limap
