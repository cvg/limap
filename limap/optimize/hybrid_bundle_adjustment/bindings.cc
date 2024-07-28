#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment.h"
#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment_config.h"

namespace py = pybind11;

namespace limap {

void bind_hybrid_bundle_adjustment(py::module &m) {
  using namespace optimize::hybrid_bundle_adjustment;

  py::class_<HybridBAConfig>(m, "HybridBAConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def("set_constant_camera", &HybridBAConfig::set_constant_camera)
      .def_readwrite("min_num_images", &HybridBAConfig::min_num_images)
      .def_readwrite("solver_options", &HybridBAConfig::solver_options)
      .def_readwrite("point_geometric_loss_function",
                     &HybridBAConfig::point_geometric_loss_function)
      .def_readwrite("line_geometric_loss_function",
                     &HybridBAConfig::line_geometric_loss_function)
      .def_readwrite("geometric_alpha", &HybridBAConfig::geometric_alpha)
      .def_readwrite("lw_point", &HybridBAConfig::lw_point)
      .def_readwrite("print_summary", &HybridBAConfig::print_summary)
      .def_readwrite("constant_intrinsics",
                     &HybridBAConfig::constant_intrinsics)
      .def_readwrite("constant_principal_point",
                     &HybridBAConfig::constant_principal_point)
      .def_readwrite("constant_pose", &HybridBAConfig::constant_pose)
      .def_readwrite("constant_point", &HybridBAConfig::constant_point)
      .def_readwrite("constant_line", &HybridBAConfig::constant_line);

  py::class_<HybridBAEngine>(m, "HybridBAEngine")
      .def(py::init<>())
      .def(py::init<const HybridBAConfig &>())
      .def("InitImagecols", &HybridBAEngine::InitImagecols)
      .def("InitPointTracks", static_cast<void (HybridBAEngine::*)(
                                  const std::vector<PointTrack> &)>(
                                  &HybridBAEngine::InitPointTracks))
      .def("InitPointTracks", static_cast<void (HybridBAEngine::*)(
                                  const std::map<int, PointTrack> &)>(
                                  &HybridBAEngine::InitPointTracks))
      .def(
          "InitLineTracks",
          static_cast<void (HybridBAEngine::*)(const std::vector<LineTrack> &)>(
              &HybridBAEngine::InitLineTracks))
      .def("InitLineTracks", static_cast<void (HybridBAEngine::*)(
                                 const std::map<int, LineTrack> &)>(
                                 &HybridBAEngine::InitLineTracks))
      .def("SetUp", &HybridBAEngine::SetUp)
      .def("Solve", &HybridBAEngine::Solve)
      .def("GetOutputImagecols", &HybridBAEngine::GetOutputImagecols)
      .def("GetOutputPoints", &HybridBAEngine::GetOutputPoints)
      .def("GetOutputPointTracks", &HybridBAEngine::GetOutputPointTracks)
      .def("GetOutputLines", &HybridBAEngine::GetOutputLines,
           py::arg("num_outliers") = 2)
      .def("GetOutputLineTracks", &HybridBAEngine::GetOutputLineTracks,
           py::arg("num_outliers") = 2);
}

} // namespace limap
