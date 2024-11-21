#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include <Eigen/Core>
#include <vector>

namespace py = pybind11;

#include "triangulation/functions.h"
#include "triangulation/global_line_triangulator.h"

namespace limap {

void bind_functions(py::module &m) {
  using namespace triangulation;

  m.def("get_normal_direction", &getNormalDirection);
  m.def("get_direction_from_VP", &getDirectionFromVP);
  m.def("compute_essential_matrix", &compute_essential_matrix);
  m.def("compute_fundamental_matrix", &compute_fundamental_matrix);
  m.def("compute_epipolar_IoU", &compute_epipolar_IoU);
  m.def("triangulate_point", &triangulate_point);
  m.def("triangulate_line_by_endpoints", &triangulate_line_by_endpoints);
  m.def("triangulate_line", &triangulate_line);
  m.def("triangulate_line_with_one_point", &triangulate_line_with_one_point);
  m.def("triangulate_line_with_direction", &triangulate_line_with_direction);
}

void bind_triangulator(py::module &m) {
  using namespace triangulation;

#define REGISTER_TRIANGULATOR_CONFIG(TriangulatorConfig)                       \
  .def(py::init<>())                                                           \
      .def(py::init<py::dict>())                                               \
      .def_readwrite("debug_mode", &TriangulatorConfig::debug_mode)            \
      .def_readwrite("add_halfpix", &TriangulatorConfig::add_halfpix)          \
      .def_readwrite("use_vp", &TriangulatorConfig::use_vp)                    \
      .def_readwrite("use_endpoints_triangulation",                            \
                     &TriangulatorConfig::use_endpoints_triangulation)         \
      .def_readwrite("disable_many_points_triangulation",                      \
                     &TriangulatorConfig::disable_many_points_triangulation)   \
      .def_readwrite("disable_one_point_triangulation",                        \
                     &TriangulatorConfig::disable_one_point_triangulation)     \
      .def_readwrite("disable_algebraic_triangulation",                        \
                     &TriangulatorConfig::disable_algebraic_triangulation)     \
      .def_readwrite("disable_vp_triangulation",                               \
                     &TriangulatorConfig::disable_vp_triangulation)            \
      .def_readwrite("min_length_2d", &TriangulatorConfig::min_length_2d)      \
      .def_readwrite("line_tri_angle_threshold",                               \
                     &TriangulatorConfig::line_tri_angle_threshold)            \
      .def_readwrite("IoU_threshold", &TriangulatorConfig::IoU_threshold)      \
      .def_readwrite("var2d", &TriangulatorConfig::var2d)

  py::class_<GlobalLineTriangulatorConfig>(m, "GlobalLineTriangulatorConfig")
      REGISTER_TRIANGULATOR_CONFIG(GlobalLineTriangulatorConfig)
          .def_readwrite("fullscore_th",
                         &GlobalLineTriangulatorConfig::fullscore_th)
          .def_readwrite("max_valid_conns",
                         &GlobalLineTriangulatorConfig::max_valid_conns)
          .def_readwrite("min_num_outer_edges",
                         &GlobalLineTriangulatorConfig::min_num_outer_edges)
          .def_readwrite("merging_strategy",
                         &GlobalLineTriangulatorConfig::merging_strategy)
          .def_readwrite("num_outliers_aggregator",
                         &GlobalLineTriangulatorConfig::num_outliers_aggregator)
          .def_readwrite("linker2d_config",
                         &GlobalLineTriangulatorConfig::linker2d_config)
          .def_readwrite("linker3d_config",
                         &GlobalLineTriangulatorConfig::linker3d_config);

#undef REGISTER_TRIANGULATOR_CONFIG

#define REGISTER_TRIANGULATOR(Triangulator)                                    \
  .def(py::init<>())                                                           \
      .def(py::init<py::dict>())                                               \
      .def("Init", &Triangulator::Init)                                        \
      .def("InitVPResults", &Triangulator::InitVPResults)                      \
      .def("TriangulateImage", &Triangulator::TriangulateImage)                \
      .def("TriangulateImageExhaustiveMatch",                                  \
           &Triangulator::TriangulateImageExhaustiveMatch)                     \
      .def("SetBipartites2d", &Triangulator::SetBipartites2d)                  \
      .def("SetSfMPoints", &Triangulator::SetSfMPoints)                        \
      .def("ComputeLineTracks", &Triangulator::ComputeLineTracks)              \
      .def("GetVPResult", &Triangulator::GetVPResult)                          \
      .def("GetVPResults", &Triangulator::GetVPResults)                        \
      .def("CountImages", &Triangulator::CountImages)                          \
      .def("CountLines", &Triangulator::CountLines)                            \
      .def("GetTracks", &Triangulator::GetTracks)                              \
      .def("SetRanges", &Triangulator::SetRanges)                              \
      .def("UnsetRanges", &Triangulator::UnsetRanges)

  py::class_<GlobalLineTriangulator>(
      m, "GlobalLineTriangulator") REGISTER_TRIANGULATOR(GlobalLineTriangulator)
      .def(py::init<const GlobalLineTriangulatorConfig &>())
      .def("GetLinker", &GlobalLineTriangulator::GetLinker)
      .def("CountAllTris", &GlobalLineTriangulator::CountAllTris)
      .def("GetScoredTrisNode", &GlobalLineTriangulator::GetScoredTrisNode)
      .def("GetValidScoredTrisNode",
           &GlobalLineTriangulator::GetValidScoredTrisNode)
      .def("GetValidScoredTrisNodeSet",
           &GlobalLineTriangulator::GetValidScoredTrisNodeSet)
      .def("CountAllValidTris", &GlobalLineTriangulator::CountAllValidTris)
      .def("GetAllValidTris", &GlobalLineTriangulator::GetAllValidTris)
      .def("GetValidTrisImage", &GlobalLineTriangulator::GetValidTrisImage)
      .def("GetValidTrisNode", &GlobalLineTriangulator::GetValidTrisNode)
      .def("GetValidTrisNodeSet", &GlobalLineTriangulator::GetValidTrisNodeSet)
      .def("GetAllBestTris", &GlobalLineTriangulator::GetAllBestTris)
      .def("GetAllValidBestTris", &GlobalLineTriangulator::GetAllValidBestTris)
      .def("GetBestTrisImage", &GlobalLineTriangulator::GetBestTrisImage)
      .def("GetBestTriNode", &GlobalLineTriangulator::GetBestTriNode)
      .def("GetBestScoredTriNode",
           &GlobalLineTriangulator::GetBestScoredTriNode)
      .def("GetSurvivedLinesImage",
           &GlobalLineTriangulator::GetSurvivedLinesImage);

#undef REGISTER_TRIANGULATOR
}

void bind_triangulation(py::module &m) {
  bind_functions(m);
  bind_triangulator(m);
}

} // namespace limap
