#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

namespace py = pybind11;

#include "structures/pl_bipartite.h"
#include "structures/vpline_bipartite.h"

namespace limap {

void bind_bipartites(py::module &m) {
  using namespace structures;

#define REGISTER_JUNCTION(NAME, SUBCLASS)                                      \
  .def(py::init<>())                                                           \
      .def(py::init<const SUBCLASS &, const std::vector<int> &>(),             \
           py::arg("p"), py::arg("line_ids") = std::vector<int>())             \
      .def_readonly("p", &NAME::p)                                             \
      .def_readonly("line_ids", &NAME::line_ids)                               \
      .def("degree", &NAME::degree)

  py::class_<Junction2d>(m, "Junction2d")
      REGISTER_JUNCTION(Junction2d, Point2d);

  py::class_<Junction3d>(m, "Junction3d")
      REGISTER_JUNCTION(Junction3d, PointTrack);

  py::class_<VP_Junction2d>(m, "VP_Junction2d")
      REGISTER_JUNCTION(VP_Junction2d, V3D);

  py::class_<VP_Junction3d>(m, "VP_Junction3d")
      REGISTER_JUNCTION(VP_Junction3d, vplib::VPTrack);

#undef REGISTER_JUNCTION

  py::class_<PL_Bipartite2dConfig>(m, "PL_Bipartite2dConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("threshold_intersection",
                     &PL_Bipartite2dConfig::threshold_intersection)
      .def_readwrite("threshold_merge_junctions",
                     &PL_Bipartite2dConfig::threshold_merge_junctions)
      .def_readwrite("threshold_keypoints",
                     &PL_Bipartite2dConfig::threshold_keypoints);

#define REGISTER_BIPARTITE_BASE(NAME)                                          \
  .def(py::init<>())                                                           \
      .def(py::init<const NAME &>())                                           \
      .def(py::init<py::dict>())                                               \
      .def(py::pickle([](const NAME &input) { return input.as_dict(); },       \
                      [](const py::dict &dict) { return NAME(dict); }))        \
      .def("as_dict", &NAME::as_dict)                                          \
      .def("add_edge", &NAME::add_edge)                                        \
      .def("delete_edge", &NAME::delete_edge)                                  \
      .def("clear_edges", &NAME::clear_edges)                                  \
      .def("add_point", &NAME::add_point, py::arg("p"),                        \
           py::arg("point_id") = -1,                                           \
           py::arg("neighbors") = std::vector<int>())                          \
      .def("add_line", &NAME::add_line, py::arg("line"),                       \
           py::arg("line_id") = -1, py::arg("neighbors") = std::vector<int>()) \
      .def("delete_point", &NAME::delete_point)                                \
      .def("delete_line", &NAME::delete_line)                                  \
      .def("update_point", &NAME::update_point)                                \
      .def("update_line", &NAME::update_line)                                  \
      .def("clear_points", &NAME::clear_points)                                \
      .def("clear_lines", &NAME::clear_lines)                                  \
      .def("init_points", &NAME::init_points, py::arg("points"),               \
           py::arg("ids") = std::vector<int>())                                \
      .def("init_lines", &NAME::init_lines, py::arg("lines"),                  \
           py::arg("ids") = std::vector<int>())                                \
      .def("reset", &NAME::reset)                                              \
      .def("count_lines", &NAME::count_lines)                                  \
      .def("count_points", &NAME::count_points)                                \
      .def("count_edges", &NAME::count_edges)                                  \
      .def("exist_point", &NAME::exist_point)                                  \
      .def("exist_line", &NAME::exist_line)                                    \
      .def("get_dict_points", &NAME::get_dict_points)                          \
      .def("get_dict_lines", &NAME::get_dict_lines)                            \
      .def("get_all_points", &NAME::get_all_points)                            \
      .def("get_all_lines", &NAME::get_all_lines)                              \
      .def("get_point_ids", &NAME::get_point_ids)                              \
      .def("get_line_ids", &NAME::get_line_ids)                                \
      .def("pdegree", &NAME::pdegree)                                          \
      .def("ldegree", &NAME::ldegree)                                          \
      .def("neighbor_lines", &NAME::neighbor_lines)                            \
      .def("neighbor_points", &NAME::neighbor_points)                          \
      .def("point", &NAME::point)                                              \
      .def("line", &NAME::line)                                                \
      .def("junc", &NAME::junc)                                                \
      .def("get_all_junctions", &NAME::get_all_junctions)                      \
      .def("add_junction", &NAME::add_junction, py::arg("junc"),               \
           py::arg("point_id") = -1)

  py::class_<PL_Bipartite2d>(m, "PL_Bipartite2d")
      REGISTER_BIPARTITE_BASE(PL_Bipartite2d)
          .def(py::init<const PL_Bipartite2dConfig &>())
          .def("add_keypoint", &PL_Bipartite2d::add_keypoint, py::arg("p"),
               py::arg("point_id") = -1)
          .def("add_keypoints_with_point3D_ids",
               &PL_Bipartite2d::add_keypoints_with_point3D_ids,
               py::arg("points"), py::arg("point3D_ids"),
               py::arg("ids") = std::vector<int>())
          .def("compute_intersection", &PL_Bipartite2d::compute_intersection)
          .def("compute_intersection_with_points",
               &PL_Bipartite2d::compute_intersection_with_points);

  py::class_<PL_Bipartite3d>(m, "PL_Bipartite3d")
      REGISTER_BIPARTITE_BASE(PL_Bipartite3d)
          .def("get_point_cloud", &PL_Bipartite3d::get_point_cloud)
          .def("get_line_cloud", &PL_Bipartite3d::get_line_cloud);

  py::class_<VPLine_Bipartite2d>(m, "VPLine_Bipartite2d")
      REGISTER_BIPARTITE_BASE(VPLine_Bipartite2d);

  py::class_<VPLine_Bipartite3d>(m, "VPLine_Bipartite3d")
      REGISTER_BIPARTITE_BASE(VPLine_Bipartite3d);

#undef REGISTER_BIPARTITE_BASE

  m.def("GetAllBipartites_VPLine2d", &GetAllBipartites_VPLine2d);
}

void bind_structures(py::module &m) { bind_bipartites(m); }

} // namespace limap
