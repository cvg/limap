#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

namespace py = pybind11;

#include "optimize/global_pl_association/global_associator.h"

namespace limap {

void bind_global_pl_association(py::module &m) {
  using namespace optimize::global_pl_association;
  py::class_<GlobalAssociatorConfig>(m, "GlobalAssociatorConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("solver_options", &GlobalAssociatorConfig::solver_options)
      .def_readwrite("problem_options",
                     &GlobalAssociatorConfig::problem_options)
      .def_readwrite("point_geometric_loss_function",
                     &GlobalAssociatorConfig::point_geometric_loss_function)
      .def_readwrite("line_geometric_loss_function",
                     &GlobalAssociatorConfig::line_geometric_loss_function)
      .def_readwrite(
          "point_line_association_3d_loss_function",
          &GlobalAssociatorConfig::point_line_association_3d_loss_function)
      .def_readwrite(
          "vp_line_association_3d_loss_function",
          &GlobalAssociatorConfig::vp_line_association_3d_loss_function)
      .def_readwrite("vp_orthogonality_loss_function",
                     &GlobalAssociatorConfig::vp_orthogonality_loss_function)
      .def_readwrite("vp_collinearity_loss_function",
                     &GlobalAssociatorConfig::vp_collinearity_loss_function)
      .def_readwrite("lw_point", &GlobalAssociatorConfig::lw_point)
      .def_readwrite("geometric_alpha",
                     &GlobalAssociatorConfig::geometric_alpha)
      .def_readwrite("th_count_lineline",
                     &GlobalAssociatorConfig::th_count_lineline)
      .def_readwrite("th_angle_lineline",
                     &GlobalAssociatorConfig::th_angle_lineline)
      .def_readwrite("lw_pointline_association",
                     &GlobalAssociatorConfig::lw_pointline_association)
      .def_readwrite("th_pixel", &GlobalAssociatorConfig::th_pixel)
      .def_readwrite("th_weight_pointline",
                     &GlobalAssociatorConfig::th_weight_pointline)
      .def_readwrite("lw_vpline_association",
                     &GlobalAssociatorConfig::lw_vpline_association)
      .def_readwrite("th_count_vpline",
                     &GlobalAssociatorConfig::th_count_vpline)
      .def_readwrite("lw_vp_orthogonality",
                     &GlobalAssociatorConfig::lw_vp_orthogonality)
      .def_readwrite("th_angle_orthogonality",
                     &GlobalAssociatorConfig::th_angle_orthogonality)
      .def_readwrite("lw_vp_collinearity",
                     &GlobalAssociatorConfig::lw_vp_orthogonality)
      .def_readwrite("th_angle_collinearity",
                     &GlobalAssociatorConfig::th_angle_collinearity)
      .def_readwrite("print_summary", &GlobalAssociatorConfig::print_summary)
      .def_readwrite("th_hard_pl_dist3d",
                     &GlobalAssociatorConfig::th_hard_pl_dist3d)
      .def_readwrite("th_hard_vpline_angle3d",
                     &GlobalAssociatorConfig::th_hard_vpline_angle3d)
      .def_readwrite("constant_intrinsics",
                     &GlobalAssociatorConfig::constant_intrinsics)
      .def_readwrite("constant_pose", &GlobalAssociatorConfig::constant_pose)
      .def_readwrite("constant_point", &GlobalAssociatorConfig::constant_point)
      .def_readwrite("constant_line", &GlobalAssociatorConfig::constant_line)
      .def_readwrite("constant_vp", &GlobalAssociatorConfig::constant_vp);

  py::class_<GlobalAssociator>(m, "GlobalAssociator")
      .def(py::init<>())
      .def(py::init<const GlobalAssociatorConfig &>())
      .def("InitImagecols", &GlobalAssociator::InitImagecols)
      .def("InitPointTracks", static_cast<void (GlobalAssociator::*)(
                                  const std::vector<PointTrack> &)>(
                                  &GlobalAssociator::InitPointTracks))
      .def("InitPointTracks", static_cast<void (GlobalAssociator::*)(
                                  const std::map<int, PointTrack> &)>(
                                  &GlobalAssociator::InitPointTracks))
      .def("InitLineTracks", static_cast<void (GlobalAssociator::*)(
                                 const std::vector<LineTrack> &)>(
                                 &GlobalAssociator::InitLineTracks))
      .def("InitLineTracks", static_cast<void (GlobalAssociator::*)(
                                 const std::map<int, LineTrack> &)>(
                                 &GlobalAssociator::InitLineTracks))
      .def("InitVPTracks", static_cast<void (GlobalAssociator::*)(
                               const std::vector<vplib::VPTrack> &)>(
                               &GlobalAssociator::InitVPTracks))
      .def("InitVPTracks", static_cast<void (GlobalAssociator::*)(
                               const std::map<int, vplib::VPTrack> &)>(
                               &GlobalAssociator::InitVPTracks))
      .def("Init2DBipartites_PointLine",
           &GlobalAssociator::Init2DBipartites_PointLine)
      .def("Init2DBipartites_VPLine",
           &GlobalAssociator::Init2DBipartites_VPLine)
      .def("ReassociateJunctions", &GlobalAssociator::ReassociateJunctions)
      .def("SetUp", &GlobalAssociator::SetUp)
      .def("Solve", &GlobalAssociator::Solve)
      .def("GetOutputImagecols", &GlobalAssociator::GetOutputImagecols)
      .def("GetOutputPoints", &GlobalAssociator::GetOutputPoints)
      .def("GetOutputPointTracks", &GlobalAssociator::GetOutputPointTracks)
      .def("GetOutputLines", &GlobalAssociator::GetOutputLines)
      .def("GetOutputLineTracks", &GlobalAssociator::GetOutputLineTracks)
      .def("GetOutputVPs", &GlobalAssociator::GetOutputVPs)
      .def("GetOutputVPTracks", &GlobalAssociator::GetOutputVPTracks)
      .def("GetBipartite3d_PointLine_Constraints",
           &GlobalAssociator::GetBipartite3d_PointLine_Constraints)
      .def("GetBipartite3d_PointLine",
           &GlobalAssociator::GetBipartite3d_PointLine)
      .def("GetBipartite3d_VPLine", &GlobalAssociator::GetBipartite3d_VPLine);
}

} // namespace limap
