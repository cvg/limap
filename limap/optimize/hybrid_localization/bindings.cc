#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "optimize/hybrid_localization/hybrid_localization.h"
#include "optimize/hybrid_localization/hybrid_localization_config.h"

namespace py = pybind11;

namespace limap {

void bind_lineloc_engine(py::module &m) {
  using namespace optimize::hybrid_localization;

  using LocEngine = LineLocEngine;
  using JointEngine = JointLocEngine;

  py::class_<LocEngine>(m, "LineLocEngine")
      .def(py::init<>())
      .def(py::init<const LineLocConfig &>())
      .def("Initialize",
           py::overload_cast<const std::vector<Line3d> &,
                             const std::vector<std::vector<Line2d>> &, M3D, M3D,
                             V3D>(&LocEngine::Initialize))
      .def("Initialize",
           py::overload_cast<const std::vector<Line3d> &,
                             const std::vector<std::vector<Line2d>> &, V4D, V4D,
                             V3D>(&LocEngine::Initialize))
      .def("SetUp", &LocEngine::SetUp)
      .def("Solve", &LocEngine::Solve)
      .def("GetFinalR", &LocEngine::GetFinalR)
      .def("GetFinalQ", &LocEngine::GetFinalQ)
      .def("GetFinalT", &LocEngine::GetFinalT)
      .def("IsSolutionUsable", &LocEngine::IsSolutionUsable)
      .def("GetInitialCost", &LocEngine::GetInitialCost)
      .def("GetFinalCost", &LocEngine::GetFinalCost);

  py::class_<JointEngine>(m, "JointLocEngine")
      .def(py::init<>())
      .def(py::init<const LineLocConfig &>())
      .def("Initialize",
           py::overload_cast<const std::vector<Line3d> &,
                             const std::vector<std::vector<Line2d>> &,
                             const std::vector<V3D> &, const std::vector<V2D> &,
                             M3D, M3D, V3D>(&JointEngine::Initialize))
      .def("Initialize",
           py::overload_cast<const std::vector<Line3d> &,
                             const std::vector<std::vector<Line2d>> &,
                             const std::vector<V3D> &, const std::vector<V2D> &,
                             V4D, V4D, V3D>(&JointEngine::Initialize))
      .def("SetUp", &JointEngine::SetUp)
      .def("Solve", &JointEngine::Solve)
      .def("GetFinalR", &JointEngine::GetFinalR)
      .def("GetFinalQ", &JointEngine::GetFinalQ)
      .def("GetFinalT", &JointEngine::GetFinalT)
      .def("IsSolutionUsable", &JointEngine::IsSolutionUsable)
      .def("GetInitialCost", &JointEngine::GetInitialCost)
      .def("GetFinalCost", &JointEngine::GetFinalCost);
}

void bind_hybrid_localization(py::module &m) {
  using namespace optimize::hybrid_localization;

  py::enum_<LineLocCostFunction>(m, "LineLocCostFunction")
      .value("E2DMidpointDist2", LineLocCostFunction::E2DMidpointDist2)
      .value("E2DMidpointAngleDist3",
             LineLocCostFunction::E2DMidpointAngleDist3)
      .value("E2DPerpendicularDist2",
             LineLocCostFunction::E2DPerpendicularDist2)
      .value("E2DPerpendicularDist4",
             LineLocCostFunction::E2DPerpendicularDist4)
      .value("E3DLineLineDist2", LineLocCostFunction::E3DLineLineDist2)
      .value("E3DPlaneLineDist2", LineLocCostFunction::E3DPlaneLineDist2);

  py::enum_<LineLocCostFunctionWeight>(m, "LineLocCostFunctionWeight")
      .value("ENoneWeight", LineLocCostFunctionWeight::ENoneWeight)
      .value("ECosineWeight", LineLocCostFunctionWeight::ECosineWeight)
      .value("ELine3dppWeight", LineLocCostFunctionWeight::ELine3dppWeight)
      .value("ELengthWeight", LineLocCostFunctionWeight::ELengthWeight)
      .value("EInvLengthWeight", LineLocCostFunctionWeight::EInvLengthWeight);

  py::class_<LineLocConfig>(m, "LineLocConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("solver_options", &LineLocConfig::solver_options)
      .def_readwrite("print_summary", &LineLocConfig::print_summary)
      .def_readwrite("weight_point", &LineLocConfig::weight_point)
      .def_readwrite("weight_line", &LineLocConfig::weight_line)
      .def_readwrite("cost_function", &LineLocConfig::cost_function)
      .def_readwrite("cost_function_weight",
                     &LineLocConfig::cost_function_weight);

  bind_lineloc_engine(m);
}

} // namespace limap
