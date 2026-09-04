#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/group_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/estimators/bundle_adjustment/wireframe_bundle_adjustment.h"
#include "pylimap/helpers.h"

namespace py = pybind11;

namespace limap {

void bind_bundle_adjustment(py::module &m) {
  using namespace estimators;

  // PointLineBundleAdjustmentOptions (inherits from
  // colmap::BundleAdjustmentOptions) Base class bindings from pycolmap are
  // inherited automatically
  auto PyPointLineBundleAdjustmentOptions =
      py::classh<PointLineBundleAdjustmentOptions,
                 colmap::BundleAdjustmentOptions>(
          m, "PointLineBundleAdjustmentOptions")
          .def(py::init<>())
          // New fields for point-line BA (base class fields inherited from
          // pycolmap)
          .def_readwrite("refine_points",
                         &PointLineBundleAdjustmentOptions::refine_points)
          .def_readwrite("refine_lines",
                         &PointLineBundleAdjustmentOptions::refine_lines)
          .def_readwrite("weight_point",
                         &PointLineBundleAdjustmentOptions::weight_point)
          .def_readwrite("weight_line",
                         &PointLineBundleAdjustmentOptions::weight_line)
          .def_readwrite("line_weight_normalization_length",
                         &PointLineBundleAdjustmentOptions::
                             line_weight_normalization_length)
          .def_readwrite(
              "loss_function_type_line",
              &PointLineBundleAdjustmentOptions::loss_function_type_line)
          .def_readwrite(
              "loss_function_scale_line",
              &PointLineBundleAdjustmentOptions::loss_function_scale_line)
          .def_readwrite(
              "use_parameter_block_ordering",
              &PointLineBundleAdjustmentOptions::use_parameter_block_ordering);
  MakeDataclass(PyPointLineBundleAdjustmentOptions);

  // PointLineBundleAdjustmentConfig (inherits from
  // colmap::BundleAdjustmentConfig) Base class bindings from pycolmap are
  // inherited automatically
  py::classh<PointLineBundleAdjustmentConfig, colmap::BundleAdjustmentConfig>(
      m, "PointLineBundleAdjustmentConfig")
      .def(py::init<>())
      // Lines (new in PointLineBundleAdjustmentConfig)
      .def("num_lines", &PointLineBundleAdjustmentConfig::NumLines)
      .def("add_variable_line",
           &PointLineBundleAdjustmentConfig::AddVariableLine)
      .def("add_constant_line",
           &PointLineBundleAdjustmentConfig::AddConstantLine)
      .def("has_line", &PointLineBundleAdjustmentConfig::HasLine)
      .def("has_variable_line",
           &PointLineBundleAdjustmentConfig::HasVariableLine)
      .def("has_constant_line",
           &PointLineBundleAdjustmentConfig::HasConstantLine)
      .def("remove_variable_line",
           &PointLineBundleAdjustmentConfig::RemoveVariableLine)
      .def("remove_constant_line",
           &PointLineBundleAdjustmentConfig::RemoveConstantLine)
      .def("variable_lines", &PointLineBundleAdjustmentConfig::VariableLines)
      .def("constant_lines", &PointLineBundleAdjustmentConfig::ConstantLines);

  // BasePointLineBundleAdjuster (inherits from colmap::CeresBundleAdjuster)
  // Base class bindings from pycolmap are inherited automatically
  py::classh<BasePointLineBundleAdjuster, colmap::CeresBundleAdjuster>(
      m, "BasePointLineBundleAdjuster")
      .def(
          py::init<PointLineBundleAdjustmentOptions,
                   PointLineBundleAdjustmentConfig, HolisticReconstruction &>(),
          py::arg("options"), py::arg("config"), py::arg("reconstruction"))
      // Structure-specific accessors (base class methods inherited from
      // pycolmap)
      .def("refine_points", &BasePointLineBundleAdjuster::refine_points)
      .def("refine_lines", &BasePointLineBundleAdjuster::refine_lines)
      .def("weight_point", &BasePointLineBundleAdjuster::weight_point)
      .def("weight_line", &BasePointLineBundleAdjuster::weight_line)
      .def("variable_lines", &BasePointLineBundleAdjuster::VariableLines)
      .def("constant_lines", &BasePointLineBundleAdjuster::ConstantLines)
      .def("has_variable_line", &BasePointLineBundleAdjuster::HasVariableLine)
      .def("has_constant_line", &BasePointLineBundleAdjuster::HasConstantLine);

  // Factory function
  m.def("create_point_line_bundle_adjuster", &CreatePointLineBundleAdjuster,
        py::arg("options"), py::arg("config"), py::arg("reconstruction"));

  //////////////////////////////////////////////////////////////////////////////
  // Group Bundle Adjustment
  //////////////////////////////////////////////////////////////////////////////

  // GroupBundleAdjustmentOptions (inherits from
  // PointLineBundleAdjustmentOptions)
  auto PyGroupBundleAdjustmentOptions =
      py::classh<GroupBundleAdjustmentOptions,
                 PointLineBundleAdjustmentOptions>(
          m, "GroupBundleAdjustmentOptions")
          .def(py::init<>())
          // Group-specific options
          .def_readwrite("refine_groups",
                         &GroupBundleAdjustmentOptions::refine_groups)
          .def_readwrite("weight_vp", &GroupBundleAdjustmentOptions::weight_vp)
          .def_readwrite("weight_plane",
                         &GroupBundleAdjustmentOptions::weight_plane)
          .def_readwrite("vp_residual_type",
                         &GroupBundleAdjustmentOptions::vp_residual_type)
          .def_readwrite(
              "loss_function_type_angle",
              &GroupBundleAdjustmentOptions::loss_function_type_angle)
          .def_readwrite(
              "loss_function_scale_angle",
              &GroupBundleAdjustmentOptions::loss_function_scale_angle)
          .def_readwrite(
              "loss_function_type_reprojection",
              &GroupBundleAdjustmentOptions::loss_function_type_reprojection)
          .def_readwrite(
              "loss_function_scale_reprojection",
              &GroupBundleAdjustmentOptions::loss_function_scale_reprojection)
          .def_readwrite(
              "min_active_group_associations",
              &GroupBundleAdjustmentOptions::min_active_group_associations)
          .def_readwrite("min_reliable_group_associations_step2",
                         &GroupBundleAdjustmentOptions::
                             min_reliable_group_associations_step2)
          .def_readwrite(
              "angular_constraint_threshold_deg",
              &GroupBundleAdjustmentOptions::angular_constraint_threshold_deg)
          .def_readwrite("angular_constraint_min_covisible_images",
                         &GroupBundleAdjustmentOptions::
                             angular_constraint_min_covisible_images)
          .def_readwrite("loss_function_type_angular_constraint",
                         &GroupBundleAdjustmentOptions::
                             loss_function_type_angular_constraint)
          .def_readwrite("loss_function_scale_angular_constraint",
                         &GroupBundleAdjustmentOptions::
                             loss_function_scale_angular_constraint)
          .def_readwrite(
              "enforce_vp_angular_constraints",
              &GroupBundleAdjustmentOptions::enforce_vp_angular_constraints)
          .def_readwrite(
              "weight_vp_angular_constraint",
              &GroupBundleAdjustmentOptions::weight_vp_angular_constraint)
          .def_readwrite(
              "enforce_plane_angular_constraints",
              &GroupBundleAdjustmentOptions::enforce_plane_angular_constraints)
          .def_readwrite(
              "weight_plane_angular_constraint",
              &GroupBundleAdjustmentOptions::weight_plane_angular_constraint);
  MakeDataclass(PyGroupBundleAdjustmentOptions);

  // GroupBundleAdjustmentConfig (inherits from PointLineBundleAdjustmentConfig)
  py::classh<GroupBundleAdjustmentConfig, PointLineBundleAdjustmentConfig>(
      m, "GroupBundleAdjustmentConfig")
      .def(py::init<>())
      // Groups (new in GroupBundleAdjustmentConfig)
      .def("num_groups", &GroupBundleAdjustmentConfig::NumGroups)
      .def("add_variable_group", &GroupBundleAdjustmentConfig::AddVariableGroup)
      .def("add_constant_group", &GroupBundleAdjustmentConfig::AddConstantGroup)
      .def("has_group", &GroupBundleAdjustmentConfig::HasGroup)
      .def("has_variable_group", &GroupBundleAdjustmentConfig::HasVariableGroup)
      .def("has_constant_group", &GroupBundleAdjustmentConfig::HasConstantGroup)
      .def("remove_variable_group",
           &GroupBundleAdjustmentConfig::RemoveVariableGroup)
      .def("remove_constant_group",
           &GroupBundleAdjustmentConfig::RemoveConstantGroup)
      .def("variable_groups", &GroupBundleAdjustmentConfig::VariableGroups)
      .def("constant_groups", &GroupBundleAdjustmentConfig::ConstantGroups);

  // GroupBundleAdjuster (inherits from BasePointLineBundleAdjuster)
  py::classh<GroupBundleAdjuster, BasePointLineBundleAdjuster>(
      m, "GroupBundleAdjuster")
      .def(py::init<GroupBundleAdjustmentOptions, GroupBundleAdjustmentConfig,
                    HolisticReconstruction &>(),
           py::arg("options"), py::arg("config"), py::arg("reconstruction"))
      // Group-specific accessors
      .def("refine_groups", &GroupBundleAdjuster::refine_groups)
      .def("weight_vp", &GroupBundleAdjuster::weight_vp)
      .def("weight_plane", &GroupBundleAdjuster::weight_plane)
      .def("variable_groups", &GroupBundleAdjuster::VariableGroups)
      .def("constant_groups", &GroupBundleAdjuster::ConstantGroups)
      .def("has_variable_group", &GroupBundleAdjuster::HasVariableGroup)
      .def("has_constant_group", &GroupBundleAdjuster::HasConstantGroup);

  // Factory function for group bundle adjuster
  m.def("create_group_bundle_adjuster", &CreateGroupBundleAdjuster,
        py::arg("options"), py::arg("config"), py::arg("reconstruction"));

  //////////////////////////////////////////////////////////////////////////////
  // Wireframe Bundle Adjustment
  //////////////////////////////////////////////////////////////////////////////

  // WireframeBundleAdjustmentOptions (inherits from
  // PointLineBundleAdjustmentOptions)
  auto PyWireframeBundleAdjustmentOptions =
      py::classh<WireframeBundleAdjustmentOptions,
                 PointLineBundleAdjustmentOptions>(
          m, "WireframeBundleAdjustmentOptions")
          .def(py::init<>())
          // Wireframe-specific options
          .def_readwrite("refine_wireframe",
                         &WireframeBundleAdjustmentOptions::refine_wireframe)
          .def_readwrite("weight_wireframe",
                         &WireframeBundleAdjustmentOptions::weight_wireframe)
          .def_readwrite(
              "loss_function_type_wireframe",
              &WireframeBundleAdjustmentOptions::loss_function_type_wireframe)
          .def_readwrite(
              "loss_function_scale_wireframe",
              &WireframeBundleAdjustmentOptions::loss_function_scale_wireframe)
          .def_readwrite(
              "force_reconstruct_wireframe3d",
              &WireframeBundleAdjustmentOptions::force_reconstruct_wireframe3d)
          .def_readwrite("wireframe_voting",
                         &WireframeBundleAdjustmentOptions::wireframe_voting);
  MakeDataclass(PyWireframeBundleAdjustmentOptions);

  // WireframeBundleAdjuster (inherits from BasePointLineBundleAdjuster)
  py::classh<WireframeBundleAdjuster, BasePointLineBundleAdjuster>(
      m, "WireframeBundleAdjuster")
      .def(
          py::init<WireframeBundleAdjustmentOptions,
                   PointLineBundleAdjustmentConfig, HolisticReconstruction &>(),
          py::arg("options"), py::arg("config"), py::arg("reconstruction"))
      // Wireframe-specific accessors
      .def("refine_wireframe", &WireframeBundleAdjuster::refine_wireframe)
      .def("weight_wireframe", &WireframeBundleAdjuster::weight_wireframe)
      .def("num_wireframe_observations",
           &WireframeBundleAdjuster::NumWireframeObservations);

  // Factory function for wireframe bundle adjuster
  m.def("create_wireframe_bundle_adjuster", &CreateWireframeBundleAdjuster,
        py::arg("options"), py::arg("config"), py::arg("reconstruction"));

  //////////////////////////////////////////////////////////////////////////////
  // Structure Bundle Adjustment (combines Groups + Wireframe)
  //////////////////////////////////////////////////////////////////////////////

  // StructureBundleAdjustmentOptions (inherits from
  // GroupBundleAdjustmentOptions)
  auto PyStructureBundleAdjustmentOptions =
      py::classh<StructureBundleAdjustmentOptions,
                 GroupBundleAdjustmentOptions>(
          m, "StructureBundleAdjustmentOptions")
          .def(py::init<>())
          // Wireframe-specific options (group options inherited)
          .def_readwrite("refine_wireframe",
                         &StructureBundleAdjustmentOptions::refine_wireframe)
          .def_readwrite("weight_wireframe",
                         &StructureBundleAdjustmentOptions::weight_wireframe)
          .def_readwrite(
              "loss_function_type_wireframe",
              &StructureBundleAdjustmentOptions::loss_function_type_wireframe)
          .def_readwrite(
              "loss_function_scale_wireframe",
              &StructureBundleAdjustmentOptions::loss_function_scale_wireframe)
          .def_readwrite(
              "force_reconstruct_wireframe3d",
              &StructureBundleAdjustmentOptions::force_reconstruct_wireframe3d)
          .def_readwrite("wireframe_voting",
                         &StructureBundleAdjustmentOptions::wireframe_voting);
  MakeDataclass(PyStructureBundleAdjustmentOptions);

  // StructureBundleAdjustmentConfig is an alias for GroupBundleAdjustmentConfig
  m.attr("StructureBundleAdjustmentConfig") =
      m.attr("GroupBundleAdjustmentConfig");

  // StructureBundleAdjuster (inherits from GroupBundleAdjuster)
  py::classh<StructureBundleAdjuster, GroupBundleAdjuster>(
      m, "StructureBundleAdjuster")
      .def(
          py::init<StructureBundleAdjustmentOptions,
                   StructureBundleAdjustmentConfig, HolisticReconstruction &>(),
          py::arg("options"), py::arg("config"), py::arg("reconstruction"))
      // Wireframe-specific accessors
      .def("refine_wireframe", &StructureBundleAdjuster::refine_wireframe)
      .def("weight_wireframe", &StructureBundleAdjuster::weight_wireframe)
      .def("num_wireframe_observations",
           &StructureBundleAdjuster::NumWireframeObservations);

  // Factory function for structure bundle adjuster
  m.def("create_structure_bundle_adjuster", &CreateStructureBundleAdjuster,
        py::arg("options"), py::arg("config"), py::arg("reconstruction"));

  //////////////////////////////////////////////////////////////////////////////
  // Group 3D-Cost Bundle Adjustment (ablation: 3D plane costs instead of 2D)
}

} // namespace limap
