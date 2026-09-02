#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/estimators/group3d/group3d_estimator.h"
#include "limap/geometry/groups.h"

namespace py = pybind11;

namespace limap {

void BindGroups(py::module &m) {
  py::enum_<GroupType>(m, "GroupType")
      .value("INVALID", GroupType::INVALID)
      .value("VP", GroupType::VP)
      .value("PLANE", GroupType::PLANE)
      .value("SPHERE", GroupType::SPHERE)
      .value("CYLINDER", GroupType::CYLINDER)
      .value("ELLIPSOID", GroupType::ELLIPSOID)
      .value("CUBOID", GroupType::CUBOID)
      .value("CONE", GroupType::CONE);

  m.def("get_num_params_in_2d_by_group_type", &GetNumParamsIn2DByGroupType,
        py::arg("type"),
        "Get number of parameters for the corresponding group type in 2D");

  m.def("get_num_params_in_3d_by_group_type", &GetNumParamsIn3DByGroupType,
        py::arg("type"),
        "Get number of parameters for the corresponding group type in 3D");

  m.def("initialize_group_params", &estimators::group3d::InitializeGroupParams,
        py::arg("type"), py::arg("points"), py::arg("lines"),
        py::arg("init_ransac_thresh") = -1.0,
        "Initialize group parameters from geometric features. Returns the "
        "parameter vector if successful, None otherwise. "
        "If init_ransac_thresh <= 0, uses the group's default threshold.");

  m.def(
      "normalize_group_params_3d",
      [](GroupType type, std::vector<double> params) {
        NormalizeGroupParams3D(type, params.data());
        return params;
      },
      py::arg("type"), py::arg("params"),
      "Normalize 3D group parameters to canonical form. Returns normalized "
      "params.");

  m.def(
      "check_group_params_3d",
      [](GroupType type, const std::vector<double> &params, double tol) {
        return CheckGroupParams3D(type, params.data(), tol);
      },
      py::arg("type"), py::arg("params"), py::arg("tol") = 1e-6,
      "Check if 3D group parameters are valid and normalized");

  m.def("get_default_group_params_2d", &GetDefaultGroupParams2D,
        py::arg("type"), "Get default 2D parameters for a group type");

  m.def("get_default_group_params_3d", &GetDefaultGroupParams3D,
        py::arg("type"), "Get default 3D parameters for a group type");
}

} // namespace limap
