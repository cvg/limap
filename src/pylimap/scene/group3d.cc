#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/group3d.h"
#include "limap/scene/group3d_with_active_labels.h"

namespace py = pybind11;

namespace limap {

void BindGroup3d(py::module &m) {
  py::classh<AssociatedFeature3d>(m, "AssociatedFeature3d")
      .def(py::init<>())
      .def(py::init<feature3D_t, double>(), py::arg("idx"), py::arg("w") = 1.0)
      .def_readwrite("idx", &AssociatedFeature3d::idx)
      .def_readwrite("w", &AssociatedFeature3d::w);

  py::classh<Group3d>(m, "Group3d")
      .def(py::init<>())
      .def(py::init<GroupType>())
      .def(py::init<GroupType, const std::vector<double> &>())
      .def(py::init<GroupType, const std::vector<double> &,
                    const std::vector<AssociatedFeature3d> &,
                    const std::vector<AssociatedFeature3d> &>(),
           py::arg("type"), py::arg("params"), py::arg("points"),
           py::arg("lines") = std::vector<AssociatedFeature3d>{})
      .def_readwrite("type", &Group3d::type)
      .def_property(
          "params",
          static_cast<const std::vector<double> &(Group3d::*)() const>(
              &Group3d::GetParams),
          &Group3d::SetParams)
      .def_readwrite("points", &Group3d::points)
      .def_readwrite("lines", &Group3d::lines)
      .def_readwrite("track", &Group3d::track)
      .def("add_point",
           py::overload_cast<const AssociatedFeature3d &>(&Group3d::AddPoint))
      .def("add_line",
           py::overload_cast<const AssociatedFeature3d &>(&Group3d::AddLine))
      .def("add_points",
           py::overload_cast<const std::vector<AssociatedFeature3d> &>(
               &Group3d::AddPoints))
      .def("add_lines",
           py::overload_cast<const std::vector<AssociatedFeature3d> &>(
               &Group3d::AddLines))
      .def("add_point",
           py::overload_cast<const feature3D_t &>(&Group3d::AddPoint))
      .def("add_line",
           py::overload_cast<const feature3D_t &>(&Group3d::AddLine))
      .def("add_points",
           py::overload_cast<const std::vector<feature3D_t> &,
                             const std::vector<double> &>(&Group3d::AddPoints))
      .def("add_lines",
           py::overload_cast<const std::vector<feature3D_t> &,
                             const std::vector<double> &>(&Group3d::AddLines))
      .def("normalize_params", &Group3d::NormalizeParams,
           "Normalize params in-place to canonical form")
      .def("check_params", &Group3d::CheckParams, py::arg("tol") = 1e-6,
           "Check if params are valid and normalized")
      .def("__repr__", [](const Group3d &g) {
        std::ostringstream oss;
        oss << "Group3d(type=" << g.type << ", len=" << g.track.Length();
        if (!g.points.empty()) {
          oss << ", n_points=" << g.points.size();
        }
        if (!g.lines.empty()) {
          oss << ", n_lines=" << g.lines.size();
        }
        oss << ")";
        return oss.str();
      });

  py::classh<Group3dWithActiveLabels, Group3d>(m, "Group3dWithActiveLabels")
      .def(py::init<>())
      .def(py::init<const Group3d &>())
      .def("set_point_active", &Group3dWithActiveLabels::SetPointActive,
           py::arg("id"))
      .def("set_point_inactive", &Group3dWithActiveLabels::SetPointInactive,
           py::arg("id"))
      .def("is_point_active", &Group3dWithActiveLabels::IsPointActive,
           py::arg("id"))
      .def("set_line_active", &Group3dWithActiveLabels::SetLineActive,
           py::arg("id"))
      .def("set_line_inactive", &Group3dWithActiveLabels::SetLineInactive,
           py::arg("id"))
      .def("is_line_active", &Group3dWithActiveLabels::IsLineActive,
           py::arg("id"))
      .def("count_active_points", &Group3dWithActiveLabels::CountActivePoints)
      .def("count_active_lines", &Group3dWithActiveLabels::CountActiveLines)
      .def("count_active_associations",
           &Group3dWithActiveLabels::CountActiveAssociations)
      .def("cleanup_inactive_sets",
           &Group3dWithActiveLabels::CleanupInactiveSets)
      .def("clear_inactive_labels",
           &Group3dWithActiveLabels::ClearInactiveLabels)
      .def("purge_inactive_associations",
           &Group3dWithActiveLabels::PurgeInactiveAssociations);
}

} // namespace limap
