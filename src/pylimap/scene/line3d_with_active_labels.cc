#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/scene/line3d_with_active_labels.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace limap {

void BindLine3dWithActiveLabels(py::module &m) {
  py::classh<Line3dWithActiveLabels, Line3d>(m, "Line3dWithActiveLabels")
      .def(py::init<>())
      .def(py::init<const Line3d &>(), "line"_a)
      // Per-observation activity
      .def("set_observation_active",
           &Line3dWithActiveLabels::SetObservationActive, "image_id"_a)
      .def("set_observation_inactive",
           &Line3dWithActiveLabels::SetObservationInactive, "image_id"_a)
      .def("is_observation_active",
           &Line3dWithActiveLabels::IsObservationActive, "image_id"_a)
      // Track-level reliability
      .def("count_active_observations",
           &Line3dWithActiveLabels::CountActiveObservations)
      .def("active_ratio", &Line3dWithActiveLabels::ActiveRatio)
      .def("is_reliable", &Line3dWithActiveLabels::IsReliable,
           "min_active"_a = 1)
      // Maintenance
      .def("cleanup_inactive_set", &Line3dWithActiveLabels::CleanupInactiveSet)
      .def("merge_inactive_from", &Line3dWithActiveLabels::MergeInactiveFrom,
           "other"_a)
      .def("clear_inactive_labels",
           &Line3dWithActiveLabels::ClearInactiveLabels)
      .def_property_readonly("inactive_images",
                             &Line3dWithActiveLabels::InactiveImages);
}

} // namespace limap
