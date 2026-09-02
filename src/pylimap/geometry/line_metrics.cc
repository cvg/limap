#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/geometry/line_metrics.h"

namespace py = pybind11;

namespace limap {

void BindLineMetrics(py::module &m) {
  py::enum_<LineMetricType>(m, "LineMetricType",
                            "Enum of supported line distance types.")
      .value("ANGULAR", LineMetricType::ANGULAR)
      .value("ENDPOINTS", LineMetricType::ENDPOINTS)
      .value("MIDPOINT", LineMetricType::MIDPOINT)
      .value("MIDPOINT_PERPENDICULAR", LineMetricType::MIDPOINT_PERPENDICULAR)
      .value("OVERLAP", LineMetricType::OVERLAP)
      .value("BIOVERLAP", LineMetricType::BIOVERLAP)
      .value("PERPENDICULAR_ONEWAY", LineMetricType::PERPENDICULAR_ONEWAY)
      .value("PERPENDICULAR", LineMetricType::PERPENDICULAR)
      .value("INNERSEG", LineMetricType::INNERSEG);

  m.def(
      "compute_distance_2d",
      [](const Line2d &l1, const Line2d &l2, const LineMetricType &type) {
        return ComputeDistance<Line2d>(l1, l2, type);
      },
      R"(
            Compute distance between two :class:`~limap.geometry.Line2d` using the specified line distance type.

            Args:
                l1 (:class:`~limap.geometry.Line2d`): First 2D line segment
                l2 (:class:`~limap.geometry.Line2d`): Second 2D line segment
                type (:class:`~limap.geometry.LineMetricType`): Line distance type
            
            Returns:
                `float`: The computed distance
        )",
      py::arg("l1"), py::arg("l2"), py::arg("type"));
  m.def(
      "compute_distance_3d",
      [](const Line3d &l1, const Line3d &l2, const LineMetricType &type) {
        return ComputeDistance<Line3d>(l1, l2, type);
      },
      R"(
            Compute distance between two :class:`~limap.geometry.Line3d` using the specified line distance type.

            Args:
                l1 (:class:`~limap.geometry.Line3d`): First 3D line segment
                l2 (:class:`~limap.geometry.Line3d`): Second 3D line segment
                type (:class:`~limap.geometry.LineMetricType`): Line distance type
            
            Returns:
                `float`: The computed distance
        )",
      py::arg("l1"), py::arg("l2"), py::arg("type"));
  m.def(
      "compute_pairwise_distance_2d",
      [](const std::vector<Line2d> &lines, const LineMetricType &type) {
        return ComputePairwiseDistance<Line2d>(lines, type);
      },
      R"(
            Compute pairwise distance among a list of :class:`~limap.geometry.Line2d` using the specified line distance type.

            Args:
                lines (list[:class:`~limap.geometry.Line2d`]): List of 2D line segments
                type (:class:`~limap.geometry.LineMetricType`): Line distance type
            
            Returns:
                :class:`np.array`: The computed pairwise distance matrix
        )",
      py::arg("lines"), py::arg("type"));
  m.def(
      "compute_pairwise_distance_3d",
      [](const std::vector<Line3d> &lines, const LineMetricType &type) {
        return ComputePairwiseDistance<Line3d>(lines, type);
      },
      R"(
            Compute pairwise distance among a list of :class:`~limap.geometry.Line3d` using the specified line distance type.

            Args:
                 lines (list[:class:`~limap.geometry.Line3d`]): List of 3D line segments
                type (:class:`~limap.geometry.LineMetricType`): Line distance type
            
            Returns:
                :class:`np.array`: The computed pairwise distance matrix
        )",
      py::arg("lines"), py::arg("type"));
}

} // namespace limap
