#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/estimators/group3d/cone_estimator.h"
#include "limap/estimators/group3d/plane_estimator.h"
#include "limap/estimators/group3d/sphere_estimator.h"
#include "limap/estimators/group3d/vp_estimator.h"

namespace limap {

namespace estimators {

namespace group3d {

void bind_group3d_estimator(py::module &m) {
  // Non-robust estimation
  m.def("estimate_vp", &EstimateVP, "Estimate VP direction from 3D lines",
        "lines"_a);

  m.def("estimate_plane_from_points", &EstimatePlaneFromPoints,
        "Estimate plane from 3D points", "points"_a);

  m.def("estimate_plane_from_lines", &EstimatePlaneFromLines,
        "Estimate plane from 3D lines", "lines"_a);

  m.def("estimate_plane", &EstimatePlane,
        "Estimate plane from 3D points and lines", "points"_a, "lines"_a);

  // Robust estimation with RANSAC - returns (optional<Model>, RansacStats)
  // tuple
  m.def(
      "estimate_vp_robust",
      [](const std::vector<Line3d> &lines, double max_error,
         const poselib::RansacOptions &options) {
        poselib::RansacStats stats;
        auto result = EstimateVPRobust(lines, max_error, options, &stats);
        return std::make_pair(result, stats);
      },
      "Robust VP estimation from 3D lines using RANSAC", py::arg("lines"),
      py::arg("max_error"), py::arg("options"));

  m.def(
      "estimate_plane_robust",
      [](const std::vector<V3D> &points, const std::vector<Line3d> &lines,
         double max_error, const poselib::RansacOptions &options) {
        poselib::RansacStats stats;
        auto result =
            EstimatePlaneRobust(points, lines, max_error, options, &stats);
        return std::make_pair(result, stats);
      },
      "Robust plane estimation from points and lines using RANSAC",
      py::arg("points"), py::arg("lines"), py::arg("max_error"),
      py::arg("options"));

  // Sphere estimation
  m.def("estimate_sphere", &EstimateSphere, "Estimate sphere from 3D points",
        "points"_a);

  m.def(
      "estimate_sphere_robust",
      [](const std::vector<V3D> &points, double max_error,
         const poselib::RansacOptions &options) {
        poselib::RansacStats stats;
        auto result = EstimateSphereRobust(points, max_error, options, &stats);
        return std::make_pair(result, stats);
      },
      "Robust sphere estimation from 3D points using RANSAC", py::arg("points"),
      py::arg("max_error"), py::arg("options"));

  // Cone estimation
  m.def("estimate_cone", &EstimateCone, "Estimate cone from 3D points",
        "points"_a);

  m.def(
      "estimate_cone_robust",
      [](const std::vector<V3D> &points, double max_error,
         const poselib::RansacOptions &options) {
        poselib::RansacStats stats;
        auto result = EstimateConeRobust(points, max_error, options, &stats);
        return std::make_pair(result, stats);
      },
      "Robust cone estimation from 3D points using RANSAC", py::arg("points"),
      py::arg("max_error"), py::arg("options"));
}

} // namespace group3d

} // namespace estimators

void bind_estimators_group3d(py::module &m) {
  using namespace estimators::group3d;
  bind_group3d_estimator(m);
}

} // namespace limap
