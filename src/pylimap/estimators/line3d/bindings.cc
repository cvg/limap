#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "limap/estimators/line3d/line3d_estimator.h"

namespace py = pybind11;

namespace limap {

void bind_line3d_fitting(py::module &m) {
  using namespace estimators::line3d;

  // Wrapper that returns (optional<Line3d>, RansacStats) tuple
  m.def(
      "estimate_line3d_robust",
      [](const std::vector<V3D> &points, double max_error,
         const poselib::RansacOptions &options) {
        poselib::RansacStats stats;
        auto result = EstimateLine3DRobust(points, max_error, options, &stats);
        return std::make_pair(result, stats);
      },
      py::arg("points"), py::arg("max_error"), py::arg("options"));
}

} // namespace limap
