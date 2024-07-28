#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include <RansacLib/ransac.h>

#include "estimators/absolute_pose/hybrid_pose_estimator.h"
#include "estimators/absolute_pose/joint_pose_estimator.h"

namespace py = pybind11;
using namespace py::literals;

namespace limap {

void bind_absolute_pose(py::module &m) {
  using namespace estimators::absolute_pose;

  py::class_<JointPoseEstimatorOptions>(m, "JointPoseEstimatorOptions")
      .def(py::init<>())
      .def_readwrite("ransac_options",
                     &JointPoseEstimatorOptions::ransac_options)
      .def_readwrite("lineloc_config",
                     &JointPoseEstimatorOptions::lineloc_config)
      .def_readwrite("cheirality_min_depth",
                     &JointPoseEstimatorOptions::cheirality_min_depth)
      .def_readwrite("cheirality_overlap_pixels",
                     &JointPoseEstimatorOptions::cheirality_overlap_pixels)
      .def_readwrite("sample_solver_first",
                     &JointPoseEstimatorOptions::sample_solver_first)
      .def_readwrite("random", &JointPoseEstimatorOptions::random);

  py::class_<HybridPoseEstimatorOptions>(m, "HybridPoseEstimatorOptions")
      .def(py::init<>())
      .def_readwrite("ransac_options",
                     &HybridPoseEstimatorOptions::ransac_options)
      .def_readwrite("lineloc_config",
                     &HybridPoseEstimatorOptions::lineloc_config)
      .def_readwrite("solver_flags", &HybridPoseEstimatorOptions::solver_flags)
      .def_readwrite("cheirality_min_depth",
                     &HybridPoseEstimatorOptions::cheirality_min_depth)
      .def_readwrite("cheirality_overlap_pixels",
                     &HybridPoseEstimatorOptions::cheirality_overlap_pixels)
      .def_readwrite("random", &HybridPoseEstimatorOptions::random);

  m.def("EstimateAbsolutePose_PointLine", &EstimateAbsolutePose_PointLine,
        "l3ds"_a, "l3d_ids"_a, "l2ds"_a, "p3ds"_a, "p2ds"_a, "cam"_a,
        "options_"_a);
  m.def("EstimateAbsolutePose_PointLine_Hybrid",
        &EstimateAbsolutePose_PointLine_Hybrid, "l3ds"_a, "l3d_ids"_a, "l2ds"_a,
        "p3ds"_a, "p2ds"_a, "cam"_a, "options_"_a);
}

} // namespace limap
