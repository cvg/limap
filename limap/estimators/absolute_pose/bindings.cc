#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include <RansacLib/ransac.h>

#include "estimators/absolute_pose/joint_pose_estimator.h"
#include "estimators/absolute_pose/hybrid_pose_estimator.h"

namespace py = pybind11;
using namespace py::literals;

namespace limap {

void bind_absolute_pose(py::module& m) {
    using namespace estimators::absolute_pose;

    m.def("EstimateAbsolutePose_PointLine", &EstimateAbsolutePose_PointLine,
            "tracks"_a, "l3d_ids"_a, "l2ds"_a, "p3ds"_a, "p2ds"_a, "cam"_a, "cfg"_a, "options_"_a, "sample_solver_first"_a = false,
            "cheirality_min_depth"_a = 0.0, "line_min_projected_length"_a = 1
    );
    m.def("EstimateAbsolutePose_PointLine_Hybrid", &EstimateAbsolutePose_PointLine_Hybrid,
            "tracks"_a, "l3d_ids"_a, "l2ds"_a, "p3ds"_a, "p2ds"_a, "cam"_a, "cfg"_a, "options_"_a, "solver_flags"_a,
            "cheirality_min_depth"_a = 0.0, "line_min_projected_length"_a = 1
    );
}

} // namespace limap

