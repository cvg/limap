#include "estimators/absolute_pose/hybrid_pose_estimator.h"
#include "estimators/absolute_pose/pl_absolute_pose_ransac.h"
#include "estimators/absolute_pose/pl_absolute_pose_hybrid_ransac.h"
#include "optimize/line_localization/cost_functions.h"

namespace limap {

namespace estimators {

namespace absolute_pose {

namespace lineloc = optimize::line_localization;

std::pair<CameraPose, ransac_lib::HybridRansacStatistics> 
EstimateAbsolutePose_PointLine_Hybrid(const std::vector<Line3d>& l3ds, const std::vector<int>& l3d_ids, const std::vector<Line2d>& l2ds, 
                                      const std::vector<V3D>& p3ds, const std::vector<V2D>& p2ds, const Camera& cam, 
                                      const ExtendedHybridLORansacOptions& options_, const lineloc::LineLocConfig& cfg, 
                                      const std::vector<bool>& solver_flags, const double cheirality_min_depth, 
                                      const double line_min_projected_length) {
    ExtendedHybridLORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    HybridPoseEstimator solver(l3ds, l3d_ids, l2ds, p3ds, p2ds, cam, cfg);
    solver.set_solver_flags(solver_flags);

    PointLineAbsolutePoseHybridRansac<CameraPose, std::vector<CameraPose>, HybridPoseEstimator> hybrid_lomsac;
    CameraPose best_model;
    ransac_lib::HybridRansacStatistics ransac_stats;

    hybrid_lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

int HybridPoseEstimator::MinimalSolver(const std::vector<std::vector<int>>& sample,
                                       const int solver_idx, std::vector<CameraPose>* poses) const {
    MinimalSolverType minimal_solver = static_cast<MinimalSolverType>(solver_idx);

    int pt_sz, lines_sz;
    switch (minimal_solver) {
    case P3P: pt_sz = 3; lines_sz = 0; break;
    case P2P1LL: pt_sz = 2; lines_sz = 1; break;
    case P1P2LL: pt_sz = 1; lines_sz = 2; break;
    case P3LL: pt_sz = 0; lines_sz = 3; break;
    }

    std::vector<int> samples;
    for (int i = 0; i < pt_sz; i++)
        samples.push_back(sample[0][i]);
    for (int i = 0; i < lines_sz; i++)
        samples.push_back(num_data_points() + sample[1][i]);
    return JointPoseEstimator::MinimalSolver(samples, poses);
}

double HybridPoseEstimator::EvaluateModelOnPoint(const CameraPose& pose, int t, int i) const {
    if (t == 0) 
        return JointPoseEstimator::EvaluateModelOnPoint(pose, i);
    else if (t == 1)
        return JointPoseEstimator::EvaluateModelOnPoint(pose, i + num_data_points());
}

} // namespace pose

} // namespace estimators

} // namespace limap
