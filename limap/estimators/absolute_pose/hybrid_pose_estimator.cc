#include "estimators/absolute_pose/hybrid_pose_estimator.h"
#include "estimators/absolute_pose/pl_absolute_pose_hybrid_ransac.h"
#include "estimators/absolute_pose/pl_absolute_pose_ransac.h"
#include "optimize/hybrid_localization/cost_functions.h"

namespace limap {

namespace estimators {

namespace absolute_pose {

namespace hybridloc = optimize::hybrid_localization;

std::pair<CameraPose, ransac_lib::HybridRansacStatistics>
EstimateAbsolutePose_PointLine_Hybrid(
    const std::vector<Line3d> &l3ds, const std::vector<int> &l3d_ids,
    const std::vector<Line2d> &l2ds, const std::vector<V3D> &p3ds,
    const std::vector<V2D> &p2ds, const Camera &cam,
    const HybridPoseEstimatorOptions &options) {
  ExtendedHybridLORansacOptions ransac_options = options.ransac_options;
  std::random_device rand_dev;
  if (options.random)
    ransac_options.random_seed_ = rand_dev();

  THROW_CHECK_EQ(ransac_options.data_type_weights_.size(), 2);
  THROW_CHECK_EQ(ransac_options.squared_inlier_thresholds_.size(), 2);

  HybridPoseEstimator solver(
      l3ds, l3d_ids, l2ds, p3ds, p2ds, cam, options.lineloc_config,
      options.cheirality_min_depth, options.cheirality_overlap_pixels);
  solver.set_solver_flags(options.solver_flags);

  PointLineAbsolutePoseHybridRansac<CameraPose, std::vector<CameraPose>,
                                    HybridPoseEstimator>
      hybrid_lomsac;
  CameraPose best_model;
  ransac_lib::HybridRansacStatistics ransac_stats;

  hybrid_lomsac.EstimateModel(ransac_options, solver, &best_model,
                              &ransac_stats);
  return std::make_pair(best_model, ransac_stats);
}

int HybridPoseEstimator::MinimalSolver(
    const std::vector<std::vector<int>> &sample, const int solver_idx,
    std::vector<CameraPose> *poses) const {
  MinimalSolverType minimal_solver = static_cast<MinimalSolverType>(solver_idx);

  int pt_sz, lines_sz;
  switch (minimal_solver) {
  case P3P:
    pt_sz = 3;
    lines_sz = 0;
    break;
  case P2P1LL:
    pt_sz = 2;
    lines_sz = 1;
    break;
  case P1P2LL:
    pt_sz = 1;
    lines_sz = 2;
    break;
  case P3LL:
    pt_sz = 0;
    lines_sz = 3;
    break;
  }

  std::vector<int> samples;
  for (int i = 0; i < pt_sz; i++)
    samples.push_back(sample[0][i]);
  for (int i = 0; i < lines_sz; i++)
    samples.push_back(num_data_points() + sample[1][i]);
  return JointPoseEstimator::MinimalSolver(samples, poses);
}

double HybridPoseEstimator::EvaluateModelOnPoint(const CameraPose &pose, int t,
                                                 int i) const {
  if (t == 0)
    return JointPoseEstimator::EvaluateModelOnPoint(pose, i);
  else if (t == 1)
    return JointPoseEstimator::EvaluateModelOnPoint(pose,
                                                    i + num_data_points());
  else
    throw std::runtime_error("Error! Not supported!");
}

} // namespace absolute_pose

} // namespace estimators

} // namespace limap
