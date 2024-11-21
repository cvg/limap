#ifndef LIMAP_ESTIMATORS_POSE_HYBRID_POSE_ESTIMATOR_H_
#define LIMAP_ESTIMATORS_POSE_HYBRID_POSE_ESTIMATOR_H_

#include "_limap/helpers.h"
#include "base/camera.h"
#include "base/linetrack.h"
#include "estimators/absolute_pose/joint_pose_estimator.h"
#include "estimators/extended_hybrid_ransac.h"
#include "optimize/hybrid_localization/hybrid_localization.h"
#include "util/types.h"

#include <RansacLib/ransac.h>

namespace limap {

using namespace optimize::hybrid_localization;

namespace estimators {

namespace absolute_pose {

class HybridPoseEstimatorOptions {
public:
  HybridPoseEstimatorOptions()
      : ransac_options(ExtendedHybridLORansacOptions()),
        lineloc_config(LineLocConfig()), solver_flags({true, true, true, true}),
        cheirality_min_depth(0.0), cheirality_overlap_pixels(10.0),
        random(true) {
    lineloc_config.print_summary = false;
    lineloc_config.solver_options.minimizer_progress_to_stdout = false;
    lineloc_config.solver_options.logging_type = ceres::LoggingType::SILENT;

    // Default values, should be changed depend on the data
    ransac_options.squared_inlier_thresholds_ = {1.0, 1.0};
    ransac_options.data_type_weights_ = {1.0, 1.0};
  }

  ExtendedHybridLORansacOptions ransac_options;
  LineLocConfig lineloc_config;
  std::vector<bool> solver_flags = {true, true, true, true};
  double cheirality_min_depth = 0.0;
  double cheirality_overlap_pixels = 10.0;
  bool random = true;
};

class HybridPoseEstimator : public JointPoseEstimator {
public:
  HybridPoseEstimator(const std::vector<Line3d> &l3ds,
                      const std::vector<int> &l3d_ids,
                      const std::vector<Line2d> &l2ds,
                      const std::vector<V3D> &p3ds,
                      const std::vector<V2D> &p2ds, const Camera &cam,
                      const LineLocConfig &cfg,
                      const double cheirality_min_depth = 0.0,
                      const double cheirality_overlap_pixels = 10.0)
      : JointPoseEstimator(l3ds, l3d_ids, l2ds, p3ds, p2ds, cam, cfg,
                           cheirality_min_depth, cheirality_overlap_pixels) {}

  inline int num_minimal_solvers() const { return 4; }

  void min_sample_sizes(std::vector<std::vector<int>> *min_sample_sizes) const {
    min_sample_sizes->resize(4);
    (*min_sample_sizes)[0] = std::vector<int>{3, 0};
    (*min_sample_sizes)[1] = std::vector<int>{2, 1};
    (*min_sample_sizes)[2] = std::vector<int>{1, 2};
    (*min_sample_sizes)[3] = std::vector<int>{0, 3};
  }

  inline int num_data_types() const { return 2; }

  void num_data(std::vector<int> *num_data) const {
    num_data->resize(2);
    (*num_data)[0] = num_data_points();
    (*num_data)[1] = num_data_lines();
  }

  void solver_probabilities(std::vector<double> *solver_probabilities) const {
    std::vector<std::vector<int>> sample_sizes;
    min_sample_sizes(&sample_sizes);
    solver_probabilities->resize(4);

    for (int i = 0; i < 4; i++) {
      if (!solver_flags_[i])
        solver_probabilities->at(i) = 0.0;
      else {
        solver_probabilities->at(i) =
            combination(num_data_points(), sample_sizes[i][0]) *
            combination(num_data_lines(), sample_sizes[i][1]);
      }
    }
  }

  int NonMinimalSolver(const std::vector<int> &sample, CameraPose *pose) const {
    return JointPoseEstimator::NonMinimalSolver(sample, pose);
  }

  int MinimalSolver(const std::vector<std::vector<int>> &sample,
                    const int solver_idx, std::vector<CameraPose> *poses) const;

  double EvaluateModelOnPoint(const CameraPose &pose, int t, int i) const;

  void LeastSquares(const std::vector<std::vector<int>> &sample,
                    CameraPose *poses) const {
    std::vector<int> samples;
    for (int i = 0; i < sample[0].size(); i++)
      samples.push_back(sample[0][i]);
    for (int i = 0; i < sample[1].size(); i++)
      samples.push_back(num_data_points() + sample[1][i]);
    JointPoseEstimator::LeastSquares(samples, poses);
  }

  void set_solver_flags(const std::vector<bool> &flags) {
    THROW_CHECK_EQ(flags.size(), 4);
    solver_flags_ = flags;
  }

private:
  std::vector<bool> solver_flags_;
  unsigned long long combination(unsigned n, unsigned m) const {
    unsigned long long num = 1;
    unsigned long long denom = 1;
    for (int i = 0; i < m; i++)
      num *= n - i;
    for (int i = 0; i < m; i++)
      denom *= i + 1;
    return num / denom;
  }
};

std::pair<CameraPose, ransac_lib::HybridRansacStatistics>
EstimateAbsolutePose_PointLine_Hybrid(
    const std::vector<Line3d> &l3ds, const std::vector<int> &l3d_ids,
    const std::vector<Line2d> &l2ds, const std::vector<V3D> &p3ds,
    const std::vector<V2D> &p2ds, const Camera &cam,
    const HybridPoseEstimatorOptions &options);

} // namespace absolute_pose

} // namespace estimators

} // namespace limap

#endif
