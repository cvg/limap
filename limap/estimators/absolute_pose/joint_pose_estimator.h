#ifndef LIMAP_ESTIMATORS_POSE_JOINT_POSE_ESTIMATOR_H_
#define LIMAP_ESTIMATORS_POSE_JOINT_POSE_ESTIMATOR_H_

#include "_limap/helpers.h"
#include "util/types.h"
#include "base/linetrack.h"
#include "base/camera.h"
#include "optimize/line_localization/lineloc.h"

#include <RansacLib/ransac.h>

namespace limap {

using namespace optimize::line_localization;

namespace estimators {

namespace absolute_pose {

class JointPoseEstimator {
public:
    JointPoseEstimator(const std::vector<Line3d>& l3ds, const std::vector<int>& l3d_ids, const std::vector<Line2d>& l2ds, 
                       const std::vector<V3D>& p3ds, const std::vector<V2D>& p2ds, 
                       const Camera& cam, const LineLocConfig& cfg,
                       const double cheirality_min_depth = 0.0, 
                       const double line_min_projected_length = 1);

    inline int min_sample_size() const { return 3; }

    inline int non_minimal_sample_size() const { return 6; }

    inline int num_data() const { return num_data_; }

    inline int num_data_points() const { return p2ds_->size(); }

    inline int num_data_lines() const { return l3d_ids_->size(); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<CameraPose>* poses) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, CameraPose* pose) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const CameraPose& pose, int i) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<int>& sample, CameraPose* pose) const;

protected:
    const std::vector<Line3d>* l3ds_;
    const std::vector<int>* l3d_ids_;
    const std::vector<Line2d>* l2ds_;
    const std::vector<V3D>* p3ds_;
    const std::vector<V2D>* p2ds_;
    int num_data_;
    LineLocConfig loc_config_;
    
    // Camera intrinsics
    Camera cam_;

    // Cheirality and filtering options
    double cheirality_min_depth_;
    double line_min_projected_length_;

private:
    bool cheirality_test_point(const V3D& p3d, const CameraPose& pose) const;
    bool cheirality_test_line(const Line2d& l2d, const Line3d& l3d, const CameraPose& pose) const;
};

std::pair<CameraPose, ransac_lib::RansacStatistics> 
EstimateAbsolutePose_PointLine(const std::vector<Line3d>& l3ds, const std::vector<int>& l3d_ids, 
                               const std::vector<Line2d>& l2ds, const std::vector<V3D>& p3ds, 
                               const std::vector<V2D>& p2ds, const Camera& cam, 
                               const ransac_lib::LORansacOptions& options_ = ransac_lib::LORansacOptions(),
                               const LineLocConfig& cfg = LineLocConfig(), bool sample_solver_first = true,
                               const double cheirality_min_depth = 0.0, const double line_min_projected_length = 1);

} // namespace pose

} // namespace estimators

} // namespace limap

#endif

