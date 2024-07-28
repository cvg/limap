#ifndef LIMAP_FITTING_LINE_ESTIMATOR_H_
#define LIMAP_FITTING_LINE_ESTIMATOR_H_

#include "_limap/helpers.h"
#include "base/infinite_line.h"
#include "base/linebase.h"
#include "util/types.h"

#include <RansacLib/ransac.h>

namespace limap {

namespace fitting {

// Implements a simple solver that estimates a 3D line from two data points.
// Reference link:
// https://github.com/B1ueber2y/RansacLib/blob/master/examples/line_estimator.h
class Line3dEstimator {
public:
  Line3dEstimator(const Eigen::Matrix3Xd &data);

  inline int min_sample_size() const { return 2; }

  inline int non_minimal_sample_size() const { return 6; }

  inline int num_data() const { return num_data_; }

  int MinimalSolver(const std::vector<int> &sample,
                    std::vector<InfiniteLine3d> *lines) const;

  // Returns 0 if no model could be estimated and 1 otherwise.
  // Implemented by a simple linear least squares solver.
  int NonMinimalSolver(const std::vector<int> &sample,
                       InfiniteLine3d *line) const;

  // Evaluates the line on the i-th data point.
  double EvaluateModelOnPoint(const InfiniteLine3d &line, int i) const;

  // Linear least squares solver. Calls NonMinimalSolver.
  inline void LeastSquares(const std::vector<int> &sample,
                           InfiniteLine3d *line) const {
    NonMinimalSolver(sample, line);
  }

protected:
  // Matrix holding the 3D points through which the line is fitted.
  Eigen::Matrix3Xd data_;
  int num_data_;
};

std::pair<Line3d, ransac_lib::RansacStatistics>
Fit3DPoints(const Eigen::Matrix3Xd points,
            const ransac_lib::LORansacOptions &options);

} // namespace fitting

} // namespace limap

#endif
