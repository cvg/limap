#pragma once

#include <map>

#include "limap/geometry/line2d.h"
#include "limap/image/groups/vplib/vpbase.h"
#include "limap/util/eigen_types.h"

namespace limap {
namespace image {
namespace groups {

namespace vplib {

class BaseVPDetectorOptions {
public:
  BaseVPDetectorOptions() {}

  double min_length = 40;        // in pixel
  double inlier_threshold = 1.0; // in pixel
  int min_num_supports = 5;
  int n_jobs = -1;
};

class BaseVPDetector {
public:
  BaseVPDetector() {}
  BaseVPDetector(const BaseVPDetectorOptions &options)
      : base_options_(options) {}

  virtual VPResult AssociateVPs(const std::vector<Line2d> &lines) const = 0;
  std::map<int, VPResult> AssociateVPsParallel(
      const std::map<int, std::vector<Line2d>> &all_lines) const;

protected:
  int CountValidSupports2D(const std::vector<Line2d> &lines,
                           const double th_perp_supports)
      const; // count supports that lie on the different infinite 2d lines
  BaseVPDetectorOptions base_options_;
};

} // namespace vplib

} // namespace groups
} // namespace image
} // namespace limap
