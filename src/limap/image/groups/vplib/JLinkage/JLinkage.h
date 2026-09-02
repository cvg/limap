#pragma once

#include "limap/image/groups/vplib/base_vp_detector.h"

namespace limap {
namespace image {
namespace groups {

namespace vplib {

namespace JLinkage {

struct JLinkageOptions {
  BaseVPDetectorOptions base_options;
  double th_perp_supports = 3.0; // in pixels
};

class JLinkage : public BaseVPDetector {
public:
  JLinkage() : BaseVPDetector() {}
  JLinkage(const JLinkageOptions &options)
      : BaseVPDetector(options.base_options), options_(options) {}

  std::vector<int> ComputeVPLabels(const std::vector<Line2d> &lines)
      const; // cluster id for each line, -1 for no associated vp
  VPResult AssociateVPs(const std::vector<Line2d> &lines) const;

private:
  V3D fitVP(const std::vector<Line2d> &lines) const;
  JLinkageOptions options_;
};

} // namespace JLinkage

} // namespace vplib

} // namespace groups
} // namespace image
} // namespace limap
