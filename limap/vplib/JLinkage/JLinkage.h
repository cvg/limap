#ifndef LIMAP_VPLIB_JLINKAGE_H_
#define LIMAP_VPLIB_JLINKAGE_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "vplib/base_vp_detector.h"

namespace py = pybind11;

namespace limap {

namespace vplib {

namespace JLinkage {

class JLinkageConfig : public BaseVPDetectorConfig {
public:
  JLinkageConfig() : BaseVPDetectorConfig() {}
  JLinkageConfig(py::dict dict) : BaseVPDetectorConfig(dict) {}
  py::dict as_dict() const { return BaseVPDetectorConfig::as_dict(); }
};

class JLinkage : public BaseVPDetector {
public:
  JLinkage() : BaseVPDetector() {}
  JLinkage(const JLinkageConfig &config) : config_(config) {}
  JLinkage(py::dict dict) : config_(JLinkageConfig(dict)) {}
  py::dict as_dict() const { return config_.as_dict(); };
  JLinkageConfig config_;

  std::vector<int> ComputeVPLabels(const std::vector<Line2d> &lines)
      const; // cluster id for each line, -1 for no associated vp
  VPResult AssociateVPs(const std::vector<Line2d> &lines) const;

private:
  V3D fitVP(const std::vector<Line2d> &lines) const;
};

} // namespace JLinkage

} // namespace vplib

} // namespace limap

#endif
