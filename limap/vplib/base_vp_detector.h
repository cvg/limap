#ifndef LIMAP_VPLIB_BASE_VP_DETECTOR_H_
#define LIMAP_VPLIB_BASE_VP_DETECTOR_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include "base/linebase.h"
#include "util/types.h"

#include "vplib/vpbase.h"

namespace py = pybind11;

namespace limap {

namespace vplib {

class BaseVPDetectorConfig {
public:
  BaseVPDetectorConfig() {}
  BaseVPDetectorConfig(py::dict dict){
      ASSIGN_PYDICT_ITEM(dict, min_length, double)
          ASSIGN_PYDICT_ITEM(dict, inlier_threshold, double)
              ASSIGN_PYDICT_ITEM(dict, min_num_supports, int)
                  ASSIGN_PYDICT_ITEM(dict, th_perp_supports, double)} py::dict
      as_dict() const;

  double min_length = 40;        // in pixel
  double inlier_threshold = 1.0; // in pixel
  int min_num_supports = 5;
  double th_perp_supports = 3.0; // in pixel. separate different supports
};

class BaseVPDetector {
public:
  BaseVPDetector() {}
  BaseVPDetector(const BaseVPDetectorConfig &config) : config_(config) {}
  BaseVPDetector(py::dict dict) : config_(BaseVPDetectorConfig(dict)) {}
  BaseVPDetectorConfig config_;

  virtual VPResult AssociateVPs(const std::vector<Line2d> &lines) const = 0;
  std::map<int, VPResult> AssociateVPsParallel(
      const std::map<int, std::vector<Line2d>> &all_lines) const;

protected:
  int count_valid_supports_2d(const std::vector<Line2d> &lines)
      const; // count supports that lie on the different infinite 2d lines
};

} // namespace vplib

} // namespace limap

#endif
