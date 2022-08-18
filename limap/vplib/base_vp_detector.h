#ifndef LIMAP_VPLIB_BASE_VP_DETECTOR_H_
#define LIMAP_VPLIB_BASE_VP_DETECTOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
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
    BaseVPDetectorConfig(py::dict dict) {
        ASSIGN_PYDICT_ITEM(dict, min_length, double)
        ASSIGN_PYDICT_ITEM(dict, inlier_threshold, double)
        ASSIGN_PYDICT_ITEM(dict, min_num_supports, int)
    }

    double min_length = 40; // in pixel
    double inlier_threshold = 1.0;
    int min_num_supports = 10;
};

class BaseVPDetector {
public:
    BaseVPDetector() {}
    BaseVPDetector(const BaseVPDetectorConfig& config): config_(config) {}
    BaseVPDetector(py::dict dict): config_(BaseVPDetectorConfig(dict)) {}
    BaseVPDetectorConfig config_;

    virtual VPResult AssociateVPs(const std::vector<Line2d>& lines) const = 0;
    std::map<int, VPResult> AssociateVPsParallel(const std::map<int, std::vector<Line2d>>& all_lines) const;
};

} // namespace vplib

} // namespace limap

#endif

