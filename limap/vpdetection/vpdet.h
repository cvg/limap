#ifndef LIMAP_VPDETECTION_VPDET_H_
#define LIMAP_VPDETECTION_VPDET_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <VPCluster.h>
#include <VPSample.h>

#include "_limap/helpers.h"
#include "base/linebase.h"
#include "util/types.h"

namespace py = pybind11;

namespace limap {

namespace vpdetection {

class VPResult {
public:
    VPResult() {}
    VPResult(const std::vector<int>& labels_, const std::vector<V3D>& vps_): labels(labels_), vps(vps_) {}
    VPResult(const VPResult& input): labels(input.labels), vps(input.vps) {}

    std::vector<int> labels;
    std::vector<V3D> vps;

    size_t count_lines() const { return labels.size(); }
    size_t count_vps() const { return vps.size(); }
    int GetVPLabel(const int& line_id) const { return labels[line_id]; }
    V3D GetVPbyCluster(const int& vp_id) const { return vps[vp_id]; }
    bool HasVP(const int& line_id) const { return GetVPLabel(line_id) >= 0; }
    V3D GetVP(const int& line_id) const { if (HasVP(line_id)) return GetVPbyCluster(GetVPLabel(line_id)); else return V3D(0., 0., 0.); }
};

class VPDetectorConfig {
public:
    VPDetectorConfig() {}
    VPDetectorConfig(py::dict dict) {
        ASSIGN_PYDICT_ITEM(dict, min_length, double)
        ASSIGN_PYDICT_ITEM(dict, inlier_threshold, double)
        ASSIGN_PYDICT_ITEM(dict, min_num_supports, int)
    }

    double min_length = 40; // in pixel
    double inlier_threshold = 1.0;
    int min_num_supports = 10;
};

class VPDetector {
public:
    VPDetector() {}
    VPDetector(const VPDetectorConfig& config): config_(config) {}
    VPDetector(py::dict dict): config_(VPDetectorConfig(dict)) {}
    VPDetectorConfig config_;

    std::vector<int> ComputeVPLabels(const std::vector<Line2d>& lines) const; // cluster id for each line, -1 for no associated vp
    VPResult AssociateVPs(const std::vector<Line2d>& lines) const;
    std::map<int, VPResult> AssociateVPsParallel(const std::map<int, std::vector<Line2d>>& all_lines) const;

private:
    V3D fitVP(const std::vector<Line2d>& lines) const;
};

} // namespace vpdetection

} // namespace limap

#endif

