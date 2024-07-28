#ifndef LIMAP_VPLIB_GLOBAL_VPTRACK_CONSTRUCTOR_H_
#define LIMAP_VPLIB_GLOBAL_VPTRACK_CONSTRUCTOR_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "base/image_collection.h"
#include "base/linetrack.h"
#include "vplib/vpbase.h"
#include "vplib/vptrack.h"

namespace limap {

namespace vplib {

class GlobalVPTrackConstructorConfig {
public:
  GlobalVPTrackConstructorConfig() {}
  GlobalVPTrackConstructorConfig(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, min_common_lines, int)
    ASSIGN_PYDICT_ITEM(dict, th_angle_verify, double)
    ASSIGN_PYDICT_ITEM(dict, min_track_length, int)
  }
  int min_common_lines = 3;
  double th_angle_verify = 10.0; // in degree, verify edge with poses
  int min_track_length = 5;
};

class GlobalVPTrackConstructor {
public:
  GlobalVPTrackConstructor() {}
  GlobalVPTrackConstructor(const GlobalVPTrackConstructorConfig &config)
      : config_(config) {}
  GlobalVPTrackConstructor(py::dict dict)
      : config_(GlobalVPTrackConstructorConfig(dict)) {}

  void Init(const std::map<int, VPResult> &vpresults) {
    vpresults_ = vpresults;
  }

  std::vector<VPTrack>
  ClusterLineTracks(const std::vector<LineTrack> &linetracks,
                    const ImageCollection &imagecols) const;

private:
  GlobalVPTrackConstructorConfig config_;
  std::map<int, VPResult> vpresults_;
};

} // namespace vplib

} // namespace limap

#endif
