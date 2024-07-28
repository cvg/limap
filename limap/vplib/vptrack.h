#ifndef LIMAP_VPLIB_VPTRACK_H_
#define LIMAP_VPLIB_VPTRACK_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_limap/helpers.h"
#include "util/types.h"

#include "base/pointtrack.h"

namespace limap {

namespace vplib {

typedef Feature2dWith3dIndex<V3D> VP2d;

class VPTrack {
public:
  VPTrack() {}
  VPTrack(const V3D &direction_, const std::vector<Node2d> &supports_)
      : direction(direction_), supports(supports_) {}
  VPTrack(const VPTrack &obj)
      : direction(obj.direction), supports(obj.supports) {}
  VPTrack(py::dict dict);
  py::dict as_dict() const;

  V3D direction;
  std::vector<Node2d> supports;
  size_t length() const { return supports.size(); }
};

std::vector<VPTrack>
MergeVPTracksByDirection(const std::vector<VPTrack> &vptracks,
                         const double th_angle_merge = 1.0);

} // namespace vplib

} // namespace limap

#endif
