#ifndef LIMAP_STRUCTURES_VPLINE_BIPARTITE_H
#define LIMAP_STRUCTURES_VPLINE_BIPARTITE_H

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "base/linetrack.h"
#include "structures/pl_bipartite_base.h"
#include "vplib/vpbase.h"
#include "vplib/vptrack.h"

namespace limap {

namespace structures {

typedef Junction<vplib::VP2d> VP_Junction2d;
typedef Junction<vplib::VPTrack> VP_Junction3d;

class VPLine_Bipartite2d : public PL_Bipartite<vplib::VP2d, Line2d> {
public:
  VPLine_Bipartite2d() {}
  ~VPLine_Bipartite2d() {}
  VPLine_Bipartite2d(const VPLine_Bipartite2d &obj)
      : PL_Bipartite<vplib::VP2d, Line2d>(obj) {}
  VPLine_Bipartite2d(py::dict dict);
  py::dict as_dict() const;
};

class VPLine_Bipartite3d : public PL_Bipartite<vplib::VPTrack, LineTrack> {
public:
  VPLine_Bipartite3d() {}
  ~VPLine_Bipartite3d() {}
  VPLine_Bipartite3d(const VPLine_Bipartite3d &obj)
      : PL_Bipartite<vplib::VPTrack, LineTrack>(obj) {}
  VPLine_Bipartite3d(py::dict dict);
  py::dict as_dict() const;
};

std::map<int, VPLine_Bipartite2d> GetAllBipartites_VPLine2d(
    const std::map<int, std::vector<Line2d>> &all_2d_lines,
    const std::map<int, vplib::VPResult> &vpresults,
    const std::vector<vplib::VPTrack> &vptracks);

} // namespace structures

} // namespace limap

#endif
