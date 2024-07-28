#ifndef LIMAP_BASE_POINTTRACK_H_
#define LIMAP_BASE_POINTTRACK_H_

#include <cmath>
#include <map>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>

namespace py = pybind11;

#include "_limap/helpers.h"
#include "util/types.h"

namespace limap {

template <typename PTYPE> struct Feature2dWith3dIndex {
  Feature2dWith3dIndex() {}
  Feature2dWith3dIndex(PTYPE p_, int point3D_id_ = -1)
      : p(p_), point3D_id(point3D_id_) {}
  Feature2dWith3dIndex(py::dict dict){
      ASSIGN_PYDICT_ITEM(dict, p, PTYPE)
          ASSIGN_PYDICT_ITEM(dict, point3D_id, int)} py::dict as_dict() const {
    py::dict output;
    output["p"] = p;
    output["point3D_id"] = point3D_id;
    return output;
  }
  PTYPE p;
  int point3D_id = -1;
};
typedef Feature2dWith3dIndex<V2D> Point2d;

class PointTrack {
public:
  PointTrack() {}
  PointTrack(const PointTrack &track);
  PointTrack(const V3D &p_, const std::vector<int> &image_id_list_,
             const std::vector<int> &p2d_id_list_,
             const std::vector<V2D> p2d_list_)
      : p(p_), image_id_list(image_id_list_), p2d_id_list(p2d_id_list_),
        p2d_list(p2d_list_) {}
  py::dict as_dict() const;
  PointTrack(py::dict dict);

  V3D p;
  std::vector<int> image_id_list;
  std::vector<int> p2d_id_list;
  std::vector<V2D> p2d_list;

  size_t count_images() const { return image_id_list.size(); }
};

} // namespace limap

#endif
