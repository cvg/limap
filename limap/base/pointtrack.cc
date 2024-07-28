#include "base/pointtrack.h"

#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>

namespace limap {

PointTrack::PointTrack(const PointTrack &track) {
  size_t n_supports = track.p2d_id_list.size();
  p = track.p;
  std::copy(track.image_id_list.begin(), track.image_id_list.end(),
            std::back_inserter(image_id_list));
  std::copy(track.p2d_id_list.begin(), track.p2d_id_list.end(),
            std::back_inserter(p2d_id_list));
  std::copy(track.p2d_list.begin(), track.p2d_list.end(),
            std::back_inserter(p2d_list));
}

py::dict PointTrack::as_dict() const {
  py::dict output;
  output["p"] = p;
  output["image_id_list"] = image_id_list;
  output["p2d_id_list"] = p2d_id_list;
  output["p2d_list"] = p2d_list;
  return output;
}

PointTrack::PointTrack(py::dict dict) {
  ASSIGN_PYDICT_ITEM(dict, p, V3D)
  ASSIGN_PYDICT_ITEM(dict, image_id_list, std::vector<int>)
  ASSIGN_PYDICT_ITEM(dict, p2d_id_list, std::vector<int>)
  ASSIGN_PYDICT_ITEM(dict, p2d_list, std::vector<V2D>)
}

} // namespace limap
