#include "vplib/vptrack.h"

namespace limap {

namespace vplib {

py::dict VPTrack::as_dict() const {
    py::dict output;
    output["direction"] = direction;
    output["supports"] = supports;
    return output;
}

VPTrack::VPTrack(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, direction, V3D)
    ASSIGN_PYDICT_ITEM(dict, supports, std::vector<Node2d>)
}

} // namespace vplib

} // namespace limap


