#include "limap/image/groups/vplib/vpbase.h"

#include <colmap/util/logging.h>

namespace limap {
namespace image {
namespace groups {

namespace vplib {

VPResult::VPResult() {}

VPResult::VPResult(const std::vector<int> &labels_,
                   const std::vector<V3D> &vps_)
    : labels(labels_), vps(vps_) {}

VPResult::VPResult(const VPResult &input)
    : labels(input.labels), vps(input.vps) {}

size_t VPResult::CountLines() const { return labels.size(); }

size_t VPResult::CountVPs() const { return vps.size(); }

int VPResult::GetVPLabel(const int &line_id) const { return labels[line_id]; }

V3D VPResult::GetVPParams(const int &vp_id) const { return vps[vp_id]; }

bool VPResult::HasVP(const int &line_id) const {
  return GetVPLabel(line_id) >= 0;
}

V3D VPResult::GetVP(const int &line_id) const {
  THROW_CHECK_EQ(HasVP(line_id), true);
  return GetVPParams(GetVPLabel(line_id));
}

} // namespace vplib

} // namespace groups
} // namespace image
} // namespace limap
