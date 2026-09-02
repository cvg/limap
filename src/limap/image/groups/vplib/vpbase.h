#pragma once

#include <vector>

#include "limap/util/eigen_types.h"

namespace limap {
namespace image {
namespace groups {

namespace vplib {

class VPResult {
public:
  VPResult();
  VPResult(const std::vector<int> &labels_, const std::vector<V3D> &vps_);
  VPResult(const VPResult &input);

  size_t CountLines() const;
  size_t CountVPs() const;
  int GetVPLabel(const int &line_id) const;
  V3D GetVPParams(const int &vp_id) const;
  bool HasVP(const int &line_id) const;
  V3D GetVP(const int &line_id) const;

  std::vector<int> labels; // -1 denotes the unassociated lines
  std::vector<V3D> vps;
};

} // namespace vplib

} // namespace groups
} // namespace image
} // namespace limap
