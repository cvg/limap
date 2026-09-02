#include "limap/image/groups/vplib/convert_to_groups.h"

namespace limap {
namespace image {
namespace groups {
namespace vplib {

std::vector<Group2d>
ConvertVPResultToGroups2d(const vplib::VPResult &vpresult) {
  Group2d group(GroupType::VP);
  std::vector<Group2d> groups;
  for (size_t vp_id = 0; vp_id < vpresult.CountVPs(); ++vp_id) {
    Group2d group(GroupType::VP);
    V3D params = vpresult.GetVPParams(vp_id);
    group.SetParams(std::vector<double>(params.data(), params.data() + 3));
    groups.push_back(group);
  }
  for (size_t line_id = 0; line_id < vpresult.CountLines(); ++line_id) {
    if (!vpresult.HasVP(line_id)) {
      continue;
    }
    int vp_label = vpresult.GetVPLabel(line_id);
    groups[vp_label].AddLine(line_id);
  }
  return groups;
}

FlatHashMap<colmap::image_t, std::vector<Group2d>> ConvertVPResultsToGroups2d(
    const FlatHashMap<colmap::image_t, vplib::VPResult> vpresults) {
  FlatHashMap<colmap::image_t, std::vector<Group2d>> outputs;
  for (const auto &[img_id, vpresult] : vpresults) {
    outputs.emplace(img_id, ConvertVPResultToGroups2d(vpresult));
  }
  return outputs;
}

} // namespace vplib
} // namespace groups
} // namespace image
} // namespace limap
