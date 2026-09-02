#pragma once

#include "limap/image/groups/vplib/vpbase.h"
#include "limap/scene/group2d.h"
#include <colmap/util/types.h>

namespace limap {
namespace image {
namespace groups {
namespace vplib {

std::vector<Group2d> ConvertVPResultToGroups2d(const vplib::VPResult &vpresult);

FlatHashMap<colmap::image_t, std::vector<Group2d>> ConvertVPResultsToGroups2d(
    const FlatHashMap<colmap::image_t, vplib::VPResult> vpresults);

} // namespace vplib
} // namespace groups
} // namespace image
} // namespace limap
