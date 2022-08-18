#include "vplib/base_vp_detector.h"
#include "base/infinite_line.h"

#include <third-party/progressbar.hpp>

namespace limap {

namespace vplib {

std::map<int, VPResult> BaseVPDetector::AssociateVPsParallel(const std::map<int, std::vector<Line2d>>& all_lines) const {
    std::vector<int> image_ids;
    for (std::map<int, std::vector<Line2d>>::const_iterator it = all_lines.begin(); it != all_lines.end(); ++it) {
        image_ids.push_back(it->first);
    }
    
    std::map<int, VPResult> vpresults;
    progressbar bar(image_ids.size());
#pragma omp parallel for
    for (const int& img_id: image_ids) {
        bar.update();
        vpresults.insert(std::make_pair(img_id, AssociateVPs(all_lines.at(img_id))));
    }
    return vpresults;
}

} // namespace vplib

} // namespace limap

