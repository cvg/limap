#include "vpdetection/vpdet.h"
#include "base/infinite_line.h"

#include <third-party/progressbar.hpp>

namespace limap {

namespace vpdetection {

std::vector<int> VPDetector::ComputeVPLabels(const std::vector<Line2d>& lines) const {
    // filter line
    size_t n_lines = lines.size();
    std::vector<int> final_labels(n_lines, -1);
    std::vector<int> valid_ids;
    for (int i = 0; i < n_lines; ++i) {
        if (lines[i].length() < config_.min_length)
            continue;
        valid_ids.push_back(i);
    }
    size_t n_valid_lines = valid_ids.size();

    // initialize pts
    std::vector<std::vector<float> *> pts;
    for (auto it = valid_ids.begin(); it != valid_ids.end(); ++it) {
        const auto& line = lines[*it];
        std::vector<float>* p = new std::vector<float>(4);
		pts.push_back(p);
		(*p)[0]=(float)line.start[0];
		(*p)[1]=(float)line.start[1];
		(*p)[2]=(float)line.end[0];
		(*p)[3]=(float)line.end[1];
    }
    if (pts.size() < 2 * std::max(config_.min_num_supports, 10))
        return final_labels;

    // compute labels
    std::vector<unsigned int> Labels;
	std::vector<unsigned int> LabelCount;
    std::vector<std::vector<float> *> *mModels = 
		VPSample::run(&pts, 5000, 2, 0, 3);
	int classNum = VPCluster::run(Labels, LabelCount, &pts, mModels, config_.inlier_threshold, 2);

    // release memory
    for (size_t i = 0; i < mModels->size(); ++i)
        delete (*mModels)[i];
    delete mModels;
    for (size_t i = 0; i < pts.size(); ++i)
        delete pts[i];

    // determine valid labels
    int counter = 0;
    while (counter < LabelCount.size()) {
        if (LabelCount[counter] < config_.min_num_supports)
            break;
        counter++;
    }
    int n_vps = counter;
    for (int i = 0; i < n_valid_lines; ++i) {
        int label = Labels[i];
        if (label >= n_vps) 
            continue;
        final_labels[valid_ids[i]] = label;
    }
    return final_labels;
}

V3D VPDetector::fitVP(const std::vector<Line2d>& lines) const {
    int n_lines = lines.size();
    Eigen::MatrixXd A(lines.size(), 3);
    for (int i = 0; i < n_lines; ++i) {
        const auto& line = lines[i];
        V3D coor = InfiniteLine2d(line).GetLineCoordinate();
        A.row(i) = coor;
    }

    // svd
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    V3D p_homo = svd.matrixV().col(2).normalized();
    return p_homo;
}

VPResult VPDetector::AssociateVPs(const std::vector<Line2d>& lines) const
{
    std::vector<int> labels;
    std::vector<V3D> vps;

    // compute vp labels
    size_t n_lines = lines.size();
    if (n_lines == 0)
        return VPResult(labels, vps);
    labels = ComputeVPLabels(lines);
    int n_vps = *std::max_element(labels.begin(), labels.end()) + 1;
    if (n_vps == 0)
        return VPResult(labels, vps);
    vps.resize(n_vps);

    // fit vp
    std::vector<std::vector<Line2d>> supports(n_vps);
    for (size_t i = 0; i < n_lines; ++i) {
        if (labels[i] < 0)
            continue;
        supports[labels[i]].push_back(lines[i]);
    }
    for (size_t i = 0; i < n_vps; ++i) {
        vps[i] = fitVP(supports[i]);
    }
    return VPResult(labels, vps);
}

std::map<int, VPResult> VPDetector::AssociateVPsParallel(const std::map<int, std::vector<Line2d>>& all_lines) const {
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

} // namespace vpdetection

} // namespace limap

