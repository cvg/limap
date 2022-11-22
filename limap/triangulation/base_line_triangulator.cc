#include "triangulation/base_line_triangulator.h"
#include "triangulation/functions.h"

#include <third-party/progressbar.hpp>
#include <iostream>
#include <algorithm>

namespace limap {

namespace triangulation {

void BaseLineTriangulator::offsetHalfPixel() {
    std::vector<int> image_ids = imagecols_.get_img_ids();
    for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
        int img_id = *it;
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            auto& line = all_lines_2d_[img_id][line_id];
            line.start = line.start + V2D(0.5, 0.5); 
            line.end = line.end + V2D(0.5, 0.5); 
        }
    }
}

void BaseLineTriangulator::Init(const std::map<int, std::vector<Line2d>>& all_2d_segs,
                            const ImageCollection& imagecols) 
{
    all_lines_2d_ = all_2d_segs;
    if (config_.add_halfpix)
        offsetHalfPixel();
    THROW_CHECK_EQ(imagecols.IsUndistorted(), true);
    imagecols_ = imagecols;
    
    // compute vanishing points (optional)
    if (config_.use_vp) {
        for (const int& img_id: imagecols.get_img_ids()) {
            vpresults_.insert(std::make_pair(img_id, vplib::VPResult()));
        }
        std::cout<<"Start vanishing point detection..."<<std::endl;
        progressbar bar(imagecols.NumImages());
#pragma omp parallel for
        for (const int& img_id: imagecols.get_img_ids()) {
            bar.update();
            vpresults_.at(img_id) = vpdetector_.AssociateVPs(all_2d_segs.at(img_id));
        }
    }

    // initialize empty containers
    for (const int& img_id: imagecols.get_img_ids()) {
        size_t n_lines = all_2d_segs.at(img_id).size();
        neighbors_.insert(std::make_pair(img_id, std::vector<int>()));
        edges_.insert(std::make_pair(img_id, std::vector<std::vector<LineNode>>()));
        edges_.at(img_id).resize(n_lines);
        tris_.insert(std::make_pair(img_id, std::vector<std::vector<TriTuple>>()));
        tris_.at(img_id).resize(n_lines);
    }
}

void BaseLineTriangulator::TriangulateImage(const int img_id,
                                        const std::vector<Eigen::MatrixXi>& matches,
                                        const std::vector<int>& neighbors) 
{
    neighbors_[img_id].clear(); neighbors_[img_id] = neighbors;
    size_t n_neighbors = neighbors.size();
    for (size_t neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id) {
        int ng_img_id = neighbors[neighbor_id];
        const auto& match_info = matches[neighbor_id];
        if (match_info.rows() != 0) {
            THROW_CHECK_EQ(match_info.cols(), 2);
        }
        size_t n_matches = match_info.rows();
        std::vector<size_t> matches;
        for (int k = 0; k < n_matches; ++k) {
            // good match exists
            int line_id = match_info(k, 0);
            int ng_line_id = match_info(k, 1);
            edges_[img_id][line_id].push_back(std::make_pair(ng_img_id, ng_line_id));
        }
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            triangulateOneNode(img_id, line_id);
            clearEdgesOneNode(img_id, line_id);
        }
        edges_[img_id].clear(); edges_[img_id].resize(CountLines(img_id));
    }
    ScoringCallback(img_id);
    if (!config_.debug_mode) {
        tris_[img_id].clear(); tris_[img_id].resize(CountLines(img_id));
    }
}

void BaseLineTriangulator::TriangulateImageExhaustiveMatch(const int img_id,
                                                       const std::vector<int>& neighbors)
{
    neighbors_[img_id].clear(); neighbors_[img_id] = neighbors;
    size_t n_neighbors = neighbors.size();
    for (size_t neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id) {
        int ng_img_id = neighbors[neighbor_id];
        int n_lines_ng = all_lines_2d_[ng_img_id].size();
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            for (int ng_line_id = 0; ng_line_id < n_lines_ng; ++ng_line_id) {
                edges_[img_id][line_id].push_back(std::make_pair(ng_img_id, ng_line_id));
            }
            triangulateOneNode(img_id, line_id);
            clearEdgesOneNode(img_id, line_id);
        }
        edges_[img_id].clear(); edges_[img_id].resize(CountLines(img_id));
    }
    ScoringCallback(img_id);
    if (!config_.debug_mode) {
        tris_[img_id].clear(); tris_[img_id].resize(CountLines(img_id));
    }
}

void BaseLineTriangulator::clearEdgesOneNode(const int img_id, const int line_id) {
    edges_[img_id][line_id].clear();
}

void BaseLineTriangulator::clearEdges() {
    for (const int& img_id: imagecols_.get_img_ids()) {
        for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
            clearEdgesOneNode(img_id, line_id);
        }
    }
}

int BaseLineTriangulator::countEdges() const {
    int counter = 0;
    for (const int& img_id: imagecols_.get_img_ids()) {
        for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
            counter += edges_.at(img_id)[line_id].size();
        }
    }
    return counter;
}

void BaseLineTriangulator::triangulateOneNode(const int img_id, const int line_id) {
    THROW_CHECK_EQ(imagecols_.exist_image(img_id), true);
    auto& connections = edges_[img_id][line_id];
    const Line2d& l1 = all_lines_2d_[img_id][line_id];
    if (l1.length() <= config_.min_length_2d)
        return;
    const CameraView& view1 = imagecols_.camview(img_id);
    size_t n_conns = connections.size();
    std::vector<std::vector<TriTuple>> results(n_conns);

#pragma omp parallel for
    for (size_t conn_id = 0; conn_id < n_conns; ++conn_id) {
        int ng_img_id = connections[conn_id].first;
        int ng_line_id = connections[conn_id].second;
        const Line2d& l2 = all_lines_2d_[ng_img_id][ng_line_id];
        if (l2.length() <= config_.min_length_2d)
            continue;
        const CameraView& view2 = imagecols_.camview(ng_img_id);

        // test degeneracy by ray-plane angles
        V3D n2 = getNormalDirection(l2, view2);
        V3D ray1_start = view1.ray_direction(l1.start);
        double angle_start = 90 - acos(std::abs(n2.dot(ray1_start))) * 180.0 / M_PI;
        if (angle_start < config_.line_tri_angle_threshold)
            continue;
        V3D ray1_end = view1.ray_direction(l1.end);
        double angle_end = 90 - acos(std::abs(n2.dot(ray1_end))) * 180.0 / M_PI;
        if (angle_end < config_.line_tri_angle_threshold)
            continue;

        // test weak epipolar constraints
        double IoU = compute_epipolar_IoU(l1, view1, l2, view2);
        if (IoU < config_.IoU_threshold)
            continue;

        // triangulation with weak epipolar constraints test
        Line3d line;
        if (!config_.use_endpoints_triangulation)
            line = triangulate(l1, view1, l2, view2);
        else
            line = triangulate_endpoints(l1, view1, l2, view2);
        if (line.sensitivity(view1) > config_.sensitivity_threshold && line.sensitivity(view2) > config_.sensitivity_threshold)
            line.score = -1;
        if (line.score > 0) {
            double u1 = line.computeUncertainty(view1, config_.var2d);
            double u2 = line.computeUncertainty(view2, config_.var2d);
            line.uncertainty = std::min(u1, u2);
            results[conn_id].push_back(std::make_tuple(line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
        }

        // triangulation with VPs
        if (config_.use_vp) {
            // vp1
            if (vpresults_[img_id].HasVP(line_id)) {
                V3D direc = getDirectionFromVP(vpresults_[img_id].GetVP(line_id), view1);
                Line3d line = triangulate_with_direction(l1, view1, l2, view2, direc);
                if (line.score > 0) {
                    double u1 = line.computeUncertainty(view1, config_.var2d);
                    double u2 = line.computeUncertainty(view2, config_.var2d);
                    line.uncertainty = std::min(u1, u2);
                    results[conn_id].push_back(std::make_tuple(line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
                }
            }
            // vp2
            if (vpresults_[ng_img_id].HasVP(ng_line_id)) {
                V3D direc = getDirectionFromVP(vpresults_[ng_img_id].GetVP(ng_line_id), view1);
                Line3d line = triangulate_with_direction(l1, view1, l2, view2, direc);
                if (line.score > 0) {
                    double u1 = line.computeUncertainty(view1, config_.var2d);
                    double u2 = line.computeUncertainty(view2, config_.var2d);
                    line.uncertainty = std::min(u1, u2);
                    results[conn_id].push_back(std::make_tuple(line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
                }
            }
        }
    }
    for (int conn_id = 0; conn_id < n_conns; ++conn_id) {
        auto& result = results[conn_id];
        for (auto it = result.begin(); it != result.end(); ++it) {
            if (ranges_flag_) {
                if (!test_line_inside_ranges(std::get<0>(*it), ranges_))
                    continue;
            }
            tris_[img_id][line_id].push_back(*it);
        }
    }
}

} // namespace triangulation

} // namespace limap

