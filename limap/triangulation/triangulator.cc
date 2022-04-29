#include "triangulation/triangulator.h"
#include "triangulation/functions.h"

#include "merging/aggregator.h"
#include "merging/merging.h"

#include <third-party/progressbar.hpp>
#include <queue>
#include <iostream>
#include <algorithm>

namespace limap {

namespace triangulation {

void Triangulator::offsetHalfPixel() {
    for (size_t img_id = 0; img_id < CountImages(); ++img_id) {
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            auto& line = all_lines_2d_[img_id][line_id];
            line.start = line.start + V2D(0.5, 0.5); 
            line.end = line.end + V2D(0.5, 0.5); 
        }
    }
}

void Triangulator::Init(const std::vector<std::vector<Line2d>>& all_2d_segs,
                        const ImageCollection& imagecols) 
{
    all_lines_2d_ = all_2d_segs;
    if (config_.add_halfpix)
        offsetHalfPixel();
    imagecols_ = imagecols;
    
    // compute vanishing points (optional)
    if (config_.use_vp) {
        size_t n_images = all_2d_segs.size();
        vpresults_.resize(n_images);
        std::cout<<"Start vanishing point detection..."<<std::endl;
        progressbar bar(n_images);
#pragma omp parallel for
        for (size_t img_id = 0; img_id < n_images; ++img_id) {
            bar.update();
            vpresults_[img_id] = vpdetector_.AssociateVPs(all_2d_segs[img_id]);
        }
    }
    
    // initialize empty containers
    size_t n_images = all_2d_segs.size();
    neighbors_.resize(n_images);
    edges_.resize(n_images);
    tris_.resize(n_images);
    valid_tris_.resize(n_images);
    valid_edges_.resize(n_images);
    tris_best_.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = all_2d_segs[img_id].size();
        edges_[img_id].resize(n_lines);
        tris_[img_id].resize(n_lines);
        valid_tris_[img_id].resize(n_lines);
        valid_edges_[img_id].resize(n_lines);
        tris_best_[img_id].resize(n_lines);
    }

    // flags for monitoring the scoring process
    already_scored_.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = all_2d_segs[img_id].size();
        already_scored_[img_id].resize(n_lines);
        std::fill(already_scored_[img_id].begin(), already_scored_[img_id].end(), false);
    }
}

void Triangulator::InitMatches(const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                               const std::vector<std::vector<int>>& all_neighbors,
                               bool use_triangulate,
                               bool use_scoring) 
{
    // collect edges
    int n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        neighbors_[img_id].clear(); neighbors_[img_id] = all_neighbors[img_id];
        size_t n_neighbors = all_neighbors[img_id].size();
        for (size_t neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id) {
            int ng_img_id = all_neighbors[img_id][neighbor_id];
            const auto& match_info = all_matches[img_id][neighbor_id];
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
        }
        if (use_triangulate) {
            size_t n_lines = CountLines(img_id);
            for (size_t line_id = 0; line_id < n_lines; ++line_id) {
                triangulateOneNode(img_id, line_id);
                clearEdgesOneNode(img_id, line_id);
            }
            edges_[img_id].clear(); edges_[img_id].resize(CountLines(img_id));
        }
        if (use_scoring) {
            LineLinker linker_scoring = linker_;
            linker_scoring.linker_3d.config.set_to_shared_parent_scoring();
            for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
                scoreOneNode(img_id, line_id, linker_scoring);
            }
            if (!config_.debug_mode) {
                tris_[img_id].clear(); tris_[img_id].resize(CountLines(img_id));
                valid_tris_[img_id].clear(); valid_tris_[img_id].resize(CountLines(img_id));
            }
        }
    }
}

void Triangulator::InitMatchImage(const int img_id,
                                  const std::vector<Eigen::MatrixXi>& matches,
                                  const std::vector<int>& neighbors,
                                  bool use_triangulate,
                                  bool use_scoring) 
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
        if (use_triangulate) {
            for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
                triangulateOneNode(img_id, line_id);
                clearEdgesOneNode(img_id, line_id);
            }
            edges_[img_id].clear(); edges_[img_id].resize(CountLines(img_id));
        }
    }
    if (use_scoring) {
        LineLinker linker_scoring = linker_;
        linker_scoring.linker_3d.config.set_to_shared_parent_scoring();
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            scoreOneNode(img_id, line_id, linker_scoring);
        }
        if (!config_.debug_mode) {
            tris_[img_id].clear(); tris_[img_id].resize(CountLines(img_id));
            valid_tris_[img_id].clear(); valid_tris_[img_id].resize(CountLines(img_id));
        }
    }
}

void Triangulator::InitExhaustiveMatchImage(const int img_id,
                                            const std::vector<int>& neighbors,
                                            bool use_scoring)
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
    if (use_scoring) {
        LineLinker linker_scoring = linker_;
        linker_scoring.linker_3d.config.set_to_shared_parent_scoring();
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            scoreOneNode(img_id, line_id, linker_scoring);
        }
        if (!config_.debug_mode) {
            tris_[img_id].clear(); tris_[img_id].resize(CountLines(img_id));
            valid_tris_[img_id].clear(); valid_tris_[img_id].resize(CountLines(img_id));
        }
    }
}

void Triangulator::InitAll(const std::vector<std::vector<Line2d>>& all_2d_segs,
                           const ImageCollection& imagecols,
                           const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                           const std::vector<std::vector<int>>& all_neighbors,
                           bool use_triangulate,
                           bool use_scoring)
{
    Init(all_2d_segs, imagecols);
    InitMatches(all_matches, all_neighbors, use_triangulate, use_scoring);
}

void Triangulator::clearEdgesOneNode(const int img_id, const int line_id) {
    edges_[img_id][line_id].clear();
}

void Triangulator::clearEdges() {
    for (int img_id = 0; img_id < CountImages(); ++img_id) {
        for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
            clearEdgesOneNode(img_id, line_id);
        }
    }
}

int Triangulator::countEdges() const {
    int counter = 0;
    for (int img_id = 0; img_id < CountImages(); ++img_id) {
        for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
            counter += edges_[img_id][line_id].size();
        }
    }
    return counter;
}

void Triangulator::triangulateOneNode(const int img_id, const int line_id) {
    auto& connections = edges_[img_id][line_id];
    const Line2d& l1 = all_lines_2d_[img_id][line_id];
    const CameraView& view1 = imagecols_.camview(img_id);
    int n_conns = connections.size();
    std::vector<std::vector<TriTuple>> results(n_conns);

#pragma omp parallel for
    for (int conn_id = 0; conn_id < n_conns; ++conn_id) {
        int ng_img_id = connections[conn_id].first;
        int ng_line_id = connections[conn_id].second;
        const Line2d& l2 = all_lines_2d_[ng_img_id][ng_line_id];
        const CameraView& view2 = imagecols_.camview(ng_img_id);

        // test degeneracy by plane angle
        V3D n1 = getNormalDirection(l1, view1);
        V3D n2 = getNormalDirection(l2, view2);
        double plane_angle = acos(std::abs(n1.dot(n2))) * 180.0 / M_PI;
        if (plane_angle < config_.plane_angle_threshold)
            continue;

        // test weak epipolar constraints
        double IoU = compute_epipolar_IoU(l1, view1, l2, view2);
        if (IoU < config_.IoU_threshold)
            continue;

        // triangulation with weak epipolar constraints test
        Line3d line = triangulate(l1, view1, l2, view2);
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

void Triangulator::RunTriangulate() {
    int num_edges = countEdges();
    if (num_edges == 0)
        return;
    std::cout<<"Start triangulating pairs..."<<std::endl;
    size_t n_images = CountImages();
    progressbar bar(n_images);
#pragma omp parallel for
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        bar.update();
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            triangulateOneNode(img_id, line_id);
        }
    }
    clearEdges();
}

void Triangulator::scoreOneNode(const int img_id, const int line_id, const LineLinker& linker) {
    if (already_scored_[img_id][line_id])
        return;
    auto& tris = tris_[img_id][line_id];
    size_t n_tris = tris.size();

    // score all the pairs
    std::vector<double> scores(n_tris, 0);
#pragma omp parallel for
    for (size_t i = 0; i < n_tris; ++i) {
        // each image contributes only once
        std::map<int, std::vector<double>> score_table;
        const Line3d& l1 = std::get<0>(tris[i]);
        int img_id = std::get<2>(tris[i]).first;
        int line_id = std::get<2>(tris[i]).second;
        const CameraView& view1 = imagecols_.camview(img_id);
        for (size_t j = 0; j < n_tris; ++j) {
            if (i == j)
                continue;
            const Line3d& l2 = std::get<0>(tris[j]);
            int ng_img_id = std::get<2>(tris[j]).first;
            int ng_line_id = std::get<2>(tris[j]).second;
            if (ng_img_id == img_id)
                continue;
            const CameraView& view2 = imagecols_.camview(ng_img_id);
            double score3d = linker.compute_score_3d(l1, l2);
            if (score3d == 0)
                continue;
            double score2d = linker.compute_score_2d(l1.projection(view2), all_lines_2d_[ng_img_id][ng_line_id]);
            if (score2d == 0)
                continue;
            double score = std::min(score3d, score2d);
            if (score_table.find(ng_img_id) == score_table.end())
                score_table.insert(std::make_pair(ng_img_id, std::vector<double>()));
            score_table[ng_img_id].push_back(score);
        }
        // one image contributes at most one support
        for (auto it = score_table.begin(); it != score_table.end(); ++it) {
            scores[i] += *std::max_element(it->second.begin(), it->second.end());
        }
    }
    for (size_t i = 0; i < n_tris; ++i) {
        std::get<1>(tris[i]) = scores[i];
    }

    // get valid tris and connections
    std::map<int, int> reverse_mapper;
    int n_neighbors = neighbors_[img_id].size();
    for (int i = 0; i < n_neighbors; ++i) {
        reverse_mapper.insert(std::make_pair(neighbors_[img_id][i], i));
    }
    std::vector<std::pair<double, int>> scores_to_sort;
    for (size_t tri_id = 0; tri_id < tris.size(); ++tri_id) {
        scores_to_sort.push_back(std::make_pair(std::get<1>(tris[tri_id]), tri_id));
    }
    std::sort(scores_to_sort.begin(), scores_to_sort.end(), std::greater<std::pair<double, int>>());
    int n_valid_conns = std::min(int(scores_to_sort.size()), config_.max_valid_conns);
    for (size_t i = 0; i < n_valid_conns; ++i) {
        int tri_id = scores_to_sort[i].second;
        auto& tri = tris[tri_id];
        double score = std::get<1>(tri);
        if (score < config_.fullscore_th)
            continue;
        valid_tris_[img_id][line_id].push_back(tri);
        auto& node = std::get<2>(tri);
        valid_edges_[img_id][line_id].push_back(std::make_pair(reverse_mapper.at(node.first), node.second));
    }

    // get best tris
    double max_score = -1;
    for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
        auto trituple = tris[tri_id];
        double score = std::get<1>(trituple);
        if (score > max_score) {
            tris_best_[img_id][line_id] = tris[tri_id];
            max_score = score;
        }
    }
    
    // clear intermediate triangulations
    if (!config_.debug_mode) {
        tris_[img_id][line_id].clear();
        valid_tris_[img_id][line_id].clear();
    }
    already_scored_[img_id][line_id] = true;
}

void Triangulator::RunScoring() {
    std::cout<<"Start perspective scoring..."<<std::endl;
    LineLinker linker_scoring = linker_;
    linker_scoring.linker_3d.config.set_to_shared_parent_scoring();

    // scoring all tris
    size_t n_images = CountImages();
    progressbar bar(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        bar.update();
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            scoreOneNode(img_id, line_id, linker_scoring);
        }
    }
}

const TriTuple& Triangulator::getBestTri(const int img_id, const int line_id) const {
    return tris_best_[img_id][line_id];
}

void Triangulator::filterNodeByNumOuterEdges(const std::vector<std::vector<std::vector<NeighborLineNode>>>& valid_edges, 
                                             std::vector<std::vector<bool>>& flags) {
    flags.clear();
    size_t n_images = CountImages();
    flags.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        flags[img_id].resize(n_lines);
        std::fill(flags[img_id].begin(), flags[img_id].end(), true);
    }

    // build checktable with all edges pointing to the node
    std::vector<std::vector<std::vector<LineNode>>> parent_neighbors;
    parent_neighbors.resize(n_images);
    std::vector<std::vector<int>> counters;
    counters.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        parent_neighbors[img_id].resize(n_lines);
        counters[img_id].resize(n_lines);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            counters[img_id][line_id] = valid_edges[img_id][line_id].size();
        }
    }
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            auto& nodes = valid_edges[img_id][line_id];
            for (auto it = nodes.begin(); it != nodes.end(); ++it) {
                int ng_img_id = neighbors_[img_id][it->first];
                int ng_line_id = it->second;
                parent_neighbors[ng_img_id][ng_line_id].push_back(std::make_pair(img_id, line_id));
            }
            if (counters[img_id][line_id] < config_.min_num_outer_edges) {
                flags[img_id][line_id] = false;
            }
        }
    }

    // iteratively filter node
    std::queue<LineNode> q;
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        for (size_t line_id = 0; line_id < CountLines(img_id); line_id++) {
            if (flags[img_id][line_id])
                continue;
            q.push(std::make_pair(img_id, line_id));
        }
    }

    while (!q.empty()) {
        LineNode node = q.front();
        q.pop();
        auto& parents = parent_neighbors[node.first][node.second];
        for (auto it = parents.begin(); it != parents.end(); ++it) {
            if (!flags[it->first][it->second])
                continue;
            counters[it->first][it->second]--;
            if (counters[it->first][it->second] < config_.min_num_outer_edges) {
                flags[it->first][it->second] = false;
                q.push(std::make_pair(it->first, it->second));
            }
        }
    }
}

void Triangulator::RunClustering() {
    std::cout<<"Start building the line graph for clustering..."<<std::endl;
    LineLinker linker_clustering = linker_;
    linker_clustering.linker_3d.config.set_to_spatial_merging();

    // get valid flags
    filterNodeByNumOuterEdges(valid_edges_, valid_flags_);

    // collect undirected edges
    std::set<std::pair<LineNode, LineNode>> edges;
    for (size_t img_id = 0; img_id < CountImages(); ++img_id) {
        for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
            auto& nodes = valid_edges_[img_id][line_id];
            for (auto it = nodes.begin(); it != nodes.end(); ++it) {
                LineNode node1 = std::make_pair(img_id, line_id);
                if (!valid_flags_[node1.first][node1.second])
                    continue;
                LineNode node2 = std::make_pair(neighbors_[img_id][it->first], it->second);
                if (!valid_flags_[node2.first][node2.second])
                    continue;
                if (node1.first > node2.first || (node1.first == node2.first && node1.second > node2.second))
                    std::swap(node1, node2);
                edges.insert(std::make_pair(node1, node2));
            }
        }
    }

    // insert edges one by one to build the graph
    for (auto it = edges.begin(); it != edges.end(); ++it) {
        int img_id1 = it->first.first; int line_id1 = it->first.second;
        int img_id2 = it->second.first; int line_id2 = it->second.second;
        const CameraView& view1 = imagecols_.camview(img_id1);
        const CameraView& view2 = imagecols_.camview(img_id2);
        const Line3d& line1 = std::get<0>(getBestTri(img_id1, line_id1));
        const Line3d& line2 = std::get<0>(getBestTri(img_id2, line_id2));
        Line2d& line2d1 = all_lines_2d_[img_id1][line_id1];
        Line2d& line2d2 = all_lines_2d_[img_id2][line_id2];

        double score_3d = linker_clustering.compute_score_3d(line1, line2);
        double score_2d_1to2 = linker_clustering.compute_score_2d(line1.projection(view2), line2d2);
        double score_2d_2to1 = linker_clustering.compute_score_2d(line2.projection(view1), line2d1);
        double score_2d = std::min(score_2d_1to2, score_2d_2to1);
        double score = std::min(score_3d, score_2d);
        score = score_3d;
        if (score == 0)
            continue;

        PatchNode* node1 = finalgraph_.FindOrCreateNode(img_id1, line_id1);
        PatchNode* node2 = finalgraph_.FindOrCreateNode(img_id2, line_id2);
        finalgraph_.AddEdge(node1, node2, score);
    }
}

void Triangulator::ComputeLineTracks() {
    std::cout<<"Start computing line tracks..."<<std::endl;
    LineLinker3d linker3d = linker_.linker_3d;
    linker3d.config.set_to_avgtest_merging();

    // collect lines for each node
    std::vector<Line3d> lines_nodes;
    for (auto it = finalgraph_.nodes.begin(); it != finalgraph_.nodes.end(); ++it) {
        int img_id = (*it)->image_idx;
        int line_id = (*it)->line_idx;
        lines_nodes.push_back(std::get<0>(getBestTri(img_id, line_id)));
    }
    std::vector<int> track_labels;
    if (config_.merging_strategy == "greedy")
        track_labels = merging::ComputeLineTrackLabelsGreedy(finalgraph_, lines_nodes);
    else if (config_.merging_strategy == "exhaustive")
        track_labels = merging::ComputeLineTrackLabelsExhaustive(finalgraph_, lines_nodes, linker3d);
    else if (config_.merging_strategy == "avg")
        track_labels = merging::ComputeLineTrackLabelsAvg(finalgraph_, lines_nodes, linker3d);
    else
        throw std::runtime_error("Error!The given merging strategy is not implemented");
    if (track_labels.empty())
        return;
    int n_tracks = *std::max_element(track_labels.begin(), track_labels.end()) + 1;
    tracks_.clear(); tracks_.resize(n_tracks);

    // set all lines into tracks
    size_t n_nodes = finalgraph_.nodes.size();
    for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
        PatchNode* node = finalgraph_.nodes[node_id];
        int img_id = node->image_idx;
        int line_id = node->line_idx;

        Line2d line2d = all_lines_2d_[img_id][line_id];
        Line3d line3d = std::get<0>(getBestTri(img_id, line_id));
        double score = std::get<1>(getBestTri(img_id, line_id));

        size_t track_id = track_labels[node_id];
        if (track_id == -1)
            continue;
        tracks_[track_id].node_id_list.push_back(node_id);
        tracks_[track_id].image_id_list.push_back(img_id);
        tracks_[track_id].line_id_list.push_back(line_id);
        tracks_[track_id].line2d_list.push_back(line2d);
        tracks_[track_id].line3d_list.push_back(line3d);
        tracks_[track_id].score_list.push_back(score);
    }

    // aggregate 3d lines to get final line proposal
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
        it->line = merging::Aggregator::aggregate_line3d_list(it->line3d_list, it->score_list, config_.num_outliers_aggregator);
    }
    finalgraph_.Clear();
}

void Triangulator::Run() {
    RunTriangulate();
    RunScoring();
    RunClustering();
    ComputeLineTracks();
}

// visualization
int Triangulator::CountAllTris() const {
    int counter = 0;
    size_t n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            int n_tris = tris_[img_id][line_id].size();
            counter += n_tris;
        }
    }
    return counter;
}

std::vector<TriTuple> Triangulator::GetScoredTrisNode(const int& image_id, const int& line_id) const {
    return tris_[image_id][line_id];
}

std::vector<TriTuple> Triangulator::GetValidScoredTrisNode(const int& image_id, const int& line_id) const {
    return valid_tris_[image_id][line_id];
}

std::vector<TriTuple> Triangulator::GetValidScoredTrisNodeSet(const int& img_id, const int& line_id) const {
    std::vector<TriTuple> res;
    auto& tris = valid_tris_[img_id][line_id];
    std::map<int, std::pair<int, double>> table; // (ng_img_id, (tri_id, score))
    int n_tris = tris.size();
    for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
        auto& tri = tris[tri_id];
        double score = std::get<1>(tri);
        int ng_img_id = std::get<2>(tri).first;
        if (table.find(ng_img_id) == table.end()) {
            table.insert(std::make_pair(ng_img_id, std::make_pair(tri_id, score)));
        }
        else {
            if (score > table.at(ng_img_id).second) {
                table.at(ng_img_id) = std::make_pair(tri_id, score);
            }
        }
    }
    for (auto it = table.begin(); it != table.end(); ++it) {
        int tri_id = it->second.first;
        res.push_back(valid_tris_[img_id][line_id][tri_id]);
    }
    return res;
}

int Triangulator::CountAllValidTris() const {
    int counter = 0;
    size_t n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            int n_tris = valid_tris_[img_id][line_id].size();
            counter += n_tris;
        }
    }
    return counter;
}

std::vector<Line3d> Triangulator::GetAllValidTris() const {
    std::vector<Line3d> res;
    size_t n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            auto& tris = valid_tris_[img_id][line_id];
            for (auto it = tris.begin(); it != tris.end(); ++it) {
                const auto& line = std::get<0>(*it);
                res.push_back(line);
            }
        }
    }
    return res;
}

std::vector<Line3d> Triangulator::GetValidTrisImage(const int& img_id) const {
    std::vector<Line3d> res;
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
        auto& tris = valid_tris_[img_id][line_id];
        for (auto it = tris.begin(); it != tris.end(); ++it) {
            const auto& line = std::get<0>(*it);
            res.push_back(line);
        }
    }
    return res;
}

std::vector<Line3d> Triangulator::GetValidTrisNode(const int& img_id, const int& line_id) const {
    std::vector<Line3d> res;
    auto& tris = valid_tris_[img_id][line_id];
    for (auto it = tris.begin(); it != tris.end(); ++it) {
        const auto& line = std::get<0>(*it);
        res.push_back(line);
    }
    return res;
}

std::vector<Line3d> Triangulator::GetValidTrisNodeSet(const int& img_id, const int& line_id) const {
    std::vector<Line3d> res;
    auto& tris = valid_tris_[img_id][line_id];
    std::map<int, std::pair<int, double>> table; // (ng_img_id, (tri_id, score))
    int n_tris = tris.size();
    for (int tri_id = 0; tri_id < n_tris; ++tri_id) {
        auto& tri = tris[tri_id];
        double score = std::get<1>(tri);
        int ng_img_id = std::get<2>(tri).first;
        if (table.find(ng_img_id) == table.end()) {
            table.insert(std::make_pair(ng_img_id, std::make_pair(tri_id, score)));
        }
        else {
            if (score > table.at(ng_img_id).second) {
                table.at(ng_img_id) = std::make_pair(tri_id, score);
            }
        }
    }
    for (auto it = table.begin(); it != table.end(); ++it) {
        int tri_id = it->second.first;
        auto& tri = tris[tri_id];
        const auto& line = std::get<0>(tri);
        res.push_back(line);   
    }
    return res;
}

std::vector<Line3d> Triangulator::GetAllBestTris() const {
    std::vector<Line3d> res;
    size_t n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            res.push_back(std::get<0>(getBestTri(img_id, line_id)));
        }
    }
    return res;
}

std::vector<Line3d> Triangulator::GetAllValidBestTris() const {
    std::vector<Line3d> res;
    size_t n_images = CountImages();
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        size_t n_lines = CountLines(img_id);
        for (size_t line_id = 0; line_id < n_lines; ++line_id) {
            if (!valid_flags_[img_id][line_id])
                continue;
            res.push_back(std::get<0>(getBestTri(img_id, line_id)));
        }
    }
    return res;
}

std::vector<Line3d> Triangulator::GetBestTrisImage(const int& img_id) const {
    std::vector<Line3d> res;
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
        res.push_back(std::get<0>(getBestTri(img_id, line_id)));
    }
    return res;
}

Line3d Triangulator::GetBestTriNode(const int& img_id, const int& line_id) const {
    return std::get<0>(getBestTri(img_id, line_id));
}

TriTuple Triangulator::GetBestScoredTriNode(const int& img_id, const int& line_id) const {
    return getBestTri(img_id, line_id);
}

std::vector<int> Triangulator::GetSurvivedLinesImage(const int& image_id, const int& n_visible_views) const {
    std::vector<int> survivers;
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
        if (it->count_images() < n_visible_views)
            continue;
        size_t n_lines = it->count_lines();
        for (size_t i = 0; i < n_lines; ++i) {
            int img_id = it->image_id_list[i];
            if (img_id != image_id)
                continue;
            survivers.push_back(it->line_id_list[i]);
        }
    }
    return survivers;
}

} // namespace triangulation

} // namespace limap

