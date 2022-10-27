#ifndef LIMAP_TRIANGULATION_TRIANGULATOR_H_
#define LIMAP_TRIANGULATION_TRIANGULATOR_H_

#include "util/types.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/line_linker.h"
#include "base/image_collection.h"
#include "base/graph.h"
#include "vplib/jlinkage.h"

#include <tuple>

namespace limap {

namespace triangulation {

typedef std::pair<uint8_t, uint16_t> NeighborLineNode; // (neighbor_id, line_id)
typedef Node2d LineNode; // (img_id, line_id)
typedef std::tuple<Line3d, double, LineNode> TriTuple; // (line, score, (ng_img_id, ng_line_id))

class TriangulatorConfig {
public:
    TriangulatorConfig() {}
    TriangulatorConfig(py::dict dict) {
        ASSIGN_PYDICT_ITEM(dict, add_halfpix, bool)
        ASSIGN_PYDICT_ITEM(dict, use_vp, bool)
        ASSIGN_PYDICT_ITEM(dict, use_endpoints_triangulation, bool)
        ASSIGN_PYDICT_ITEM(dict, min_length_2d, double)
        ASSIGN_PYDICT_ITEM(dict, var2d, double);
        ASSIGN_PYDICT_ITEM(dict, plane_angle_threshold, double)
        ASSIGN_PYDICT_ITEM(dict, IoU_threshold, double)
        ASSIGN_PYDICT_ITEM(dict, sensitivity_threshold, double)
        ASSIGN_PYDICT_ITEM(dict, fullscore_th, double)
        ASSIGN_PYDICT_ITEM(dict, max_valid_conns, int)
        ASSIGN_PYDICT_ITEM(dict, min_num_outer_edges, int)
        ASSIGN_PYDICT_ITEM(dict, merging_strategy, std::string)
        ASSIGN_PYDICT_ITEM(dict, num_outliers_aggregator, int)
        ASSIGN_PYDICT_ITEM(dict, debug_mode, bool)
        if (dict.contains("vpdet_config"))
            vpdet_config = vplib::JLinkageConfig(dict["vpdet_config"]);
        if (dict.contains("linker2d_config"))
            linker2d_config = LineLinker2dConfig(dict["linker2d_config"]);
        if (dict.contains("linker3d_config"))
            linker3d_config = LineLinker3dConfig(dict["linker3d_config"]);
    }

    bool add_halfpix = false; // offset half pixel for each line
    bool use_vp = true;
    bool use_endpoints_triangulation = false;
    vplib::JLinkageConfig vpdet_config;

    double min_length_2d = 20.0;
    double var2d = 2.0;
    double plane_angle_threshold = 5.0;
    double IoU_threshold = 0.1;
    double sensitivity_threshold = 70.0;
    double fullscore_th = 1.0;
    int max_valid_conns = 1000; // maximum valid connections for each node
    int min_num_outer_edges = 1; // filter node by num outer edge
    std::string merging_strategy = "greedy";
    int num_outliers_aggregator = 2;
    bool debug_mode = false;

    LineLinker2dConfig linker2d_config;
    LineLinker3dConfig linker3d_config;
};

class Triangulator {
public:
    Triangulator() {}
    Triangulator(const TriangulatorConfig& config): config_(config), linker_(config.linker2d_config, config.linker3d_config), vpdetector_(config.vpdet_config) {}
    Triangulator(py::dict dict): Triangulator(TriangulatorConfig(dict)) {};
    TriangulatorConfig config_;
    LineLinker linker_;
    const vplib::JLinkage vpdetector_;

    void Init(const std::map<int, std::vector<Line2d>>& all_2d_segs,
              const ImageCollection& imagecols);
    void InitMatches(const std::map<int, std::vector<Eigen::MatrixXi>>& all_matches,
                     const std::map<int, std::vector<int>>& all_neighbors,
                     bool use_triangulate=true,
                     bool use_scoring=false);
    void InitMatchImage(const int img_id,
                        const std::vector<Eigen::MatrixXi>& matches,
                        const std::vector<int>& neighbors,
                        bool use_triangulate=true,
                        bool use_scoring=false);
    void InitExhaustiveMatchImage(const int img_id, 
                                  const std::vector<int>& neighbors,
                                  bool use_scoring=true);
    void InitAll(const std::map<int, std::vector<Line2d>>& all_2d_segs,
                 const ImageCollection& imagecols,
                 const std::map<int, std::vector<Eigen::MatrixXi>>& all_matches,
                 const std::map<int, std::vector<int>>& all_neighbors,
                 bool use_triangulate=false,
                 bool use_scoring=false);
    void RunTriangulate();
    void RunScoring();
    void RunClustering();
    void ComputeLineTracks();
    void Run();
    
    // interfaces 
    void SetRanges(const std::pair<V3D, V3D>& ranges) { ranges_flag_ = true; ranges_ = ranges; }
    void UnsetRanges() { ranges_flag_ = false; }
    LineLinker GetLinker() const { return linker_; }
    std::vector<LineTrack> GetTracks() const {return tracks_; };
    vplib::VPResult GetVPResult(const int& image_id) const {return vpresults_.at(image_id); }
    std::map<int, vplib::VPResult> GetVPResults() const {return vpresults_; }

    // infos
    size_t CountImages() const { return all_lines_2d_.size(); }
    size_t CountLines(const int& img_id) const { return all_lines_2d_.at(img_id).size(); }

    // interface for visualization
    int CountAllTris() const;
    std::vector<TriTuple> GetScoredTrisNode(const int& image_id, const int& line_id) const;
    std::vector<TriTuple> GetValidScoredTrisNode(const int& image_id, const int& line_id) const;
    std::vector<TriTuple> GetValidScoredTrisNodeSet(const int& image_id, const int& line_id) const;
    int CountAllValidTris() const;
    std::vector<Line3d> GetAllValidTris() const;
    std::vector<Line3d> GetValidTrisImage(const int& image_id) const;
    std::vector<Line3d> GetValidTrisNode(const int& image_id, const int& line_id) const;
    std::vector<Line3d> GetValidTrisNodeSet(const int& image_id, const int& line_id) const;
    std::vector<Line3d> GetAllBestTris() const;
    std::vector<Line3d> GetAllValidBestTris() const;
    std::vector<Line3d> GetBestTrisImage(const int& image_id) const;
    Line3d GetBestTriNode(const int& image_id, const int& line_id) const;
    TriTuple GetBestScoredTriNode(const int& image_id, const int& line_id) const;
    std::vector<int> GetSurvivedLinesImage(const int& image_id, const int& n_visible_views) const;

private:
    // ranges
    bool ranges_flag_ = false;
    std::pair<V3D, V3D> ranges_;

    // initialization
    void offsetHalfPixel();
    std::map<int, std::vector<Line2d>> all_lines_2d_;
    ImageCollection imagecols_;
    std::map<int, std::vector<int>> neighbors_; // visual neighbors for each image, initialized with InitMatch interfaces
    std::map<int, vplib::VPResult> vpresults_; // vp results

    // connections
    std::map<int, std::vector<std::vector<LineNode>>> edges_; // list of (img_id, line_id) for each node, cleared after triangulation
    std::map<int, std::vector<std::vector<NeighborLineNode>>> valid_edges_; // valid connections with survived triangulations
    std::map<int, std::vector<bool>> valid_flags_;

    // triangulation and scoring
    void clearEdgesOneNode(const int img_id, const int line_id);
    void clearEdges();
    int countEdges() const;
    const TriTuple& getBestTri(const int img_id, const int line_id) const;
    // functions per node
    void triangulateOneNode(const int img_id, const int line_id);
    void scoreOneNode(const int img_id, const int line_id, const LineLinker& linker);
    std::map<int, std::vector<bool>> already_scored_; // monotoring the scoring process

    // only saved at debug mode
    std::map<int, std::vector<std::vector<TriTuple>>> tris_; // list of TriTuple for each node
    std::map<int, std::vector<std::vector<TriTuple>>> valid_tris_;

    // saved data 
    std::map<int, std::vector<TriTuple>> tris_best_; // the best TriTuple for each node
    Graph finalgraph_;
    void filterNodeByNumOuterEdges(const std::map<int, std::vector<std::vector<NeighborLineNode>>>& valid_edges, 
                                   std::map<int, std::vector<bool>>& flags);

    // tracks
    std::vector<LineTrack> tracks_;
};

} // namespace triangulation

} // namespace limap

#endif

