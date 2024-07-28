#ifndef LIMAP_TRIANGULATION_GLOBAL_LINE_TRIANGULATOR_H_
#define LIMAP_TRIANGULATION_GLOBAL_LINE_TRIANGULATOR_H_

#include "base/graph.h"
#include "base/line_linker.h"
#include "triangulation/base_line_triangulator.h"

namespace limap {

namespace triangulation {

class GlobalLineTriangulatorConfig : public BaseLineTriangulatorConfig {
public:
  GlobalLineTriangulatorConfig() : BaseLineTriangulatorConfig() {}
  GlobalLineTriangulatorConfig(py::dict dict)
      : BaseLineTriangulatorConfig(dict) {
    ASSIGN_PYDICT_ITEM(dict, fullscore_th, double)
    ASSIGN_PYDICT_ITEM(dict, max_valid_conns, int)
    ASSIGN_PYDICT_ITEM(dict, min_num_outer_edges, int)
    ASSIGN_PYDICT_ITEM(dict, merging_strategy, std::string)
    ASSIGN_PYDICT_ITEM(dict, num_outliers_aggregator, int)
    if (dict.contains("linker2d_config"))
      linker2d_config = LineLinker2dConfig(dict["linker2d_config"]);
    if (dict.contains("linker3d_config"))
      linker3d_config = LineLinker3dConfig(dict["linker3d_config"]);
  }

  double fullscore_th = 1.0;
  int max_valid_conns = 1000;  // maximum valid connections for each node
  int min_num_outer_edges = 1; // filter node by num outer edge
  std::string merging_strategy = "greedy";
  int num_outliers_aggregator = 2;

  LineLinker2dConfig linker2d_config;
  LineLinker3dConfig linker3d_config;
};

class GlobalLineTriangulator : public BaseLineTriangulator {
public:
  GlobalLineTriangulator() : BaseLineTriangulator() {}
  GlobalLineTriangulator(const GlobalLineTriangulatorConfig &config)
      : BaseLineTriangulator(config), config_(config),
        linker_(config.linker2d_config, config.linker3d_config) {}
  GlobalLineTriangulator(py::dict dict)
      : GlobalLineTriangulator(GlobalLineTriangulatorConfig(dict)) {}
  GlobalLineTriangulatorConfig config_;

  // interfaces
  void Init(const std::map<int, std::vector<Line2d>> &all_2d_segs,
            const ImageCollection &imagecols); // overwrite
  std::vector<LineTrack> ComputeLineTracks();
  LineLinker GetLinker() const { return linker_; }

  // interface for visualization
  int CountAllTris() const;
  std::vector<TriTuple> GetScoredTrisNode(const int &image_id,
                                          const int &line_id) const;
  std::vector<TriTuple> GetValidScoredTrisNode(const int &image_id,
                                               const int &line_id) const;
  std::vector<TriTuple> GetValidScoredTrisNodeSet(const int &image_id,
                                                  const int &line_id) const;
  int CountAllValidTris() const;
  std::vector<Line3d> GetAllValidTris() const;
  std::vector<Line3d> GetValidTrisImage(const int &image_id) const;
  std::vector<Line3d> GetValidTrisNode(const int &image_id,
                                       const int &line_id) const;
  std::vector<Line3d> GetValidTrisNodeSet(const int &image_id,
                                          const int &line_id) const;
  std::vector<Line3d> GetAllBestTris() const;
  std::vector<Line3d> GetAllValidBestTris() const;
  std::vector<Line3d> GetBestTrisImage(const int &image_id) const;
  Line3d GetBestTriNode(const int &image_id, const int &line_id) const;
  TriTuple GetBestScoredTriNode(const int &image_id, const int &line_id) const;
  std::vector<int> GetSurvivedLinesImage(const int &image_id,
                                         const int &n_visible_views) const;

protected:
  // implement virtual function
  void ScoringCallback(const int img_id);

  // linker
  LineLinker linker_;

  // scoring
  void scoreOneNode(const int img_id, const int line_id,
                    const LineLinker &linker);
  std::map<int, std::vector<bool>>
      already_scored_; // monotoring the scoring process
  std::map<int, std::vector<std::vector<TriTuple>>>
      valid_tris_; // need to be cleared. only saved at debug mode
  std::map<int, std::vector<TriTuple>>
      tris_best_; // the best TriTuple for each node
  const TriTuple &getBestTri(const int img_id, const int line_id) const;

  // clustering
  void run_clustering(Graph *graph);
  void build_tracks_from_clusters(Graph *graph);
  std::map<int, std::vector<std::vector<NeighborLineNode>>>
      valid_edges_; // valid connections with survived triangulations
  std::map<int, std::vector<bool>> valid_flags_;
  void filterNodeByNumOuterEdges(
      const std::map<int, std::vector<std::vector<NeighborLineNode>>>
          &valid_edges,
      std::map<int, std::vector<bool>> &flags);
};

} // namespace triangulation

} // namespace limap

#endif
