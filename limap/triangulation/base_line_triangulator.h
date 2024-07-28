#ifndef LIMAP_TRIANGULATION_BASE_LINE_TRIANGULATOR_H_
#define LIMAP_TRIANGULATION_BASE_LINE_TRIANGULATOR_H_

#include "base/image_collection.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "structures/pl_bipartite.h"
#include "util/types.h"
#include "vplib/vpbase.h"
#include <tuple>

namespace limap {

namespace triangulation {

typedef std::pair<uint8_t, uint16_t> NeighborLineNode; // (neighbor_id, line_id)
typedef Node2d LineNode;                               // (img_id, line_id)
typedef std::tuple<Line3d, double, LineNode>
    TriTuple; // (line, score, (ng_img_id, ng_line_id))

class BaseLineTriangulatorConfig {
public:
  BaseLineTriangulatorConfig() {}
  BaseLineTriangulatorConfig(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, add_halfpix, bool)
    ASSIGN_PYDICT_ITEM(dict, use_vp, bool)
    ASSIGN_PYDICT_ITEM(dict, use_endpoints_triangulation, bool)
    ASSIGN_PYDICT_ITEM(dict, disable_many_points_triangulation, bool)
    ASSIGN_PYDICT_ITEM(dict, disable_one_point_triangulation, bool)
    ASSIGN_PYDICT_ITEM(dict, disable_algebraic_triangulation, bool)
    ASSIGN_PYDICT_ITEM(dict, disable_vp_triangulation, bool)
    ASSIGN_PYDICT_ITEM(dict, min_length_2d, double)
    ASSIGN_PYDICT_ITEM(dict, line_tri_angle_threshold, double)
    ASSIGN_PYDICT_ITEM(dict, IoU_threshold, double)
    ASSIGN_PYDICT_ITEM(dict, debug_mode, bool)
    ASSIGN_PYDICT_ITEM(dict, sensitivity_threshold, double)
    ASSIGN_PYDICT_ITEM(dict, var2d, double);
  }

  // general options
  bool debug_mode = false;
  bool add_halfpix = false; // offset half pixel for each line
  bool use_vp = false;
  bool use_endpoints_triangulation = false;

  // proposal types
  bool disable_many_points_triangulation = false;
  bool disable_one_point_triangulation = false;
  bool disable_algebraic_triangulation = false;
  bool disable_vp_triangulation = false;

  // hyperparameters
  double min_length_2d = 20.0;
  double line_tri_angle_threshold = 5.0;
  double IoU_threshold = 0.1;
  double sensitivity_threshold = 70.0;
  double var2d = 2.0;
};

class BaseLineTriangulator {
public:
  BaseLineTriangulator() {}
  BaseLineTriangulator(const BaseLineTriangulatorConfig &config)
      : config_(config) {}
  const BaseLineTriangulatorConfig config_;

  // interfaces
  void Init(const std::map<int, std::vector<Line2d>> &all_2d_segs,
            const ImageCollection *imagecols);
  void Init(const std::map<int, std::vector<Line2d>> &all_2d_segs,
            const ImageCollection &imagecols);
  void InitVPResults(const std::map<int, vplib::VPResult> &vpresults) {
    vpresults_ = vpresults;
  }
  void TriangulateImage(const int img_id,
                        const std::map<int, Eigen::MatrixXi> &matches);
  void TriangulateImageExhaustiveMatch(const int img_id,
                                       const std::vector<int> &neighbors);
  virtual std::vector<LineTrack> ComputeLineTracks() = 0;
  void SetRanges(const std::pair<V3D, V3D> &ranges) {
    ranges_flag_ = true;
    ranges_ = ranges;
  }
  void UnsetRanges() { ranges_flag_ = false; }

  // optional (for pointsfm)
  void
  SetBipartites2d(const std::map<int, structures::PL_Bipartite2d> &all_bpt2ds) {
    use_pointsfm_ = true;
    all_bpt2ds_ =
        std::make_shared<std::map<int, structures::PL_Bipartite2d>>(all_bpt2ds);
  }
  void SetSfMPoints(const std::map<int, V3D> &points) { sfm_points_ = points; }

  // data
  std::vector<LineTrack> GetTracks() const { return tracks_; };
  vplib::VPResult GetVPResult(const int &image_id) const {
    return vpresults_.at(image_id);
  }
  std::map<int, vplib::VPResult> GetVPResults() const { return vpresults_; }

  // utilities
  size_t CountImages() const { return all_lines_2d_.size(); }
  size_t CountLines(const int &img_id) const {
    return all_lines_2d_.at(img_id).size();
  }

protected:
  virtual void
  ScoringCallback(const int img_id) = 0; // score (and track) triangulations

  // ranges
  bool ranges_flag_ = false;
  std::pair<V3D, V3D> ranges_;

  // initialization
  void offsetHalfPixel();
  std::map<int, std::vector<Line2d>> all_lines_2d_;
  const ImageCollection *imagecols_;
  std::map<int, std::vector<int>>
      neighbors_; // visual neighbors for each image, initialized with InitMatch
                  // interfaces
  std::map<int, vplib::VPResult> vpresults_; // vp results

  // connections
  std::map<int, std::vector<std::vector<LineNode>>>
      edges_; // list of (img_id, line_id) for each node, cleared after
              // triangulation

  // [optional]
  bool use_pointsfm_ = false;
  std::shared_ptr<std::map<int, structures::PL_Bipartite2d>> all_bpt2ds_;
  std::map<int, V3D> sfm_points_;

  // triangulation
  void clearEdgesOneNode(const int img_id, const int line_id);
  void clearEdges();
  int countEdges() const;
  void triangulateOneNode(const int img_id, const int line_id);
  std::map<int, std::vector<std::vector<TriTuple>>>
      tris_; // list of TriTuple for each node, need to be cleared after scoring

  // tracks
  std::vector<LineTrack> tracks_;
};

} // namespace triangulation

} // namespace limap

#endif
