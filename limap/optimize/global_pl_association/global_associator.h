#ifndef LIMAP_OPTIMIZE_GLOBAL_PL_ASSOCIATION_GLOBAL_ASSOCIATOR_H_
#define LIMAP_OPTIMIZE_GLOBAL_PL_ASSOCIATION_GLOBAL_ASSOCIATOR_H_

#include "base/image_collection.h"
#include "base/infinite_line.h"
#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment.h"
#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment_config.h"
#include "structures/pl_bipartite.h"
#include "structures/vpline_bipartite.h"
#include "util/types.h"

#include <ceres/ceres.h>

namespace limap {

namespace optimize {

namespace global_pl_association {

class GlobalAssociatorConfig : public hybrid_bundle_adjustment::HybridBAConfig {
public:
  GlobalAssociatorConfig() : hybrid_bundle_adjustment::HybridBAConfig() {
    InitConfig();
  }
  GlobalAssociatorConfig(py::dict dict)
      : hybrid_bundle_adjustment::HybridBAConfig(dict) {
    InitConfig();
    ASSIGN_PYDICT_ITEM(dict, constant_vp, bool)
    ASSIGN_PYDICT_ITEM(dict, th_angle_lineline, double)
    ASSIGN_PYDICT_ITEM(dict, th_count_lineline, int)
    ASSIGN_PYDICT_ITEM(dict, lw_pointline_association, double)
    ASSIGN_PYDICT_ITEM(dict, th_pixel, double)
    ASSIGN_PYDICT_ITEM(dict, th_weight_pointline, double)
    ASSIGN_PYDICT_ITEM(dict, lw_vpline_association, double)
    ASSIGN_PYDICT_ITEM(dict, th_count_vpline, int)
    ASSIGN_PYDICT_ITEM(dict, lw_vp_orthogonality, double)
    ASSIGN_PYDICT_ITEM(dict, th_angle_orthogonality, double)
    ASSIGN_PYDICT_ITEM(dict, lw_vp_collinearity, double)
    ASSIGN_PYDICT_ITEM(dict, th_angle_collinearity, double)
    ASSIGN_PYDICT_ITEM(dict, th_hard_pl_dist3d, double)
    ASSIGN_PYDICT_ITEM(dict, th_hard_vpline_angle3d, double)
  }

  // association config
  int th_count_lineline = 3;       // for junction reassociation
  double th_angle_lineline = 30.0; // for junction reassociation
  std::shared_ptr<ceres::LossFunction> point_line_association_3d_loss_function;
  double lw_pointline_association = 10.0;
  double th_pixel = 2.0; // in pixels
  double th_weight_pointline = 3.0;
  std::shared_ptr<ceres::LossFunction> vp_line_association_3d_loss_function;
  double lw_vpline_association = 1.0;
  int th_count_vpline = 3;
  std::shared_ptr<ceres::LossFunction> vp_orthogonality_loss_function;
  double lw_vp_orthogonality = 1.0;
  double th_angle_orthogonality = 87.0;
  std::shared_ptr<ceres::LossFunction> vp_collinearity_loss_function;
  double lw_vp_collinearity = 0.0;
  double th_angle_collinearity = 1.0;

  // hard association for output
  double th_hard_pl_dist3d = 2.0;
  double th_hard_vpline_angle3d = 5.0;

  // optimize
  bool constant_vp = false;

private:
  void InitConfig() {
    point_line_association_3d_loss_function.reset(new ceres::HuberLoss(0.01));
    vp_line_association_3d_loss_function.reset(new ceres::HuberLoss(0.01));
    vp_orthogonality_loss_function.reset(new ceres::TrivialLoss());
    vp_collinearity_loss_function.reset(new ceres::TrivialLoss());
  }
};

class GlobalAssociator : public hybrid_bundle_adjustment::HybridBAEngine {
public:
  GlobalAssociator() {}
  ~GlobalAssociator() {}
  GlobalAssociator(const GlobalAssociatorConfig &config)
      : hybrid_bundle_adjustment::HybridBAEngine(config), config_(config) {}

  // init
  void InitVPTracks(const std::map<int, vplib::VPTrack> &vptracks);
  void InitVPTracks(const std::vector<vplib::VPTrack> &vptracks);
  void Init2DBipartites_PointLine(
      const std::map<int, structures::PL_Bipartite2d> &all_bpt2ds) {
    enable_pointline_ = true;
    all_bpt2ds_ = all_bpt2ds;
  }
  void Init2DBipartites_VPLine(
      const std::map<int, structures::VPLine_Bipartite2d> &all_bpt2ds_vp) {
    enable_vpline_ = true;
    all_bpt2ds_vp_ = all_bpt2ds_vp;
  }
  void ReassociateJunctions(); // add newly associated junctions to bpt3d_

  // setup
  void SetUp();
  bool Solve();

  // output
  std::map<int, V3D> GetOutputVPs() const;
  std::map<int, vplib::VPTrack> GetOutputVPTracks() const;
  structures::PL_Bipartite3d GetBipartite3d_PointLine_Constraints() const;
  structures::PL_Bipartite3d GetBipartite3d_PointLine() const;
  structures::VPLine_Bipartite3d GetBipartite3d_VPLine() const;

protected:
  const GlobalAssociatorConfig config_;

  // point-line bipartites on 3d and 2d
  bool enable_pointline_ = false;
  std::map<int, structures::PL_Bipartite2d> all_bpt2ds_;

  // vp-line bipartites on 3d and 2d
  bool enable_vpline_ = false;
  std::map<int, structures::VPLine_Bipartite2d> all_bpt2ds_vp_;

  // minimal data and track information
  std::map<int, V3D> vps_;
  std::map<int, vplib::VPTrack> vp_tracks_;

  // parameterization
  void ParameterizeVPs();

  // structural residuals
  void AddPointLineAssociationResiduals(
      const std::map<std::pair<int, int>, double> &weights);
  void AddVPLineAssociationResiduals(
      const std::map<std::pair<int, int>, int> &weights);
  void AddVPOrthogonalityResiduals();
  void AddVPCollinearityResiduals();

private:
  // sparse weight matrix
  double dist2weight(const double &dist) const;
  double compute_overall_weight(
      const std::vector<std::pair<int, double>> &weights) const;
  std::map<std::pair<int, int>, double>
  construct_weights_pointline(const double th_weight)
      const; // soft association. Returns: ((point_id, line_id), weight)
  std::map<std::pair<int, int>, int>
  construct_weights_vpline(const int th_count)
      const; // soft association. Returns: ((vp_id, line_id), weight)
  std::vector<std::pair<int, int>>
  construct_pairs_vp_orthogonality(const std::map<int, V3D> &vps,
                                   const double th_angle_orthogonality) const;
  std::vector<std::pair<int, int>>
  construct_pairs_vp_collinearity(const std::map<int, V3D> &vps,
                                  const double th_angle_collinearity) const;

  // test line track validity
  bool test_linetrack_validity(const LineTrack &track) const;
};

} // namespace global_pl_association

} // namespace optimize

} // namespace limap

#endif
