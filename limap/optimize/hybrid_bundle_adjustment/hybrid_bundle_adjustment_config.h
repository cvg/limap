#ifndef LIMAP_OPTIMIZE_HYBRIDBA_HYBRIDBA_CONFIG_H_
#define LIMAP_OPTIMIZE_HYBRIDBA_HYBRIDBA_CONFIG_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "optimize/line_refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace hybrid_bundle_adjustment {

class HybridBAConfig : public line_refinement::RefinementConfig {
public:
  HybridBAConfig() : line_refinement::RefinementConfig() { InitConfig(); }
  HybridBAConfig(py::dict dict) : line_refinement::RefinementConfig(dict) {
    InitConfig();
    ASSIGN_PYDICT_ITEM(dict, constant_intrinsics, bool);
    ASSIGN_PYDICT_ITEM(dict, constant_principal_point, bool);
    ASSIGN_PYDICT_ITEM(dict, constant_pose, bool);
    ASSIGN_PYDICT_ITEM(dict, constant_point, bool);
    ASSIGN_PYDICT_ITEM(dict, constant_line, bool);
    ASSIGN_PYDICT_ITEM(dict, lw_point, double);
  }
  bool constant_intrinsics = false;
  bool constant_principal_point = true;
  bool constant_pose = false;
  bool constant_point = false;
  bool constant_line = false;

  // point geometric config
  std::shared_ptr<ceres::LossFunction> point_geometric_loss_function;
  double lw_point = 0.1;

  // functions
  void set_constant_camera() {
    constant_intrinsics = true;
    constant_pose = true;
  }

private:
  void InitConfig() {
    point_geometric_loss_function.reset(new ceres::TrivialLoss());
  }
};

} // namespace hybrid_bundle_adjustment

} // namespace optimize

} // namespace limap

#endif
