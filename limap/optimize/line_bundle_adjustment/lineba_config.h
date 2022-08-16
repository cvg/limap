#ifndef LIMAP_OPTIMIZE_LINEBA_LINEBA_CONFIG_H_
#define LIMAP_OPTIMIZE_LINEBA_LINEBA_CONFIG_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "optimize/line_refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace line_bundle_adjustment {

class LineBAConfig: public line_refinement::RefinementConfig {
public:
    LineBAConfig(): line_refinement::RefinementConfig() {}
    LineBAConfig(py::dict dict): line_refinement::RefinementConfig(dict) {
        ASSIGN_PYDICT_ITEM(dict, constant_intrinsics, bool);
        ASSIGN_PYDICT_ITEM(dict, constant_pose, bool);
        ASSIGN_PYDICT_ITEM(dict, constant_line, bool);
    }
    bool constant_intrinsics = true;
    bool constant_pose = false;
    bool constant_line = false;
};

} // namespace line_bundle_adjustment

} // namespace optimize 

} // namespace limap

#endif

