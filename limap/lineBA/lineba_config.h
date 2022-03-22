#ifndef LIMAP_LINEBA_LINEBA_CONFIG_H_
#define LIMAP_LINEBA_LINEBA_CONFIG_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

namespace lineBA {

class LineBAConfig: public refinement::RefinementConfig {
public:
    LineBAConfig(): refinement::RefinementConfig() {}
    LineBAConfig(py::dict dict): refinement::RefinementConfig(dict) {
        ASSIGN_PYDICT_ITEM(dict, constant_pose, bool);
    }
    bool constant_pose = false;
    bool constant_line = false;
};

} // namespace lineBA 

} // namespace limap

#endif

