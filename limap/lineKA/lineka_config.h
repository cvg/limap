#ifndef LIMAP_LINEKA_LINEKA_CONFIG_H_
#define LIMAP_LINEKA_LINEKA_CONFIG_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

namespace lineKA {

class LineKAConfig: public refinement::RefinementConfig {
public:
    LineKAConfig(): refinement::RefinementConfig() {}
    LineKAConfig(py::dict dict): refinement::RefinementConfig(dict) {
        ASSIGN_PYDICT_ITEM(dict, line2d_type, std::string);
        ASSIGN_PYDICT_ITEM(dict, n_samples_feature_2d, int);
    }
    std::string line2d_type = "fixedprojection";
    int n_samples_feature_2d = 10;
};

} // namespace lineKA 

} // namespace limap

#endif

