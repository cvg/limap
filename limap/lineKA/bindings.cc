#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "lineKA/lineka.h"
#include "lineKA/lineka_config.h"

namespace py = pybind11;

namespace limap {

template <typename DTYPE, int CHANNELS>
void bind_lineka_engine(py::module& m, std::string type_suffix) {
    using namespace lineKA;

    using KAEngine = LineKAEngine<DTYPE, CHANNELS>;

    py::class_<KAEngine>(m, ("LineKAEngine" + type_suffix).c_str())
        .def(py::init<>())
        .def(py::init<const LineKAConfig&>())
        .def("Initialize", &KAEngine::Initialize)
        .def("InitializeMatches", &KAEngine::InitializeMatches)
        .def("InitializeHeatmaps", &KAEngine::InitializeHeatmaps)
        .def("InitializePatches", &KAEngine::InitializePatches)
        .def("SetUp", &KAEngine::SetUp)
        .def("Solve", &KAEngine::Solve)
        .def("GetOutputLines", &KAEngine::GetOutputLines);
}

void bind_lineKA(py::module &m) {
    using namespace lineKA;

    py::class_<LineKAConfig>(m, "LineKAConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("min_num_images", &LineKAConfig::min_num_images)
        .def_readwrite("sample_range_min", &LineKAConfig::sample_range_min)
        .def_readwrite("sample_range_max", &LineKAConfig::sample_range_max)
        .def_readwrite("n_samples_heatmap", &LineKAConfig::n_samples_heatmap)
        .def_readwrite("n_samples_feature", &LineKAConfig::n_samples_feature)
        .def_readwrite("solver_options", &LineKAConfig::solver_options)
        .def_readwrite("heatmap_interpolation_config", &LineKAConfig::heatmap_interpolation_config)
        .def_readwrite("feature_interpolation_config", &LineKAConfig::feature_interpolation_config)
        .def_readwrite("print_summary", &LineKAConfig::print_summary)
        .def_readwrite("line2d_type", &LineKAConfig::line2d_type);

#define REGISTER_CHANNEL(CHANNELS) \
    bind_lineka_engine<float16, CHANNELS>(m, "_f16_c" + std::to_string(CHANNELS)); \

    REGISTER_CHANNEL(1);
    REGISTER_CHANNEL(3);
    REGISTER_CHANNEL(128);

#undef REGISTER_CHANNEL
}

} // namespace limap


