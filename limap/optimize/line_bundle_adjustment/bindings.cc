#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"

#include "optimize/line_bundle_adjustment/lineba.h"
#include "optimize/line_bundle_adjustment/lineba_config.h"

namespace py = pybind11;

namespace limap {

template <typename DTYPE, int CHANNELS>
void bind_lineba_engine(py::module& m, std::string type_suffix) {
    using namespace optimize::line_bundle_adjustment;

    using BAEngine = LineBAEngine<DTYPE, CHANNELS>;

    py::class_<BAEngine>(m, ("LineBAEngine" + type_suffix).c_str())
        .def(py::init<>())
        .def(py::init<const LineBAConfig&>())
        .def("Initialize", &BAEngine::Initialize)
        .def("InitializeReconstruction", &BAEngine::InitializeReconstruction)
        .def("InitializeVPs", &BAEngine::InitializeVPs)
        .def("InitializeHeatmaps", &BAEngine::InitializeHeatmaps)
        .def("InitializePatches", &BAEngine::InitializePatches)
        .def("SetUp", &BAEngine::SetUp)
        .def("Solve", &BAEngine::Solve)
        .def("GetOutputCameras", &BAEngine::GetOutputCameras)
        .def("GetOutputLines", &BAEngine::GetOutputLines)
        .def("GetOutputTracks", &BAEngine::GetOutputTracks)
        .def("GetOutputReconstruction", &BAEngine::GetOutputReconstruction)
        .def("GetHeatmapIntersections", &BAEngine::GetHeatmapIntersections);
}

void bind_line_bundle_adjustment(py::module &m) {
    using namespace optimize::line_bundle_adjustment;

    py::class_<LineBAConfig>(m, "LineBAConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("use_geometric", &LineBAConfig::use_geometric)
        .def_readwrite("min_num_images", &LineBAConfig::min_num_images)
        .def_readwrite("sample_range_min", &LineBAConfig::sample_range_min)
        .def_readwrite("sample_range_max", &LineBAConfig::sample_range_max)
        .def_readwrite("vp_multiplier", &LineBAConfig::vp_multiplier)
        .def_readwrite("n_samples_heatmap", &LineBAConfig::n_samples_heatmap)
        .def_readwrite("heatmap_multiplier", &LineBAConfig::heatmap_multiplier)
        .def_readwrite("n_samples_feature", &LineBAConfig::n_samples_feature)
        .def_readwrite("use_ref_descriptor", &LineBAConfig::use_ref_descriptor)
        .def_readwrite("ref_multiplier", &LineBAConfig::ref_multiplier)
        .def_readwrite("fconsis_multiplier", &LineBAConfig::fconsis_multiplier)
        .def_readwrite("solver_options", &LineBAConfig::solver_options)
        .def_readwrite("heatmap_interpolation_config", &LineBAConfig::heatmap_interpolation_config)
        .def_readwrite("feature_interpolation_config", &LineBAConfig::feature_interpolation_config)
        .def_readwrite("geometric_loss_function", &LineBAConfig::geometric_loss_function)
        .def_readwrite("geometric_alpha", &LineBAConfig::geometric_alpha)
        .def_readwrite("vp_loss_function", &LineBAConfig::vp_loss_function)
        .def_readwrite("heatmap_loss_function", &LineBAConfig::heatmap_loss_function)
        .def_readwrite("fconsis_loss_function", &LineBAConfig::fconsis_loss_function)

        .def_readwrite("print_summary", &LineBAConfig::print_summary)
        .def_readwrite("constant_intrinsics", &LineBAConfig::constant_intrinsics)
        .def_readwrite("constant_pose", &LineBAConfig::constant_pose)
        .def_readwrite("constant_line", &LineBAConfig::constant_line);

    bind_lineba_engine<float16, 128>(m, "_f16_c128");

// #define REGISTER_CHANNEL(CHANNELS) \
//     bind_lineba_engine<float16, CHANNELS>(m, "_f16_c" + std::to_string(CHANNELS)); \
// 
//     REGISTER_CHANNEL(1);
//     REGISTER_CHANNEL(3);
//     REGISTER_CHANNEL(128);
// 
// #undef REGISTER_CHANNEL

}

} // namespace limap


