#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "optimize/line_refinement/refine.h"
#include "optimize/line_refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

template <typename DTYPE, int CHANNELS>
void bind_refinement_engine(py::module &m, std::string type_suffix) {
  using namespace optimize::line_refinement;

  using RFEngine = RefinementEngine<DTYPE, CHANNELS>;

  py::class_<RFEngine>(m, ("RefinementEngine" + type_suffix).c_str())
      .def(py::init<>())
      .def(py::init<const RefinementConfig &>())
      .def("Initialize", &RFEngine::Initialize)
      .def("InitializeVPs", &RFEngine::InitializeVPs)
      .def("SetUp", &RFEngine::SetUp)
      .def("Solve", &RFEngine::Solve)
      .def("GetLine3d", &RFEngine::GetLine3d)
#ifdef INTERPOLATION_ENABLED
      .def("InitializeHeatmaps", &RFEngine::InitializeHeatmaps)
      .def("InitializeFeatures", &RFEngine::InitializeFeatures)
      .def("InitializeFeaturesAsPatches",
           &RFEngine::InitializeFeaturesAsPatches)
      .def("GetHeatmapIntersections", &RFEngine::GetHeatmapIntersections)
      .def("GetFConsistencyIntersections",
           &RFEngine::GetFConsistencyIntersections)
#endif // INTERPOLATION_ENABLED
      .def("GetAllStates", &RFEngine::GetAllStates);
}

void bind_line_refinement(py::module &m) {
  using namespace optimize::line_refinement;

  py::class_<RefinementConfig>(m, "RefinementConfig")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("use_geometric", &RefinementConfig::use_geometric)
      .def_readwrite("min_num_images", &RefinementConfig::min_num_images)
      .def_readwrite("sample_range_min", &RefinementConfig::sample_range_min)
      .def_readwrite("sample_range_max", &RefinementConfig::sample_range_max)
      .def_readwrite("vp_multiplier", &RefinementConfig::vp_multiplier)
      .def_readwrite("n_samples_heatmap", &RefinementConfig::n_samples_heatmap)
      .def_readwrite("heatmap_multiplier",
                     &RefinementConfig::heatmap_multiplier)
      .def_readwrite("n_samples_feature", &RefinementConfig::n_samples_feature)
      .def_readwrite("use_ref_descriptor",
                     &RefinementConfig::use_ref_descriptor)
      .def_readwrite("ref_multiplier", &RefinementConfig::ref_multiplier)
      .def_readwrite("fconsis_multiplier",
                     &RefinementConfig::fconsis_multiplier)
      .def_readwrite("solver_options", &RefinementConfig::solver_options)
      .def_readwrite("heatmap_interpolation_config",
                     &RefinementConfig::heatmap_interpolation_config)
      .def_readwrite("feature_interpolation_config",
                     &RefinementConfig::feature_interpolation_config)
      .def_readwrite("line_geometric_loss_function",
                     &RefinementConfig::line_geometric_loss_function)
      .def_readwrite("geometric_alpha", &RefinementConfig::geometric_alpha)
      .def_readwrite("vp_loss_function", &RefinementConfig::vp_loss_function)
      .def_readwrite("heatmap_loss_function",
                     &RefinementConfig::heatmap_loss_function)
      .def_readwrite("fconsis_loss_function",
                     &RefinementConfig::fconsis_loss_function)
      .def_readwrite("print_summary", &RefinementConfig::print_summary);

  bind_refinement_engine<float16, 128>(m, "_f16_c128");
}

} // namespace limap
