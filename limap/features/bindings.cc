#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <vector>

#include "features/dense_sift.h"
#include "features/featurepatch.h"
#include "features/line_patch_extractor.h"

namespace py = pybind11;

namespace limap {

template <typename DTYPE>
void bind_patchinfo(py::module &m, std::string type_suffix) {
  using namespace features;

  using PInfo = PatchInfo<DTYPE>;
  py::class_<PInfo>(m, ("PatchInfo" + type_suffix).c_str())
      .def(py::init<>())
      .def(py::init<py::array_t<DTYPE, py::array::c_style>, M2D, V2D,
                    std::pair<int, int>>())
      .def_readwrite("array", &PInfo::array)
      .def_readwrite("R", &PInfo::R)
      .def_readwrite("tvec", &PInfo::tvec)
      .def_readwrite("img_hw", &PInfo::img_hw);
}

template <typename DTYPE, int CHANNELS>
void bind_extractor_dtype(py::module &m, std::string type_suffix) {
  using namespace features;

  using LPExtractor = LinePatchExtractor<DTYPE, CHANNELS>;

  py::class_<LPExtractor>(m, ("LinePatchExtractor" + type_suffix).c_str())
      .def(py::init<>())
      .def(py::init<const LinePatchExtractorOptions &>())
      .def("GetLine2DRange", &LPExtractor::GetLine2DRange)
      .def("ExtractLinePatch", &LPExtractor::ExtractLinePatch)
      .def("ExtractLinePatches", &LPExtractor::ExtractLinePatches)
      .def("ExtractOneImage", &LPExtractor::ExtractOneImage)
      .def("Extract",
           [](LPExtractor &extractor, const LineTrack &track,
              const std::vector<CameraView> &p_views,
              const std::vector<py::array_t<DTYPE, py::array::c_style>>
                  &p_features) {
             std::vector<PatchInfo<DTYPE>> patchinfos;
             extractor.Extract(track, p_views, p_features, patchinfos);
             return patchinfos;
           });
}

void bind_extractor(py::module &m) {
  using namespace features;

  py::class_<LinePatchExtractorOptions>(m, "LinePatchExtractorOptions")
      .def(py::init<>())
      .def(py::init<py::dict>())
      .def_readwrite("k_stretch", &LinePatchExtractorOptions::k_stretch)
      .def_readwrite("t_stretch", &LinePatchExtractorOptions::t_stretch)
      .def_readwrite("range_perp", &LinePatchExtractorOptions::range_perp);

#define REGISTER_CHANNEL(CHANNELS)                                             \
  bind_extractor_dtype<double, CHANNELS>(m,                                    \
                                         "_f64_c" + std::to_string(CHANNELS));

  REGISTER_CHANNEL(1);
  REGISTER_CHANNEL(2);
  REGISTER_CHANNEL(3);
  REGISTER_CHANNEL(16);
  REGISTER_CHANNEL(32);
  REGISTER_CHANNEL(64);
  REGISTER_CHANNEL(128);

#undef REGISTER_CHANNEL
}

void bind_features(py::module &m) {
  using namespace features;

  m.def("extract_dsift", &extract_dsift, py::arg("image"), py::arg("steps") = 1,
        py::arg("bin_size") = 4, "Extract DSIFT features.");

  bind_patchinfo<float16>(m, "_f16");
  bind_patchinfo<float>(m, "_f32");
  bind_patchinfo<double>(m, "_f64");
  bind_extractor(m);
}

} // namespace limap
