#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/fitted_line_merging.h"

namespace limap {

void BindFittedLineMerging(py::module &m) {
  py::classh<LineMergingOptions> PyLMOpts(m, "LineMergingOptions");
  PyLMOpts.def(py::init<>())
      .def_readwrite("linker2d_options", &LineMergingOptions::linker2d_options)
      .def_readwrite("linker3d_options", &LineMergingOptions::linker3d_options)
      .def_readwrite("linker3d_remerge_options",
                     &LineMergingOptions::linker3d_remerge_options)
      .def_readwrite("num_outliers_aggregator",
                     &LineMergingOptions::num_outliers_aggregator)
      .def_readwrite("enable_remerge", &LineMergingOptions::enable_remerge)
      .def_readwrite("filtering2d_th_angular_2d",
                     &LineMergingOptions::filtering2d_th_angular_2d)
      .def_readwrite("filtering2d_th_perp_2d",
                     &LineMergingOptions::filtering2d_th_perp_2d)
      .def_readwrite("min_visible_views",
                     &LineMergingOptions::min_visible_views)
      .def_readwrite("num_threads", &LineMergingOptions::num_threads)
      .def("check", &LineMergingOptions::Check);
  MakeDataclass(PyLMOpts);

  m.def(
      "merge_fitted_lines_3d",
      [](HolisticReconstruction &recon,
         const std::map<colmap::image_t, std::vector<colmap::image_t>>
             &neighbors,
         const LineMergingOptions &options) {
        MergeFittedLines3D(recon, neighbors, options);
      },
      "recon"_a, "neighbors"_a, "options"_a,
      "Merge pre-fitted Line3d based on 2D+3D similarity. Modifies recon "
      "in-place.");
}

} // namespace limap
