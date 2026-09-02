#include <optional>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/sfm/global_line_triangulation.h"

namespace limap {

void BindGlobalLineTriangulation(py::module &m) {
  using namespace limap;

  py::classh<GlobalLineTriangulationOptions> PyGLOpts(
      m, "GlobalLineTriangulationOptions");
  PyGLOpts.def(py::init<>())
      .def_readwrite("wireframe2d_th",
                     &GlobalLineTriangulationOptions::wireframe2d_th)
      .def_readwrite(
          "use_point_assisted_triangulation",
          &GlobalLineTriangulationOptions::use_point_assisted_triangulation)
      .def_readwrite(
          "use_vp_assisted_triangulation",
          &GlobalLineTriangulationOptions::use_vp_assisted_triangulation)
      .def_readwrite("min_length_2d",
                     &GlobalLineTriangulationOptions::min_length_2d)
      .def_readwrite("iou_threshold",
                     &GlobalLineTriangulationOptions::IoU_threshold)
      .def_readwrite("sensitivity_threshold",
                     &GlobalLineTriangulationOptions::sensitivity_threshold)
      .def_readwrite("var2d", &GlobalLineTriangulationOptions::var2d)
      .def_readwrite("scale_inv_th",
                     &GlobalLineTriangulationOptions::scale_inv_th)
      .def_readwrite("fullscore_th",
                     &GlobalLineTriangulationOptions::fullscore_th)
      .def_readwrite("num_outliers_aggregator",
                     &GlobalLineTriangulationOptions::num_outliers_aggregator)
      .def_readwrite("linker2d_options",
                     &GlobalLineTriangulationOptions::linker2d_options)
      .def_readwrite("linker3d_options",
                     &GlobalLineTriangulationOptions::linker3d_options)
      .def_readwrite("enable_remerge",
                     &GlobalLineTriangulationOptions::enable_remerge)
      .def_readwrite("filtering2d_th_angular_2d",
                     &GlobalLineTriangulationOptions::filtering2d_th_angular_2d)
      .def_readwrite("filtering2d_th_perp_2d",
                     &GlobalLineTriangulationOptions::filtering2d_th_perp_2d)
      .def_readwrite(
          "filtering2d_th_sv_angular_3d",
          &GlobalLineTriangulationOptions::filtering2d_th_sv_angular_3d)
      .def_readwrite(
          "filtering2d_th_sv_num_supports",
          &GlobalLineTriangulationOptions::filtering2d_th_sv_num_supports)
      .def_readwrite("filtering2d_th_overlap",
                     &GlobalLineTriangulationOptions::filtering2d_th_overlap)
      .def_readwrite(
          "filtering2d_th_overlap_num_supports",
          &GlobalLineTriangulationOptions::filtering2d_th_overlap_num_supports)
      .def_readwrite("min_visible_views",
                     &GlobalLineTriangulationOptions::min_visible_views)
      .def_readwrite("num_threads",
                     &GlobalLineTriangulationOptions::num_threads)
      .def("check", &GlobalLineTriangulationOptions::Check);
  MakeDataclass(PyGLOpts);

  py::classh<GlobalLineTriangulationController>(
      m, "GlobalLineTriangulationController")
      .def(py::init<const GlobalLineTriangulationOptions &,
                    const std::shared_ptr<HolisticReconstruction> &,
                    const colmap::CorrespondenceGraph &,
                    const ExhaustiveMatchNeighbors &>(),
           "options"_a, "reconstruction"_a, "correspondence_graph"_a,
           "exhaustive_match_neighbors"_a = ExhaustiveMatchNeighbors())
      .def("run", &GlobalLineTriangulationController::Run);

  m.def(
      "global_line_triangulation",
      [](const GlobalLineTriangulationOptions &options,
         const std::shared_ptr<HolisticReconstruction> &recon,
         const colmap::CorrespondenceGraph &corr_graph,
         const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
        GlobalLineTriangulationController ctrl(options, recon, corr_graph,
                                               exhaustive_match_neighbors);
        ctrl.Run();
      },
      "options"_a, "reconstruction"_a, "correspondence_graph"_a,
      "exhaustive_match_neighbors"_a = ExhaustiveMatchNeighbors());

  m.def(
      "global_line_triangulation_pipeline",
      [](const StructureDatabaseCache &sdc,
         const std::shared_ptr<HolisticReconstruction> &recon,
         const GlobalLineTriangulationOptions &opts,
         std::optional<estimators::PointLineBundleAdjustmentOptions> ba,
         const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
        GlobalLineTriangulationPipeline(sdc, recon, opts, ba ? &*ba : nullptr,
                                        exhaustive_match_neighbors);
      },
      "structure_db_cache"_a, "reconstruction"_a, "options"_a,
      "ba_options"_a = py::none(),
      "exhaustive_match_neighbors"_a = ExhaustiveMatchNeighbors());
}

} // namespace limap
