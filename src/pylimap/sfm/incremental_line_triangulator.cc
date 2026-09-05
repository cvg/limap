#include <optional>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/estimators/bundle_adjustment/bundle_adjustment.h"
#include "limap/sfm/incremental_line_triangulator.h"

namespace limap {

void BindIncrementalLineTriangulator(py::module &m) {
  using namespace limap;

  using Options = IncrementalLineTriangulator::Options;
  py::classh<Options> PyILTOpts(m, "IncrementalLineTriangulatorOptions");
  PyILTOpts.def(py::init<>())
      .def_readwrite("max_transitivity", &Options::max_transitivity)
      .def_readwrite("exhaustive_match_neighbors",
                     &Options::exhaustive_match_neighbors)
      .def_readwrite("use_point_assisted_triangulation",
                     &Options::use_point_assisted_triangulation)
      .def_readwrite("use_vp_assisted_triangulation",
                     &Options::use_vp_assisted_triangulation)
      .def_readwrite("min_length_2d", &Options::min_length_2d)
      .def_readwrite("iou_threshold", &Options::IoU_threshold)
      .def_readwrite("sensitivity_threshold", &Options::sensitivity_threshold)
      .def_readwrite("var2d", &Options::var2d)
      .def_readwrite("fullscore_th", &Options::fullscore_th)
      .def_readwrite("scale_inv_th", &Options::scale_inv_th)
      .def_readwrite("linker2d_options", &Options::linker2d_options)
      .def_readwrite("linker3d_options", &Options::linker3d_options)
      .def_readwrite("create_max_angle_error", &Options::create_max_angle_error)
      .def_readwrite("continue_max_angle_error",
                     &Options::continue_max_angle_error)
      .def_readwrite("merge_max_reproj_error", &Options::merge_max_reproj_error)
      .def_readwrite("merge_max_angle_error_2d",
                     &Options::merge_max_angle_error_2d)
      .def_readwrite("complete_max_reproj_error",
                     &Options::complete_max_reproj_error)
      .def_readwrite("complete_max_transitivity",
                     &Options::complete_max_transitivity)
      .def_readwrite("re_max_angle_error", &Options::re_max_angle_error)
      .def_readwrite("re_min_ratio", &Options::re_min_ratio)
      .def_readwrite("re_max_trials", &Options::re_max_trials)
      .def_readwrite("min_angle", &Options::min_angle)
      .def_readwrite("ignore_two_view_tracks", &Options::ignore_two_view_tracks)
      .def_readwrite("num_threads", &Options::num_threads)
      .def("check", &Options::Check);
  MakeDataclass(PyILTOpts);

  py::classh<IncrementalLineTriangulator>(m, "IncrementalLineTriangulator")
      .def(py::init<std::shared_ptr<const StructureCorrespondenceGraph>,
                    HolisticReconstruction &,
                    std::shared_ptr<StructureObservationManager>>(),
           "structure_correspondence_graph"_a, "reconstruction"_a,
           "obs_manager"_a = nullptr)
      .def("triangulate_image", &IncrementalLineTriangulator::TriangulateImage,
           "options"_a, "image_id"_a)
      .def("complete_image", &IncrementalLineTriangulator::CompleteImage,
           "options"_a, "image_id"_a)
      .def("complete_tracks", &IncrementalLineTriangulator::CompleteTracks,
           "options"_a, "line3d_ids"_a)
      .def("complete_all_tracks",
           &IncrementalLineTriangulator::CompleteAllTracks, "options"_a)
      .def("merge_tracks", &IncrementalLineTriangulator::MergeTracks,
           "options"_a, "line3d_ids"_a)
      .def("merge_all_tracks", &IncrementalLineTriangulator::MergeAllTracks,
           "options"_a)
      .def("retriangulate", &IncrementalLineTriangulator::Retriangulate,
           "options"_a)
      .def("add_modified_line3d",
           &IncrementalLineTriangulator::AddModifiedLine3D, "line3d_id"_a)
      .def("get_modified_lines3d",
           &IncrementalLineTriangulator::GetModifiedLines3D)
      .def("clear_modified_lines3d",
           &IncrementalLineTriangulator::ClearModifiedLines3D);

  m.def(
      "incremental_line_triangulation_pipeline",
      [](const StructureDatabaseCache &sdc, HolisticReconstruction &recon,
         const IncrementalLineTriangulator::Options &opts,
         std::optional<estimators::PointLineBundleAdjustmentOptions> ba) {
        IncrementalLineTriangulationPipeline(sdc, recon, opts,
                                             ba ? &*ba : nullptr);
      },
      "structure_db_cache"_a, "reconstruction"_a, "options"_a,
      "ba_options"_a = py::none());
}

} // namespace limap
