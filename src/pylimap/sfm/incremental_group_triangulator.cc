#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/incremental_group_triangulator.h"

namespace limap {

void BindIncrementalGroupTriangulator(py::module &m) {
  using namespace limap;

  using Options = IncrementalGroupTriangulator::Options;
  py::classh<Options> PyIGTOpts(m, "IncrementalGroupTriangulatorOptions");
  PyIGTOpts.def(py::init<>())
      .def_readwrite("max_transitivity", &Options::max_transitivity)
      .def_readwrite("min_num_observations", &Options::min_num_observations)
      .def_readwrite("min_num_supports_for_association",
                     &Options::min_num_supports_for_association)
      .def_readwrite("init_ransac_thresh", &Options::init_ransac_thresh)
      .def_readwrite("verification_threshold", &Options::verification_threshold)
      .def_readwrite("default_vp_threshold", &Options::default_vp_threshold)
      .def_readwrite("default_reproj_threshold",
                     &Options::default_reproj_threshold)
      .def_readwrite("verification_min_num_inliers",
                     &Options::verification_min_num_inliers)
      .def_readwrite("verification_min_inlier_ratio",
                     &Options::verification_min_inlier_ratio)
      .def_readwrite("verification_obs_inlier_ratio",
                     &Options::verification_obs_inlier_ratio)
      .def_readwrite("verification_filter_outliers",
                     &Options::verification_filter_outliers)
      .def_readwrite("continue_max_reproj_error",
                     &Options::continue_max_reproj_error)
      .def_readwrite("continue_max_angle_error_vp",
                     &Options::continue_max_angle_error_vp)
      .def_readwrite("complete_max_reproj_error",
                     &Options::complete_max_reproj_error)
      .def_readwrite("complete_max_angle_error_vp",
                     &Options::complete_max_angle_error_vp)
      .def_readwrite("complete_max_transitivity",
                     &Options::complete_max_transitivity)
      .def_readwrite("merge_max_reproj_error", &Options::merge_max_reproj_error)
      .def_readwrite("merge_max_angle_error_vp",
                     &Options::merge_max_angle_error_vp)
      .def_readwrite("merge_max_error_ratio", &Options::merge_max_error_ratio)
      .def_readwrite("merge_max_plane_normal_angle",
                     &Options::merge_max_plane_normal_angle)
      .def_readwrite("association_filter_min_num_features",
                     &Options::association_filter_min_num_features)
      .def_readwrite("association_filter_reproj_threshold",
                     &Options::association_filter_reproj_threshold)
      .def_readwrite("min_active_to_keep_params",
                     &Options::min_active_to_keep_params)
      .def_readwrite("surface_dist_mad_factor",
                     &Options::surface_dist_mad_factor)
      .def_readwrite("ignore_two_view_tracks", &Options::ignore_two_view_tracks)
      .def("check", &Options::Check);
  MakeDataclass(PyIGTOpts);

  py::classh<IncrementalGroupTriangulator>(m, "IncrementalGroupTriangulator")
      .def(py::init<std::shared_ptr<const StructureCorrespondenceGraph>,
                    HolisticReconstruction &,
                    std::shared_ptr<StructureObservationManager>>(),
           "structure_correspondence_graph"_a, "reconstruction"_a,
           "obs_manager"_a = nullptr)
      .def("triangulate_image", &IncrementalGroupTriangulator::TriangulateImage,
           "options"_a, "image_id"_a)
      .def("complete_image", &IncrementalGroupTriangulator::CompleteImage,
           "options"_a, "image_id"_a)
      .def("complete_tracks", &IncrementalGroupTriangulator::CompleteTracks,
           "options"_a, "group3d_ids"_a)
      .def("complete_all_tracks",
           &IncrementalGroupTriangulator::CompleteAllTracks, "options"_a)
      .def("merge_tracks", &IncrementalGroupTriangulator::MergeTracks,
           "options"_a, "group3d_ids"_a)
      .def("merge_all_tracks", &IncrementalGroupTriangulator::MergeAllTracks,
           "options"_a)
      .def("add_modified_group3d",
           &IncrementalGroupTriangulator::AddModifiedGroup3D, "group3d_id"_a)
      .def("get_modified_groups3d",
           &IncrementalGroupTriangulator::GetModifiedGroups3D)
      .def("clear_modified_groups3d",
           &IncrementalGroupTriangulator::ClearModifiedGroups3D);
}

} // namespace limap
