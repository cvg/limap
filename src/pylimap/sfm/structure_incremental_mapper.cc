#include <optional>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/structure_incremental_mapper.h"

namespace limap {

void BindStructureIncrementalMapper(py::module &m) {
  using namespace limap;

  // === Options ===
  using Options = StructureIncrementalMapper::Options;
  py::classh<Options> PyOpts(m, "StructureIncrementalMapperOptions");
  PyOpts.def(py::init<>())
      .def_readwrite("line_visibility_weight", &Options::line_visibility_weight,
                     "Weight for line visibility in next-image ranking")
      .def_readwrite("use_lines_for_registration",
                     &Options::use_lines_for_registration,
                     "Whether to use lines for hybrid PnPL registration")
      .def_readwrite("registration", &Options::registration,
                     "Point-line absolute pose estimation options")
      .def_readwrite("triangulation", &Options::triangulation,
                     "Structure triangulation options")
      .def_readwrite("structure_ba", &Options::structure_ba,
                     "Structure bundle adjustment options")
      .def_readwrite("filter_max_line_angular_error",
                     &Options::filter_max_line_angular_error,
                     "Max angular error (degrees) for line filtering")
      .def_readwrite("filter_max_line_perp_error",
                     &Options::filter_max_line_perp_error,
                     "Max perpendicular error (pixels) for line filtering")
      .def_readwrite("group_verification", &Options::group_verification,
                     "Post-BA group verification options")
      .def_readwrite("use_two_step_ba", &Options::use_two_step_ba,
                     "Enable 2-step BA (reliable then unreliable lines)")
      .def_readwrite("min_active_line_observations",
                     &Options::min_active_line_observations,
                     "Min active observations for reliable classification")
      .def_readwrite("min_active_ratio_for_deletion",
                     &Options::min_active_ratio_for_deletion,
                     "Min active ratio before hard-deleting a line track")
      .def_readwrite("pixel_uncertainty_threshold",
                     &Options::pixel_uncertainty_threshold,
                     "Covariance-based pixel uncertainty (std dev) threshold "
                     "for reliability classification (0 to disable)")
      .def("check", &Options::Check);
  MakeDataclass(PyOpts);

  // === LocalBundleAdjustmentReport ===
  using Report = StructureIncrementalMapper::LocalBundleAdjustmentReport;
  py::classh<Report> PyReport(m, "StructureLocalBundleAdjustmentReport");
  PyReport.def(py::init<>())
      .def_readonly("num_merged_observations", &Report::num_merged_observations)
      .def_readonly("num_completed_observations",
                    &Report::num_completed_observations)
      .def_readonly("num_filtered_observations",
                    &Report::num_filtered_observations)
      .def_readonly("num_adjusted_observations",
                    &Report::num_adjusted_observations)
      .def_readonly("num_merged_line_observations",
                    &Report::num_merged_line_observations)
      .def_readonly("num_completed_line_observations",
                    &Report::num_completed_line_observations)
      .def_readonly("num_merged_group_observations",
                    &Report::num_merged_group_observations)
      .def_readonly("num_completed_group_observations",
                    &Report::num_completed_group_observations);
  MakeDataclass(PyReport);

  // === StructureIncrementalMapper ===
  py::classh<StructureIncrementalMapper>(m, "StructureIncrementalMapper")
      .def(py::init<std::shared_ptr<const colmap::DatabaseCache>,
                    std::shared_ptr<const StructureDatabaseCache>>(),
           "database_cache"_a, "structure_db_cache"_a)
      .def("begin_reconstruction",
           &StructureIncrementalMapper::BeginReconstruction, "reconstruction"_a,
           py::keep_alive<1, 2>())
      .def("end_reconstruction", &StructureIncrementalMapper::EndReconstruction,
           "discard"_a)
      .def(
          "find_initial_image_pair",
          [](StructureIncrementalMapper &self,
             const colmap::IncrementalMapper::Options &options)
              -> std::optional<std::tuple<colmap::image_t, colmap::image_t,
                                          colmap::Rigid3d>> {
            colmap::image_t image_id1, image_id2;
            colmap::Rigid3d cam2_from_cam1;
            if (self.FindInitialImagePair(options, image_id1, image_id2,
                                          cam2_from_cam1)) {
              return std::make_tuple(image_id1, image_id2, cam2_from_cam1);
            }
            return std::nullopt;
          },
          "options"_a)
      .def("register_initial_image_pair",
           &StructureIncrementalMapper::RegisterInitialImagePair, "options"_a,
           "image_id1"_a, "image_id2"_a, "cam2_from_cam1"_a)
      .def(
          "estimate_initial_two_view_geometry",
          [](StructureIncrementalMapper &self,
             const colmap::IncrementalMapper::Options &options,
             colmap::image_t image_id1,
             colmap::image_t image_id2) -> std::optional<colmap::Rigid3d> {
            colmap::Rigid3d cam2_from_cam1;
            if (self.EstimateInitialTwoViewGeometry(
                    options, image_id1, image_id2, cam2_from_cam1)) {
              return cam2_from_cam1;
            }
            return std::nullopt;
          },
          "options"_a, "image_id1"_a, "image_id2"_a)
      .def("find_next_images", &StructureIncrementalMapper::FindNextImages,
           "options"_a, "structure_options"_a, "structure_less"_a = false)
      .def("register_next_image",
           &StructureIncrementalMapper::RegisterNextImage, "mapper_options"_a,
           "options"_a, "image_id"_a)
      .def("register_next_structure_less_image",
           &StructureIncrementalMapper::RegisterNextStructureLessImage,
           "mapper_options"_a, "image_id"_a)
      .def("triangulate_image", &StructureIncrementalMapper::TriangulateImage,
           "options"_a, "image_id"_a)
      .def("adjust_local_bundle",
           &StructureIncrementalMapper::AdjustLocalBundle, "mapper_options"_a,
           "options"_a, "image_id"_a)
      .def("adjust_global_bundle",
           &StructureIncrementalMapper::AdjustGlobalBundle, "mapper_options"_a,
           "options"_a)
      .def("iterative_local_refinement",
           &StructureIncrementalMapper::IterativeLocalRefinement,
           "max_num_refinements"_a, "max_refinement_change"_a,
           "mapper_options"_a, "options"_a, "image_id"_a)
      .def("iterative_global_refinement",
           &StructureIncrementalMapper::IterativeGlobalRefinement,
           "max_num_refinements"_a, "max_refinement_change"_a,
           "mapper_options"_a, "options"_a, "normalize_reconstruction"_a = true)
      .def("filter_frames", &StructureIncrementalMapper::FilterFrames,
           "options"_a)
      .def("filter_points", &StructureIncrementalMapper::FilterPoints,
           "options"_a)
      .def("soft_filter_lines", &StructureIncrementalMapper::SoftFilterLines,
           "options"_a,
           "Soft filter: tag observations as inactive instead of deleting")
      .def("complete_and_merge_tracks",
           &StructureIncrementalMapper::CompleteAndMergeTracks, "options"_a)
      .def("retriangulate", &StructureIncrementalMapper::Retriangulate,
           "options"_a)
      .def("colmap_mapper",
           py::overload_cast<>(&StructureIncrementalMapper::ColmapMapper),
           py::return_value_policy::reference_internal)
      .def("structure_triangulator",
           &StructureIncrementalMapper::StructureTriangulator,
           py::return_value_policy::reference_internal)
      .def_property_readonly("reconstruction",
                             &StructureIncrementalMapper::Reconstruction)
      .def_property_readonly("num_total_reg_images",
                             &StructureIncrementalMapper::NumTotalRegImages)
      .def_property_readonly("num_shared_reg_images",
                             &StructureIncrementalMapper::NumSharedRegImages)
      .def("reset_initialization_stats",
           &StructureIncrementalMapper::ResetInitializationStats);
}

} // namespace limap
