#include <optional>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/sfm/group_verification.h"
#include "limap/sfm/incremental_structure_triangulator.h"

namespace limap {

void BindIncrementalStructureTriangulator(py::module &m) {
  using namespace limap;

  using Options = IncrementalStructureTriangulator::Options;
  py::classh<Options> PyISTOpts(m, "IncrementalStructureTriangulatorOptions");
  PyISTOpts.def(py::init<>())
      .def_readwrite("triangulate_points", &Options::triangulate_points)
      .def_readwrite("triangulate_lines", &Options::triangulate_lines)
      .def_readwrite("triangulate_groups", &Options::triangulate_groups)
      .def_readwrite("complete_tracks", &Options::complete_tracks)
      .def_readwrite("merge_tracks", &Options::merge_tracks)
      .def_readwrite("point_options", &Options::point_options)
      .def_readwrite("line_options", &Options::line_options)
      .def_readwrite("group_options", &Options::group_options)
      .def("check", &Options::Check);
  MakeDataclass(PyISTOpts);

  py::classh<IncrementalStructureTriangulator>(
      m, "IncrementalStructureTriangulator")
      .def(py::init<std::shared_ptr<const colmap::CorrespondenceGraph>,
                    std::shared_ptr<const StructureCorrespondenceGraph>,
                    HolisticReconstruction &>(),
           "point_corr_graph"_a, "structure_corr_graph"_a, "reconstruction"_a,
           py::keep_alive<1, 4>()) // Keep reconstruction alive
      .def("triangulate_image",
           &IncrementalStructureTriangulator::TriangulateImage, "options"_a,
           "image_id"_a)
      .def("complete_image", &IncrementalStructureTriangulator::CompleteImage,
           "options"_a, "image_id"_a)
      .def("complete_all_tracks",
           &IncrementalStructureTriangulator::CompleteAllTracks, "options"_a)
      .def("merge_all_tracks",
           &IncrementalStructureTriangulator::MergeAllTracks, "options"_a)
      .def("point_triangulator",
           py::overload_cast<>(
               &IncrementalStructureTriangulator::PointTriangulator),
           py::return_value_policy::reference_internal)
      .def("line_triangulator",
           py::overload_cast<>(
               &IncrementalStructureTriangulator::LineTriangulator),
           py::return_value_policy::reference_internal)
      .def("group_triangulator",
           py::overload_cast<>(
               &IncrementalStructureTriangulator::GroupTriangulator),
           py::return_value_policy::reference_internal)
      .def("get_modified_points3d",
           &IncrementalStructureTriangulator::GetModifiedPoints3D)
      .def("get_modified_lines3d",
           &IncrementalStructureTriangulator::GetModifiedLines3D)
      .def("get_modified_groups3d",
           &IncrementalStructureTriangulator::GetModifiedGroups3D)
      .def("clear_modified", &IncrementalStructureTriangulator::ClearModified);

  m.def(
      "incremental_triangulate_structure",
      [](std::shared_ptr<const colmap::CorrespondenceGraph> pcg,
         const StructureDatabaseCache &sdc, HolisticReconstruction &recon,
         const IncrementalStructureTriangulator::Options &opts,
         std::optional<estimators::StructureBundleAdjustmentOptions> ba,
         std::optional<GroupVerificationOptions> fo) {
        IncrementalTriangulateStructure(std::move(pcg), sdc, recon, opts,
                                        ba ? &*ba : nullptr,
                                        fo ? &*fo : nullptr);
      },
      "point_corr_graph"_a, "structure_db_cache"_a, "reconstruction"_a,
      "options"_a, "ba_options"_a = py::none(),
      "filter_options"_a = py::none());
}

} // namespace limap
