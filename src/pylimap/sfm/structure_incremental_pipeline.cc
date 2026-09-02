#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/structure_incremental_pipeline.h"

namespace limap {

void BindStructureIncrementalPipeline(py::module &m) {
  using namespace limap;

  // === StructureIncrementalPipelineOptions ===
  py::classh<StructureIncrementalPipelineOptions> PyPipelineOpts(
      m, "StructureIncrementalPipelineOptions");
  PyPipelineOpts.def(py::init<>())
      .def_readwrite("colmap_options",
                     &StructureIncrementalPipelineOptions::colmap_options,
                     "COLMAP pipeline options")
      .def_readwrite("structure_options",
                     &StructureIncrementalPipelineOptions::structure_options,
                     "Structure mapper options")
      .def("mapper", &StructureIncrementalPipelineOptions::Mapper)
      .def("is_initial_pair_provided",
           &StructureIncrementalPipelineOptions::IsInitialPairProvided)
      .def("local_structure_ba",
           &StructureIncrementalPipelineOptions::LocalStructureBA)
      .def("global_structure_ba",
           &StructureIncrementalPipelineOptions::GlobalStructureBA)
      .def("check", &StructureIncrementalPipelineOptions::Check);
  MakeDataclass(PyPipelineOpts);

  // === Status enum ===
  py::enum_<StructureIncrementalPipeline::Status>(
      m, "StructureIncrementalPipelineStatus")
      .value("SUCCESS", StructureIncrementalPipeline::Status::SUCCESS)
      .value("INTERRUPTED", StructureIncrementalPipeline::Status::INTERRUPTED)
      .value("CONTINUE", StructureIncrementalPipeline::Status::CONTINUE)
      .value("STOP", StructureIncrementalPipeline::Status::STOP)
      .value("NO_INITIAL_PAIR",
             StructureIncrementalPipeline::Status::NO_INITIAL_PAIR)
      .value("BAD_INITIAL_PAIR",
             StructureIncrementalPipeline::Status::BAD_INITIAL_PAIR);

  // === StructureIncrementalPipeline ===
  py::classh<StructureIncrementalPipeline>(m, "StructureIncrementalPipeline")
      .def(py::init<std::shared_ptr<StructureIncrementalPipelineOptions>,
                    std::shared_ptr<colmap::DatabaseCache>,
                    std::shared_ptr<StructureDatabaseCache>,
                    std::shared_ptr<HolisticReconstructionManager>>(),
           "options"_a, "database_cache"_a, "structure_db_cache"_a,
           "reconstruction_manager"_a)
      .def(
          "run",
          [](StructureIncrementalPipeline &self,
             std::function<void()> initial_image_pair_callback,
             std::function<void()> next_image_callback) {
            py::gil_scoped_release release;
            PyInterrupt py_interrupt(1.0);

            // Wrap callbacks with interrupt checking (mirrors COLMAP's
            // pycolmap/pipeline/sfm.cc pattern)
            auto init_cb = [&py_interrupt,
                            cb = std::move(initial_image_pair_callback)]() {
              if (py_interrupt.Raised()) {
                throw py::error_already_set();
              }
              if (cb) {
                cb();
              }
            };
            auto next_cb = [&py_interrupt,
                            cb = std::move(next_image_callback)]() {
              if (py_interrupt.Raised()) {
                throw py::error_already_set();
              }
              if (cb) {
                cb();
              }
            };

            self.Run(std::move(init_cb), std::move(next_cb));
          },
          "initial_image_pair_callback"_a = py::none(),
          "next_image_callback"_a = py::none())
      .def("reconstruct", &StructureIncrementalPipeline::Reconstruct,
           "mapper"_a, "mapper_options"_a, "continue_reconstruction"_a,
           "initial_image_pair_callback"_a = py::none(),
           "next_image_callback"_a = py::none())
      .def("reconstruct_sub_model",
           &StructureIncrementalPipeline::ReconstructSubModel, "mapper"_a,
           "mapper_options"_a, "reconstruction"_a,
           "initial_image_pair_callback"_a = py::none(),
           "next_image_callback"_a = py::none())
      .def("initialize_reconstruction",
           &StructureIncrementalPipeline::InitializeReconstruction, "mapper"_a,
           "mapper_options"_a, "reconstruction"_a)
      .def("check_run_global_refinement",
           &StructureIncrementalPipeline::CheckRunGlobalRefinement,
           "reconstruction"_a, "ba_prev_num_reg_frames"_a,
           "ba_prev_num_points"_a)
      .def_property_readonly("options", &StructureIncrementalPipeline::Options)
      .def_property_readonly(
          "reconstruction_manager",
          &StructureIncrementalPipeline::ReconstructionManager);
}

} // namespace limap
