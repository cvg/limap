#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ceres/loss_function.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/line_pixel_uncertainty.h"
#include "limap/geometry/minimal_inf_line3d.h"
#include "limap/scene/structure_reconstruction.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace limap {

void BindStructureReconstruction(py::module &m) {
  // WireframeVotingOptions
  py::classh<WireframeVotingOptions>(m, "WireframeVotingOptions")
      .def(py::init<>())
      .def_readwrite("min_num_votes", &WireframeVotingOptions::min_num_votes)
      .def_readwrite("min_weight_sum", &WireframeVotingOptions::min_weight_sum);

  py::classh<StructureReconstruction>(m, "StructureReconstruction")
      .def(py::init<const colmap::Reconstruction &>(), py::arg("point_recon"))
      .def("num_lines3D", &StructureReconstruction::NumLines3D)
      .def("num_groups3D", &StructureReconstruction::NumGroups3D)
      .def("read", &StructureReconstruction::Read)
      .def("write", &StructureReconstruction::Write)
      .def("read_binary", &StructureReconstruction::ReadBinary)
      .def("read_text", &StructureReconstruction::ReadText)
      .def("write_binary", &StructureReconstruction::WriteBinary)
      .def("write_text", &StructureReconstruction::WriteText)
      .def("load", &StructureReconstruction::Load,
           py::arg("structure_database"))
      .def("initialize_all_wireframes",
           &StructureReconstruction::InitializeAllWireframes,
           py::arg("threshold") = 2.0)
      .def_property_readonly("point_recon",
                             &StructureReconstruction::PointRecon,
                             py::return_value_policy::reference_internal)
      .def("structure2d",
           static_cast<const Structure2d &(
               StructureReconstruction::*)(const colmap::image_t) const>(
               &StructureReconstruction::Structure2d),
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "structures2d",
          static_cast<const NodeHashMap<colmap::image_t, class Structure2d> &(
              StructureReconstruction::*)() const>(
              &StructureReconstruction::Structures2d),
          py::return_value_policy::reference_internal)
      .def("line",
           static_cast<const Line3dWithActiveLabels &(
               StructureReconstruction::*)(const line3D_t) const>(
               &StructureReconstruction::Line),
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "lines3D",
          static_cast<const NodeHashMap<line3D_t, Line3dWithActiveLabels> &(
              StructureReconstruction::*)() const>(
              &StructureReconstruction::Lines3D),
          py::return_value_policy::reference_internal)
      .def("group",
           static_cast<const Group3dWithActiveLabels &(
               StructureReconstruction::*)(const group3D_t) const>(
               &StructureReconstruction::Group),
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "groups3D",
          static_cast<const NodeHashMap<group3D_t, Group3dWithActiveLabels> &(
              StructureReconstruction::*)() const>(
              &StructureReconstruction::Groups3D),
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "wireframe",
          static_cast<const Wireframe3d &(StructureReconstruction::*)() const>(
              &StructureReconstruction::Wireframe),
          py::return_value_policy::reference_internal)
      // lines3D mutation (aligned with COLMAP Reconstruction interface)
      .def("exists_line3D", &StructureReconstruction::ExistsLine3D, "id"_a)
      .def("add_line3D",
           static_cast<line3D_t (StructureReconstruction::*)(const Line3d &)>(
               &StructureReconstruction::AddLine3D),
           "line"_a, "Add a 3D line and return the assigned line3D_id")
      .def("add_line3D",
           static_cast<void (StructureReconstruction::*)(
               line3D_t, const Line3d &)>(&StructureReconstruction::AddLine3D),
           "id"_a, "line"_a, "Add a 3D line with a specific ID")
      .def("add_line_observation", &StructureReconstruction::AddLineObservation,
           "line3D_id"_a, "track_element"_a,
           "Add a line observation to an existing 3D line")
      .def("merge_lines3D", &StructureReconstruction::MergeLines3D,
           "line3D_id1"_a, "line3D_id2"_a, "Merge two 3D lines into one")
      .def("delete_line_observation",
           &StructureReconstruction::DeleteLineObservation, "image_id"_a,
           "line2D_idx"_a, "Delete a line observation from its 3D line")
      .def("delete_line3D", &StructureReconstruction::DeleteLine3D, "id"_a)
      .def("clear_lines3D", &StructureReconstruction::ClearLines3D)
      .def("line3D_ids", &StructureReconstruction::Line3DIds)
      // structures2d mutation
      .def("exists_structure2d", &StructureReconstruction::ExistsStructure2D,
           "id"_a)
      .def("add_structure2d", &StructureReconstruction::AddStructure2D, "id"_a,
           "structure"_a)
      .def("delete_structure2d", &StructureReconstruction::DeleteStructure2D,
           "id"_a)
      .def("clear_structures2d", &StructureReconstruction::ClearStructures2D)
      // groups3D mutation (aligned with COLMAP Reconstruction interface)
      .def("exists_group3D", &StructureReconstruction::ExistsGroup3D, "id"_a)
      .def("add_group3D",
           static_cast<group3D_t (StructureReconstruction::*)(const Group3d &)>(
               &StructureReconstruction::AddGroup3D),
           "group"_a, "Add a 3D group and return the assigned group3D_id")
      .def("add_group3D",
           static_cast<void (StructureReconstruction::*)(group3D_t,
                                                         const Group3d &)>(
               &StructureReconstruction::AddGroup3D),
           "id"_a, "group"_a, "Add a 3D group with a specific ID")
      .def("add_group_observation",
           &StructureReconstruction::AddGroupObservation, "group3D_id"_a,
           "track_element"_a, "Add a group observation to an existing 3D group")
      .def("merge_groups3D", &StructureReconstruction::MergeGroups3D,
           "group3D_id1"_a, "group3D_id2"_a, "Merge two 3D groups into one")
      .def("delete_group_observation",
           &StructureReconstruction::DeleteGroupObservation, "image_id"_a,
           "group2D_idx"_a, "Delete a group observation from its 3D group")
      .def("delete_group3D", &StructureReconstruction::DeleteGroup3D, "id"_a)
      .def("clear_groups3D", &StructureReconstruction::ClearGroups3D)
      .def("group3D_ids", &StructureReconstruction::Group3DIds)
      // Wireframe 3D construction from 2D via voting
      .def("construct_wireframe3d_from_2d",
           &StructureReconstruction::ConstructWireframe3dFrom2d,
           py::arg("options") = WireframeVotingOptions(),
           "Construct 3D wireframe from 2D associations via voting. "
           "Edges meeting vote count and weight thresholds are added.")
      // Pixel uncertainty computation with backprojected endpoints
      .def(
          "compute_line_pixel_uncertainties",
          [](const StructureReconstruction &self,
             double cauchy_scale) -> FlatHashMap<line3D_t, double> {
            const auto &point_recon = self.PointRecon();
            FlatHashMap<line3D_t, double> uncertainties;

            // Create Cauchy loss if requested (matches BA loss function)
            std::unique_ptr<ceres::LossFunction> loss;
            if (cauchy_scale > 0) {
              loss = std::make_unique<ceres::CauchyLoss>(cauchy_scale);
            }

            for (const auto &[line_id, line] : self.Lines3D()) {
              const auto &track = line.track;
              if (track.Length() < 2) {
                continue;
              }

              // Gather observations
              std::vector<Eigen::Quaterniond> rotations;
              std::vector<Eigen::Vector3d> translations;
              std::vector<Eigen::Vector4d> kvecs;
              std::vector<Line2d> lines2d;
              rotations.reserve(track.Length());
              translations.reserve(track.Length());
              kvecs.reserve(track.Length());
              lines2d.reserve(track.Length());

              for (const auto &elem : track.Elements()) {
                const colmap::image_t image_id = elem.image_id;
                const line2D_t line2D_idx =
                    static_cast<line2D_t>(elem.point2D_idx);

                if (!point_recon.ExistsImage(image_id) ||
                    !self.ExistsStructure2D(image_id)) {
                  continue;
                }

                const colmap::Image &image = point_recon.Image(image_id);
                const colmap::Camera &cam =
                    point_recon.Camera(image.CameraId());

                // Get pose
                const auto cam_from_world = image.CamFromWorld();
                rotations.push_back(cam_from_world.rotation());
                translations.push_back(cam_from_world.translation());

                // Get kvec from camera
                double kvec_arr[4];
                ParamsToKvec(cam.model_id, cam.params.data(), kvec_arr);
                kvecs.emplace_back(kvec_arr[0], kvec_arr[1], kvec_arr[2],
                                   kvec_arr[3]);

                const Line2d &line2d =
                    self.Structure2d(image_id).Line(line2D_idx);
                lines2d.push_back(line2d);
              }

              if (rotations.size() < 2) {
                continue;
              }

              // Convert line to MinimalPlucker parameters
              MinimalInfiniteLine3d min_line(line);
              const double *params = min_line.data.data();

              // Compute uncertainty (pixel std dev)
              double uncertainty = ComputeLinePixelUncertainty(
                  params, line, rotations, translations, kvecs, lines2d,
                  loss.get());
              uncertainties[line_id] = uncertainty;
            }

            return uncertainties;
          },
          py::arg("cauchy_scale") = 1.0,
          "Compute pixel uncertainty (std dev) for all lines using "
          "backprojected "
          "endpoints. Returns dict mapping line3D_id to uncertainty (-1 = not "
          "computable, inf = degenerate). Uses Cauchy loss (matching BA).")
      .def("__repr__", [](const StructureReconstruction &s) {
        std::ostringstream oss;
        oss << "StructureReconstruction(num_lines3D=" << s.NumLines3D()
            << ", num_groups3D=" << s.NumGroups3D()
            << ", num_wf_edges=" << s.Wireframe().CountEdges() << ")";
        return oss.str();
      });
}

} // namespace limap
