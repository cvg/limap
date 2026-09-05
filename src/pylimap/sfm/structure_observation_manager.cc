#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/sfm/structure_observation_manager.h"

namespace limap {

void BindStructureObservationManager(py::module &m) {
  py::classh<StructureObservationManager>(m, "StructureObservationManager")
      .def(py::init<StructureReconstruction &>(), "structure_reconstruction"_a,
           py::keep_alive<1,
                          2>()) // manager keeps structure_reconstruction alive
      // CRUD operations for lines
      .def("add_line3d", &StructureObservationManager::AddLine3D, "line3d"_a,
           "Add a new 3D line with its track. Returns the new line3D_id.")
      .def("add_line_observation",
           &StructureObservationManager::AddLineObservation, "line3D_id"_a,
           "track_element"_a, "Add a line observation to an existing 3D line.")
      .def("delete_line3d", &StructureObservationManager::DeleteLine3D,
           "line3D_id"_a,
           "Delete a 3D line and clear all its 2D->3D assignments.")
      .def("delete_line_observation",
           &StructureObservationManager::DeleteLineObservation, "image_id"_a,
           "line2D_idx"_a,
           "Delete a line observation from its associated 3D line.")
      .def("merge_lines3d", &StructureObservationManager::MergeLines3D,
           "line3D_id1"_a, "line3D_id2"_a,
           "Merge two 3D lines into one. Returns the ID of the merged line.")
      // CRUD operations for groups
      .def("add_group3d", &StructureObservationManager::AddGroup3D, "group3d"_a,
           "Add a new 3D group with its track. Returns the new group3D_id.")
      .def("add_group_observation",
           &StructureObservationManager::AddGroupObservation, "group3D_id"_a,
           "track_element"_a,
           "Add a group observation to an existing 3D group.")
      .def("delete_group3d", &StructureObservationManager::DeleteGroup3D,
           "group3D_id"_a,
           "Delete a 3D group and clear all its 2D->3D assignments.")
      .def("delete_group_observation",
           &StructureObservationManager::DeleteGroupObservation, "image_id"_a,
           "group2D_idx"_a,
           "Delete a group observation from its associated 3D group.")
      .def("merge_groups3d", &StructureObservationManager::MergeGroups3D,
           "group3D_id1"_a, "group3D_id2"_a,
           "Merge two 3D groups into one. Returns the ID of the merged group.")
      // Frame de-registration
      .def("deregister_frame", &StructureObservationManager::DeRegisterFrame,
           "frame_id"_a,
           "Remove all structure observations (lines + groups) for a frame.")
      // Filtering operations
      .def("filter_lines3d_by_reprojection",
           &StructureObservationManager::FilterLines3dByReprojection,
           "th_angular_2d"_a, "th_perp_2d"_a, "num_outliers"_a = 2)
      .def("filter_lines3d_by_sensitivity",
           &StructureObservationManager::FilterLines3dBySensitivity,
           "th_sensitivity_3d"_a, "min_support_ns"_a)
      .def("filter_lines3d_by_overlap",
           &StructureObservationManager::FilterLines3dByOverlap, "th_overlap"_a,
           "min_support_ns"_a)
      .def("filter_lines3d_by_min_visible_views",
           &StructureObservationManager::FilterLines3dByMinVisibleViews,
           "min_visible_views"_a)
      .def("filter_all_lines3d", &StructureObservationManager::FilterAllLines3D,
           "max_angular_error"_a, "max_perp_error"_a,
           "Filter line observations by reprojection error. "
           "Returns the number of filtered observations.")
      // Active/Inactive operations (soft filtering)
      .def("is_reliable_track", &StructureObservationManager::IsReliableTrack,
           "line3D_id"_a, "min_active"_a = 1,
           "Check if a line track is reliable (enough active observations).")
      .def(
          "classify_line_tracks",
          [](const StructureObservationManager &self,
             size_t min_active_observations) {
            FlatHashSet<line3D_t> reliable, unreliable;
            self.ClassifyLineTracks(min_active_observations, reliable,
                                    unreliable);
            return std::make_pair(reliable, unreliable);
          },
          "min_active_observations"_a,
          "Classify line tracks into (reliable, unreliable) sets.")
      .def("update_line_observation_activity",
           &StructureObservationManager::UpdateLineObservationActivity,
           "max_angular_error"_a, "max_perp_error"_a,
           "Soft filter: mark bad observations as inactive. "
           "Returns number of newly inactivated observations.")
      .def("update_active_supports",
           &StructureObservationManager::UpdateActiveSupports,
           "max_angular_error"_a, "max_perp_error"_a,
           "Re-evaluate all observations. Can reactivate observations. "
           "Returns number of changed observations.")
      .def("delete_supportless_line_tracks",
           &StructureObservationManager::DeleteSupportlessLineTracks,
           "min_active_ratio"_a,
           "Hard-delete tracks with active ratio below threshold. "
           "Returns number of deleted tracks.");
}

} // namespace limap
