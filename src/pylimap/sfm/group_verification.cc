#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/group_verification.h"

namespace limap {

void BindGroupVerification(py::module &m) {
  // GroupVerificationOptions
  py::classh<GroupVerificationOptions> PyGroupVerificationOptions(
      m, "GroupVerificationOptions");
  PyGroupVerificationOptions.def(py::init<>())
      .def_readwrite("inlier_threshold",
                     &GroupVerificationOptions::inlier_threshold,
                     "Threshold for inlier determination (-1 = use default)")
      .def_readwrite("default_vp_threshold",
                     &GroupVerificationOptions::default_vp_threshold,
                     "Default VP threshold in degrees (angular error)")
      .def_readwrite("default_reproj_threshold",
                     &GroupVerificationOptions::default_reproj_threshold,
                     "Default reprojection threshold in pixels")
      .def_readwrite("min_num_inliers",
                     &GroupVerificationOptions::min_num_inliers,
                     "Minimum inlier count to validate")
      .def_readwrite("min_inlier_ratio",
                     &GroupVerificationOptions::min_inlier_ratio,
                     "Minimum inlier ratio to validate")
      .def_readwrite("obs_inlier_ratio",
                     &GroupVerificationOptions::obs_inlier_ratio,
                     "For multi-view: ratio of observations that must pass")
      .def_readwrite("filter_outliers",
                     &GroupVerificationOptions::filter_outliers,
                     "Whether to filter outlier associations");
  MakeDataclass(PyGroupVerificationOptions);

  // FilterGroupsStats
  py::classh<FilterGroupsStats> PyFilterGroupsStats(m, "FilterGroupsStats");
  PyFilterGroupsStats.def(py::init<>())
      .def_readonly("num_groups_processed",
                    &FilterGroupsStats::num_groups_processed,
                    "Total groups processed")
      .def_readonly("num_groups_passed", &FilterGroupsStats::num_groups_passed,
                    "Number of groups that passed verification")
      .def_readonly("num_groups_failed", &FilterGroupsStats::num_groups_failed,
                    "Number of groups that failed verification")
      .def_readonly("num_associations_marked",
                    &FilterGroupsStats::num_associations_marked,
                    "Number of associations marked inactive")
      .def_readonly("num_associations_purged",
                    &FilterGroupsStats::num_associations_purged,
                    "Number of inactive associations permanently removed");
  MakeDataclass(PyFilterGroupsStats);

  // FilterGroupAssociations (replaces FilterGroupsByError)
  m.def(
      "filter_group_associations", &FilterGroupAssociations, py::arg("recon"),
      py::arg("options") = GroupVerificationOptions(),
      py::arg("min_active_for_purge") = 10,
      "Soft-filter all Group3D associations by error. "
      "Marks bad associations inactive, purges for groups with enough active. "
      "Never deletes groups. Returns FilterGroupsStats.");

  // DeleteSupportlessGroups
  m.def("delete_supportless_groups", &DeleteSupportlessGroups, py::arg("recon"),
        "Delete groups with zero active associations or track length < 2. "
        "Returns number of groups deleted.");

  m.def("verify_vp_matches", &VerifyVPMatches, py::arg("recon"),
        py::arg("structure_db"), py::arg("threshold_degrees") = 10.0,
        "Verify all VP matches in the database and update in place. "
        "Returns number of removed matches.");

  m.def("verify_plane_matches", &VerifyPlaneMatches, py::arg("recon"),
        py::arg("structure_db"), py::arg("threshold_degrees") = 30.0,
        "Verify all plane matches in the database and update in place. "
        "Compares camera-frame normals (from Group2d params) transformed to "
        "world frame. Returns number of removed matches.");
}

} // namespace limap
