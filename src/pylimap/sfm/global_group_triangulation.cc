#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/sfm/global_group_triangulation.h"

namespace limap {

void BindGlobalGroupTriangulation(py::module &m) {
  py::classh<GlobalGroupTriangulationOptions> PyGlobalGroupTriangulationOptions(
      m, "GlobalGroupTriangulationOptions");
  PyGlobalGroupTriangulationOptions.def(py::init<>())
      .def_readwrite("group_types",
                     &GlobalGroupTriangulationOptions::group_types)
      .def_readwrite("init_ransac_thresh",
                     &GlobalGroupTriangulationOptions::init_ransac_thresh)
      .def_readwrite("min_cluster_size",
                     &GlobalGroupTriangulationOptions::min_cluster_size)
      .def_readwrite("min_shared_features_for_group_connection",
                     &GlobalGroupTriangulationOptions::
                         min_shared_features_for_group_connection)
      .def_readwrite(
          "min_num_supports_for_association",
          &GlobalGroupTriangulationOptions::min_num_supports_for_association)
      // Verification options
      .def_readwrite("enable_verification",
                     &GlobalGroupTriangulationOptions::enable_verification,
                     "Enable group verification after parameter initialization")
      .def_readwrite(
          "verification_threshold",
          &GlobalGroupTriangulationOptions::verification_threshold,
          "Inlier threshold per group type (-1 or missing = use default). "
          "VP: degrees, Plane/Sphere/etc: pixels")
      .def_readwrite(
          "verification_min_num_inliers",
          &GlobalGroupTriangulationOptions::verification_min_num_inliers,
          "Minimum inlier count to validate group")
      .def_readwrite(
          "verification_min_inlier_ratio",
          &GlobalGroupTriangulationOptions::verification_min_inlier_ratio,
          "Minimum inlier ratio to validate group")
      .def_readwrite(
          "verification_obs_inlier_ratio",
          &GlobalGroupTriangulationOptions::verification_obs_inlier_ratio,
          "For multi-view: ratio of views that must pass")
      .def_readwrite(
          "verification_filter_outliers",
          &GlobalGroupTriangulationOptions::verification_filter_outliers,
          "Whether to filter outlier associations from validated groups")
      .def_readwrite(
          "verification_default_vp_threshold",
          &GlobalGroupTriangulationOptions::verification_default_vp_threshold,
          "Default VP verification threshold in degrees (when not set "
          "per-type)")
      .def_readwrite(
          "verification_default_reproj_threshold",
          &GlobalGroupTriangulationOptions::
              verification_default_reproj_threshold,
          "Default reprojection threshold in pixels (when not set per-type)")
      .def_readwrite(
          "pixel_uncertainty_threshold",
          &GlobalGroupTriangulationOptions::pixel_uncertainty_threshold,
          "Pixel uncertainty (std dev) threshold for line classification "
          "(0 = disable). Lines above threshold are classified as unreliable "
          "and refined separately in 2-step BA.")
      .def("get_effective_group_types",
           &GlobalGroupTriangulationOptions::GetEffectiveGroupTypes)
      .def("get_init_ransac_thresh",
           &GlobalGroupTriangulationOptions::GetInitRansacThresh)
      .def("get_verification_thresh",
           &GlobalGroupTriangulationOptions::GetVerificationThresh);
  MakeDataclass(PyGlobalGroupTriangulationOptions);

  py::classh<GlobalGroupTriangulationController>(
      m, "GlobalGroupTriangulationController")
      .def(py::init<const GlobalGroupTriangulationOptions &,
                    const std::shared_ptr<HolisticReconstruction> &,
                    const colmap::CorrespondenceGraph &>(),
           "options"_a, "reconstruction"_a, "correspondence_graph"_a)
      .def("run", &GlobalGroupTriangulationController::Run);

  m.def(
      "global_group_triangulation",
      [](const GlobalGroupTriangulationOptions &options,
         const std::shared_ptr<HolisticReconstruction> &recon,
         const colmap::CorrespondenceGraph &corr_graph) {
        GlobalGroupTriangulationController controller(options, recon,
                                                      corr_graph);
        controller.Run();
      },
      "options"_a, "reconstruction"_a, "correspondence_graph"_a);
}

} // namespace limap
