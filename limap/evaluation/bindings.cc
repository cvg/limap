#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include "base/linebase.h"
#include <Eigen/Core>
#include <vector>

namespace py = pybind11;

#include "evaluation/mesh_evaluator.h"
#include "evaluation/point_cloud_evaluator.h"
#include "evaluation/refline_evaluator.h"

namespace limap {

void bind_evaluator(py::module &m) {
  using namespace evaluation;

  py::class_<PointCloudEvaluator>(m, "PointCloudEvaluator",
                                  "The evaluator for line maps with respect to "
                                  "a GT point cloud (using a K-D Tree).")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const std::vector<V3D> &>(), R"(
            Constructor from list[:class:`np.array`] of shape (3,)
        )")
      .def(py::init<const Eigen::MatrixXd &>(), R"(
            Constructor from :class:`np.array` of shape (N, 3)
        )")
      .def("Build", &PointCloudEvaluator::Build,
           R"(Build the indexes of the K-D Tree)")
      .def("Save", &PointCloudEvaluator::Save, R"(
            Save the built K-D Tree into a file

            Args:
                filename (str): The file to write to
        )")
      .def("Load", &PointCloudEvaluator::Load, R"(
            Read the pre-built K-D Tree from a file

            Args:
                filename (str): The file to read from
        )")
      .def("ComputeDistPoint", &PointCloudEvaluator::ComputeDistPoint, R"(
            Compute the distance from a query point to the point cloud

            Args:
                :class:`np.array` of shape (3,): The query point

            Returns:
                float: The distance from the point to the GT point cloud
        )")
      .def(
          "ComputeDistLine",
          [](PointCloudEvaluator &self, const Line3d &line, int n_samples) {
            return self.ComputeDistLine(line, n_samples);
          },
          R"(
                    Compute the distance for a set of uniformly sampled points along the line

                    Args:
                        line (Line3d): :class:`~limap.base.Line3d`: instance
                        n_samples (int, optional): number of samples (default = 1000)

                    Returns:
                        :class:`np.array` of shape (n_samples,): the computed distances
                        
                )",
          py::arg("line"), py::arg("n_samples") = 1000)
      .def(
          "ComputeInlierRatio",
          [](PointCloudEvaluator &self, const Line3d &line, double threshold,
             int n_samples) {
            return self.ComputeInlierRatio(line, threshold, n_samples);
          },
          R"(
                Compute the percentage of the line lying with a certain threshold to the point cloud

                Args:
                    line (Line3d): :class:`~limap.base.Line3d`: instance
                    threshold (float): threshold
                    n_samples (int, optional): number of samples (default = 1000)

                Returns:
                    float: The computed percentage
            )",
          py::arg("line"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def(
          "ComputeInlierSegs",
          [](PointCloudEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeInlierSegs(lines, threshold, n_samples);
          },
          R"(
                Compute the inlier parts of the lines that are within a certain threshold to the point cloud, for visualization.

                Args:
                    lines (list[:class:`limap.base.Line3d`]): Input 3D line segments
                    threshold (float): threshold
                    n_samples (int): number of samples (default = 1000)

                Returns:
                    list[:class:`limap.base.Line3d`]: Inlier parts of all the lines, useful for visualization

            )",
          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def(
          "ComputeOutlierSegs",
          [](PointCloudEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeOutlierSegs(lines, threshold, n_samples);
          },
          R"(
                Compute the outlier parts of the lines that are at least a certain threshold far away from the point cloud, for visualization.

                Args:
                    lines (list[:class:`limap.base.Line3d`]): Input 3D line segments
                    threshold (float): threshold
                    n_samples (int): number of samples (default = 1000)

                Returns:
                    list[:class:`limap.base.Line3d`]: Outlier parts of all the lines, useful for visualization

            )",
          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def("ComputeDistsforEachPoint",
           &PointCloudEvaluator::ComputeDistsforEachPoint)
      .def("ComputeDistsforEachPoint_KDTree",
           &PointCloudEvaluator::ComputeDistsforEachPoint_KDTree);

  py::class_<MeshEvaluator>(m, "MeshEvaluator")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const std::string &, const double &>(), R"(
           Constructor from a mesh file (str) and a scale (float)
        )")
      .def("ComputeDistPoint", &MeshEvaluator::ComputeDistPoint, R"(
            Compute the distance from a query point to the mesh

            Args:
                :class:`np.array` of shape (3,): The query point

            Returns:
                float: The distance from the point to the GT mesh
        )")
      .def(
          "ComputeDistLine",
          [](MeshEvaluator &self, const Line3d &line, int n_samples) {
            return self.ComputeDistLine(line, n_samples);
          },
          R"(
                    Compute the distance for a set of uniformly sampled points along the line

                    Args:
                        line (Line3d): :class:`~limap.base.Line3d`: instance
                        n_samples (int, optional): number of samples (default = 1000)

                    Returns:
                        :class:`np.array` of shape (n_samples,): the computed distances
                        
                )",
          py::arg("line"), py::arg("n_samples") = 1000)
      .def(
          "ComputeInlierRatio",
          [](MeshEvaluator &self, const Line3d &line, double threshold,
             int n_samples) {
            return self.ComputeInlierRatio(line, threshold, n_samples);
          },
          R"(
                Compute the percentage of the line lying with a certain threshold to the mesh

                Args:
                    line (Line3d): :class:`~limap.base.Line3d`: instance
                    threshold (float): threshold
                    n_samples (int, optional): number of samples (default = 1000)

                Returns:
                    float: The computed percentage
            )",
          py::arg("line"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def(
          "ComputeInlierSegs",
          [](MeshEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeInlierSegs(lines, threshold, n_samples);
          },
          R"(
                Compute the inlier parts of the lines that are within a certain threshold to the mesh, for visualization.

                Args:
                    lines (list[:class:`limap.base.Line3d`]): Input 3D line segments
                    threshold (float): threshold
                    n_samples (int): number of samples (default = 1000)

                Returns:
                    list[:class:`limap.base.Line3d`]: Inlier parts of all the lines, useful for visualization

            )",

          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def(
          "ComputeOutlierSegs",
          [](MeshEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeOutlierSegs(lines, threshold, n_samples);
          },
          R"(
                Compute the outlier parts of the lines that are at least a certain threshold far away from the mesh, for visualization.

                Args:
                    lines (list[:class:`limap.base.Line3d`]): Input 3D line segments
                    threshold (float): threshold
                    n_samples (int): number of samples (default = 1000)

                Returns:
                    list[:class:`limap.base.Line3d`]: Outlier parts of all the lines, useful for visualization

            )",
          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000);

  py::class_<RefLineEvaluator>(m, "RefLineEvaluator")
      .def(py::init<>())
      .def(py::init<const std::vector<Line3d> &>())
      .def("SumLength", &RefLineEvaluator::SumLength)
      .def(
          "ComputeRecallRef",
          [](RefLineEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeRecallRef(lines, threshold, n_samples);
          },
          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000)
      .def(
          "ComputeRecallTested",
          [](RefLineEvaluator &self, const std::vector<Line3d> &lines,
             double threshold, int n_samples) {
            return self.ComputeRecallTested(lines, threshold, n_samples);
          },
          py::arg("lines"), py::arg("threshold"), py::arg("n_samples") = 1000);
}

void bind_evaluation(py::module &m) { bind_evaluator(m); }

} // namespace limap
