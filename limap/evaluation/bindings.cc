#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"
#include "base/linebase.h"

namespace py = pybind11;

#include "evaluation/point_cloud_evaluator.h"
#include "evaluation/mesh_evaluator.h"
#include "evaluation/refline_evaluator.h"

namespace limap {

void bind_evaluator(py::module& m) {
    using namespace evaluation;

    py::class_<PointCloudEvaluator>(m, "PointCloudEvaluator")
        .def(py::init<>())
        .def(py::init<const std::vector<V3D>&>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def("Build", &PointCloudEvaluator::Build)
        .def("Save", &PointCloudEvaluator::Save)
        .def("Load", &PointCloudEvaluator::Load)
        .def("ComputeDistPoint", &PointCloudEvaluator::ComputeDistPoint)
        .def("ComputeDistLine",
                [] (PointCloudEvaluator& self,
                    const Line3d& line, 
                    int n_samples) {
                    return self.ComputeDistLine(line, n_samples);
                },
                py::arg("line"),
                py::arg("n_samples") = 1000
        )
        .def("ComputeInlierRatio", 
            [] (PointCloudEvaluator& self,
                const Line3d& line,
                double threshold,
                int n_samples) {
                return self.ComputeInlierRatio(line, threshold, n_samples);
            },
            py::arg("line"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeInlierSegs", 
            [] (PointCloudEvaluator& self,
                const std::vector<Line3d>& lines, 
                double threshold,
                int n_samples) {
                return self.ComputeInlierSegs(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeOutlierSegs", 
            [] (PointCloudEvaluator& self,
                const std::vector<Line3d>& lines, 
                double threshold,
                int n_samples) {
                return self.ComputeOutlierSegs(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeDistsforEachPoint", &PointCloudEvaluator::ComputeDistsforEachPoint)
        .def("ComputeDistsforEachPoint_KDTree", &PointCloudEvaluator::ComputeDistsforEachPoint_KDTree);

    py::class_<MeshEvaluator>(m, "MeshEvaluator")
        .def(py::init<>())
        .def(py::init<const std::string&, const double&>())
        .def("ComputeDistPoint", &MeshEvaluator::ComputeDistPoint)
        .def("ComputeDistLine",
                [] (MeshEvaluator& self,
                    const Line3d& line, 
                    int n_samples) {
                    return self.ComputeDistLine(line, n_samples);
                },
                py::arg("line"),
                py::arg("n_samples") = 1000
        )
        .def("ComputeInlierRatio", 
            [] (MeshEvaluator& self,
                const Line3d& line,
                double threshold,
                int n_samples) {
                return self.ComputeInlierRatio(line, threshold, n_samples);
            },
            py::arg("line"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeInlierSegs", 
            [] (MeshEvaluator& self,
                const std::vector<Line3d>& lines, 
                double threshold,
                int n_samples) {
                return self.ComputeInlierSegs(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeOutlierSegs", 
            [] (MeshEvaluator& self,
                const std::vector<Line3d>& lines, 
                double threshold,
                int n_samples) {
                return self.ComputeOutlierSegs(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        );

    py::class_<RefLineEvaluator>(m, "RefLineEvaluator")
        .def(py::init<>())
        .def(py::init<const std::vector<Line3d>&>())
        .def("SumLength", &RefLineEvaluator::SumLength)
        .def("ComputeRecallRef",
            [] (RefLineEvaluator& self,
                const std::vector<Line3d>& lines,
                double threshold,
                int n_samples) {
                return self.ComputeRecallRef(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        )
        .def("ComputeRecallTested",
            [] (RefLineEvaluator& self,
                const std::vector<Line3d>& lines,
                double threshold,
                int n_samples) {
                return self.ComputeRecallTested(lines, threshold, n_samples);
            },
            py::arg("lines"),
            py::arg("threshold"),
            py::arg("n_samples") = 1000
        );
}

void bind_evaluation(py::module& m) {
    bind_evaluator(m);
}

} // namespace limap

