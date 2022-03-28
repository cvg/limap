#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <Eigen/Core>
#include "_limap/helpers.h"
#include "base/linebase.h"
#include "base/linetrack.h"

namespace py = pybind11;

#include "triangulation/functions.h"
#include "triangulation/triangulator.h"

namespace limap {

void bind_functions(py::module &m) {
    using namespace triangulation;

    m.def("get_normal_direction", &getNormalDirection);
    m.def("get_direction_from_VP", &getDirectionFromVP);
    m.def("compute_epipolar_IoU", &compute_epipolar_IoU);
    m.def("triangulate_endpoints", &triangulate_endpoints);
    m.def("triangulate", &triangulate);
    m.def("triangulate_with_direction", &triangulate_with_direction);
}

void bind_triangulator(py::module &m) {
    using namespace triangulation;

    py::class_<TriangulatorConfig>(m, "TriangulatorConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("add_halfpix", &TriangulatorConfig::add_halfpix)
        .def_readwrite("use_vp", &TriangulatorConfig::use_vp)
        .def_readwrite("var2d", &TriangulatorConfig::var2d)
        .def_readwrite("plane_angle_threshold", &TriangulatorConfig::plane_angle_threshold)
        .def_readwrite("IoU_threshold", &TriangulatorConfig::IoU_threshold)
        .def_readwrite("fullscore_th", &TriangulatorConfig::fullscore_th)
        .def_readwrite("max_valid_conns", &TriangulatorConfig::max_valid_conns)
        .def_readwrite("min_num_outer_edges", &TriangulatorConfig::min_num_outer_edges)
        .def_readwrite("merging_strategy", &TriangulatorConfig::merging_strategy)
        .def_readwrite("num_outliers_aggregator", &TriangulatorConfig::num_outliers_aggregator)
        .def_readwrite("debug_mode", &TriangulatorConfig::debug_mode)
        .def_readwrite("linker2d_config", &TriangulatorConfig::linker2d_config)
        .def_readwrite("linker3d_config", &TriangulatorConfig::linker3d_config);

    py::class_<Triangulator>(m, "Triangulator")
        .def(py::init<>())
        .def(py::init<const TriangulatorConfig&>())
        .def(py::init<py::dict>())
        .def("Init", &Triangulator::Init)
        .def("InitMatches", 
                [] (Triangulator& self,
                    const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                    const std::vector<std::vector<int>>& all_neighbors,
                    bool use_triangulate, bool use_scoring) {
                    return self.InitMatches(all_matches, all_neighbors, use_triangulate, use_scoring);
                },
                py::arg("all_matches"),
                py::arg("all_neigbhors"),
                py::arg("triangulate") = true,
                py::arg("scoring") = false
        )
        .def("InitMatchImage", 
                [] (Triangulator& self, 
                    const int img_id, 
                    const std::vector<Eigen::MatrixXi>& matches, 
                    const std::vector<int>& neighbors,
                    bool use_triangulate,
                    bool use_scoring) {
                    return self.InitMatchImage(img_id, matches, neighbors, use_triangulate, use_scoring);
                },
                py::arg("img_id"),
                py::arg("matches"),
                py::arg("neighbors"),
                py::arg("triangulate") = true,
                py::arg("scoring") = false
        )
        .def("InitExhaustiveMatchImage", 
                [] (Triangulator& self,
                    const int img_id,
                    const std::vector<int>& neighbors,
                    bool use_scoring) {
                    return self.InitExhaustiveMatchImage(img_id, neighbors, use_scoring);
                },
                py::arg("img_id"),
                py::arg("neighbors"),
                py::arg("scoring") = true
        )
        .def("InitAll", 
                [] (Triangulator& self,
                    const std::vector<std::vector<Line2d>>& all_2d_segs,
                    const std::vector<CameraView>& views,
                    const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                    const std::vector<std::vector<int>>& all_neighbors,
                    bool use_triangulate,
                    bool use_scoring) {
                    return self.InitAll(all_2d_segs, views, all_matches, all_neighbors, use_triangulate, use_scoring);
                },
                py::arg("all_2d_segs"),
                py::arg("views"),
                py::arg("all_matches"),
                py::arg("all_neighbors"),
                py::arg("triangulate") = true,
                py::arg("scoring") = false
        )
        .def("SetRanges", &Triangulator::SetRanges)
        .def("UnsetRanges", &Triangulator::UnsetRanges)
        .def("GetLinker", &Triangulator::GetLinker)
        .def("RunTriangulate", &Triangulator::RunTriangulate)
        .def("RunScoring", &Triangulator::RunScoring)
        .def("RunClustering", &Triangulator::RunClustering)
        .def("ComputeLineTracks", &Triangulator::ComputeLineTracks)
        .def("Run", &Triangulator::Run)
        .def("GetTracks", &Triangulator::GetTracks)
        .def("GetVPResult", &Triangulator::GetVPResult)
        .def("GetVPResults", &Triangulator::GetVPResults)
        .def("CountImages", &Triangulator::CountImages)
        .def("CountLines", &Triangulator::CountLines)
        .def("CountAllTris", &Triangulator::CountAllTris)
        .def("GetScoredTrisNode", &Triangulator::GetScoredTrisNode)
        .def("GetValidScoredTrisNode", &Triangulator::GetValidScoredTrisNode)
        .def("GetValidScoredTrisNodeSet", &Triangulator::GetValidScoredTrisNodeSet)
        .def("CountAllValidTris", &Triangulator::CountAllValidTris)
        .def("GetAllValidTris", &Triangulator::GetAllValidTris)
        .def("GetValidTrisImage", &Triangulator::GetValidTrisImage)
        .def("GetValidTrisNode", &Triangulator::GetValidTrisNode)
        .def("GetValidTrisNodeSet", &Triangulator::GetValidTrisNodeSet)
        .def("GetAllBestTris", &Triangulator::GetAllBestTris)
        .def("GetAllValidBestTris", &Triangulator::GetAllValidBestTris)
        .def("GetBestTrisImage", &Triangulator::GetBestTrisImage)
        .def("GetBestTriNode", &Triangulator::GetBestTriNode)
        .def("GetBestScoredTriNode", &Triangulator::GetBestScoredTriNode)
        .def("GetSurvivedLinesImage", &Triangulator::GetSurvivedLinesImage);
}

void bind_triangulation(py::module& m) {
    bind_functions(m);
    bind_triangulator(m);
}

} // namespace limap

