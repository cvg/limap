#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_limap/helpers.h"

#include <colmap/util/threading.h>
#include "util/types.h"
#include "util/kd_tree.h"

#include "base/graph.h"
#include "base/camera.h"
#include "base/camera_view.h"
#include "base/image_collection.h"
#include "base/linebase.h"
#include "base/infinite_line.h"
#include "base/linetrack.h"
#include "base/line_dists.h"
#include "base/line_linker.h"
#include "base/pointtrack.h"

namespace limap {

void bind_general_structures(py::module& m) {
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<>())
        .def(py::init<const std::vector<V3D>&>())
        .def(py::init<const M3D&>())
        .def("point_distance", &KDTree::point_distance)
        .def("query_nearest", &KDTree::query_nearest)
        .def("save", &KDTree::save)
        .def("load", &KDTree::load);
}

void bind_graph(py::module& m) {
    py::class_<PatchNode>(m, "PatchNode")
        .def(py::init<int, int>())
        .def_readonly("image_idx", &PatchNode::image_idx)
        .def_readonly("line_idx", &PatchNode::line_idx)
        .def_readonly("node_idx", &PatchNode::node_idx)
        .def_readonly("out_edges", &PatchNode::out_edges)
        .def_readonly("in_edges", &PatchNode::in_edges);

    py::class_<Edge>(m,"Edge")
        .def(py::init<size_t,size_t,double>())
        .def_readonly("node_idx1", &Edge::node_idx1)
        .def_readonly("node_idx2", &Edge::node_idx2)
        .def_readonly("edge_idx", &Edge::edge_idx)
        .def_readwrite("similarity", &Edge::sim);

    py::class_<Graph>(m,"Graph")
        .def(py::init<>())
        .def_property_readonly("input_degrees", &Graph::GetInputDegrees)
        .def_property_readonly("output_degrees", &Graph::GetOutputDegrees)
        .def_property_readonly("scores", &Graph::GetScores)
        .def_readonly("nodes", &Graph::nodes)
        .def_readonly("undirected_edges", &Graph::undirected_edges)
        .def_readonly("node_map", &Graph::node_map)
        .def("find_or_create_node", &Graph::FindOrCreateNode)
        .def("get_node_id", &Graph::GetNodeID)
        .def("add_edge", &Graph::AddEdge)
        .def("register_matches", [](
            Graph& self,
            int im1_idx,
            int im2_idx,
            py::array_t<size_t, py::array::c_style>& matches,
            py::array_t<double, py::array::c_style>& similarities
        ) {
            py::buffer_info matches_info = matches.request();

            THROW_CHECK_EQ(matches_info.ndim,2);

            size_t* matches_ptr = static_cast<size_t*>(matches_info.ptr);
            std::vector<ssize_t> matches_shape = matches_info.shape;

            THROW_CHECK_EQ(matches_shape[1],2);

            size_t n_matches = static_cast<size_t>(matches_shape[0]);
            py::buffer_info sim_info = similarities.request();
            double* sim_ptr = static_cast<double*>(sim_info.ptr);

            assert(n_matches == static_cast<size_t>(sim_info.shape[0]));

            self.RegisterMatches(im1_idx, im2_idx, matches_ptr, sim_ptr, n_matches);
        });

    py::class_<DirectedGraph>(m, "DirectedGraph")
        .def(py::init<>())
        .def_property_readonly("input_degrees", &DirectedGraph::GetInputDegrees)
        .def_property_readonly("output_degrees", &DirectedGraph::GetOutputDegrees)
        .def_property_readonly("scores", &DirectedGraph::GetScores)
        .def_readonly("nodes", &DirectedGraph::nodes)
        .def_readonly("directed_edges", &DirectedGraph::directed_edges)
        .def_readonly("node_map", &DirectedGraph::node_map)
        .def("find_or_create_node", &DirectedGraph::FindOrCreateNode)
        .def("get_node_id", &DirectedGraph::GetNodeID)
        .def("add_edge", &DirectedGraph::AddEdgeDirected)
        .def("register_matches", [](
            DirectedGraph& self,
            int im1_idx,
            int im2_idx,
            py::array_t<size_t, py::array::c_style>& matches,
            py::array_t<double, py::array::c_style>& similarities
        ) {
            py::buffer_info matches_info = matches.request();

            THROW_CHECK_EQ(matches_info.ndim,2);

            size_t* matches_ptr = static_cast<size_t*>(matches_info.ptr);
            std::vector<ssize_t> matches_shape = matches_info.shape;

            THROW_CHECK_EQ(matches_shape[1],2);

            size_t n_matches = static_cast<size_t>(matches_shape[0]);
            py::buffer_info sim_info = similarities.request();
            double* sim_ptr = static_cast<double*>(sim_info.ptr);

            assert(n_matches == static_cast<size_t>(sim_info.shape[0]));

            self.RegisterMatchesDirected(im1_idx, im2_idx, matches_ptr, sim_ptr, n_matches);
        });

    m.def("compute_track_labels", &ComputeTrackLabels);
    m.def("compute_score_labels", &ComputeScoreLabels);
    m.def("compute_root_labels", &ComputeRootLabels);
    m.def("count_track_edges", &CountTrackEdges);
}

void bind_transforms(py::module& m) {
    py::class_<SimilarityTransform3>(m, "SimilarityTransform3")
        .def(py::init<>())
        .def(py::init<V4D, V3D, double>(), py::arg("qvec"), py::arg("tvec"), py::arg("scale") = 1.0)
        .def(py::init<M3D, V3D, double>(), py::arg("R"), py::arg("T"), py::arg("scale") = 1.0)
        .def("R", &SimilarityTransform3::R)
        .def("T", &SimilarityTransform3::T)
        .def("s", &SimilarityTransform3::s);
}

void bind_linebase(py::module& m) {
    py::class_<Line2d>(m, "Line2d")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def(py::init<V2D, V2D>(), py::arg("start"), py::arg("end"))
        .def(py::init<V2D, V2D, double>(), py::arg("start"), py::arg("end"), py::kw_only(), py::arg("score"))
        .def(py::pickle(
            [](const Line2d& input) { // dump
                return input.as_array();
            },
            [](const Eigen::MatrixXd& arr) { // load
                return Line2d(arr);
            }
        ))
        .def_readonly("start", &Line2d::start)
        .def_readonly("end", &Line2d::end)
        .def_readonly("score", &Line2d::score)
        .def("length", &Line2d::length)
        .def("coords", &Line2d::coords)
        .def("as_array", &Line2d::as_array)
        .def("midpoint", &Line2d::midpoint)
        .def("direction", &Line2d::direction)
        .def("point_projection", &Line2d::point_projection)
        .def("point_distance", &Line2d::point_distance);

    py::class_<Line3d>(m, "Line3d")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def(py::init<V3D, V3D>(), py::arg("start"), py::arg("end"))
        .def(py::init<V3D, V3D, double, double, double, double>(), py::arg("start"), py::arg("end"), py::kw_only(), py::arg("score"), py::arg("depth_start"), py::arg("depth_end"), py::arg("uncertainty"))
        .def(py::pickle(
            [](const Line3d& input) { // dump
                return input.as_array();
            },
            [](const Eigen::MatrixXd& arr) { // load
                return Line3d(arr);
            }
        ))
        .def_readonly("start", &Line3d::start)
        .def_readonly("end", &Line3d::end)
        .def_readonly("score", &Line3d::score)
        .def_readonly("depths", &Line3d::depths)
        .def_readonly("uncertainty", &Line3d::uncertainty)
        .def("set_uncertainty", &Line3d::set_uncertainty)
        .def("length", &Line3d::length)
        .def("as_array", &Line3d::as_array)
        .def("projection", &Line3d::projection)
        .def("sensitivity", &Line3d::sensitivity)
        .def("computeUncertainty", &Line3d::computeUncertainty)
        .def("midpoint", &Line3d::midpoint)
        .def("direction", &Line3d::direction)
        .def("point_projection", &Line3d::point_projection)
        .def("point_distance", &Line3d::point_distance);

    m.def("_GetLine2dVectorFromArray", &GetLine2dVectorFromArray);
    m.def("_GetLine3dVectorFromArray", &GetLine3dVectorFromArray);

    py::class_<InfiniteLine2d>(m, "InfiniteLine2d")
        .def(py::init<>())
        .def(py::init<const V3D&>()) // coords
        .def(py::init<const V2D&, const V2D&>()) // point + direction
        .def(py::init<const Line2d&>())
        .def_readonly("coords", &InfiniteLine2d::coords)
        .def("point", &InfiniteLine2d::point)
        .def("direction", &InfiniteLine2d::direction)
        .def("point_projection", &InfiniteLine2d::point_projection)
        .def("point_distance", &InfiniteLine2d::point_distance);

    py::class_<InfiniteLine3d>(m, "InfiniteLine3d")
        .def(py::init<>())
        .def(py::init<const V3D&, const V3D&, bool>())
        .def(py::init<const Line3d&>())
        .def_readonly("d", &InfiniteLine3d::d)
        .def_readonly("m", &InfiniteLine3d::m)
        .def("point", &InfiniteLine3d::point)
        .def("direction", &InfiniteLine3d::direction)
        .def("matrix", &InfiniteLine3d::matrix)
        .def("point_projection", &InfiniteLine3d::point_projection)
        .def("point_distance", &InfiniteLine3d::point_distance)
        .def("projection", &InfiniteLine3d::projection)
        .def("unprojection", &InfiniteLine3d::unprojection)
        .def("project_from_infinite_line", &InfiniteLine3d::project_from_infinite_line)
        .def("project_to_infinite_line", &InfiniteLine3d::project_to_infinite_line);

    m.def("_GetLineSegmentFromInfiniteLine3d", py::overload_cast<const InfiniteLine3d&, const std::vector<CameraView>&, const std::vector<Line2d>&, const int>(&GetLineSegmentFromInfiniteLine3d), py::arg("inf_line"), py::arg("camviews"), py::arg("line2ds"), py::arg("num_outliers") = 2);
    m.def("_GetLineSegmentFromInfiniteLine3d", py::overload_cast<const InfiniteLine3d&, const std::vector<Line3d>&, const int>(&GetLineSegmentFromInfiniteLine3d), py::arg("inf_line"), py::arg("line3ds"), py::arg("num_outliers") = 2);
}

void bind_linetrack(py::module& m) {
    py::class_<LineTrack>(m, "LineTrack")
        .def(py::init<>())
        .def(py::init<LineTrack>())
        .def(py::init<const Line3d&, const std::vector<int>&, const std::vector<int>&, const std::vector<Line2d>&>())
        .def(py::init<py::dict>())
        .def("as_dict", &LineTrack::as_dict)
        .def(py::pickle(
            [](const LineTrack& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return LineTrack(dict);
            }
        ))
        .def_readwrite("line", &LineTrack::line)
        .def_readonly("node_id_list", &LineTrack::node_id_list)
        .def_readonly("image_id_list", &LineTrack::image_id_list)
        .def_readonly("line_id_list", &LineTrack::line_id_list)
        .def_readonly("line3d_list", &LineTrack::line3d_list)
        .def_readonly("line2d_list", &LineTrack::line2d_list)
        .def_readonly("score_list", &LineTrack::score_list)
        .def_readonly("active", &LineTrack::active)
        .def("count_lines", &LineTrack::count_lines)
        .def("GetSortedImageIds", &LineTrack::GetSortedImageIds)
        .def("count_images", &LineTrack::count_images)
        .def("projection", &LineTrack::projection)
        .def("HasImage", &LineTrack::HasImage)
        .def("Read", &LineTrack::Read)
        .def("Write", &LineTrack::Write);
}

void bind_line_dists(py::module& m) {
    py::enum_<LineDistType>(m, "LineDistType")
        .value("ANGULAR", LineDistType::ANGULAR)
        .value("ANGULAR_DIST", LineDistType::ANGULAR_DIST)
        .value("ENDPOINTS", LineDistType::ENDPOINTS)
        .value("MIDPOINT", LineDistType::MIDPOINT)
        .value("MIDPOINT_PERPENDICULAR", LineDistType::MIDPOINT_PERPENDICULAR)
        .value("OVERLAP", LineDistType::OVERLAP)
        .value("BIOVERLAP", LineDistType::BIOVERLAP)
        .value("OVERLAP_DIST", LineDistType::OVERLAP_DIST)
        .value("PERPENDICULAR_ONEWAY", LineDistType::PERPENDICULAR_ONEWAY)
        .value("PERPENDICULAR", LineDistType::PERPENDICULAR)
        .value("PERPENDICULAR_SCALEINV_ONEWAY", LineDistType::PERPENDICULAR_SCALEINV_ONEWAY)
        .value("PERPENDICULAR_SCALEINV", LineDistType::PERPENDICULAR_SCALEINV)
        .value("ENDPOINTS_SCALEINV_ONEWAY", LineDistType::ENDPOINTS_SCALEINV_ONEWAY)
        .value("ENDPOINTS_SCALEINV", LineDistType::ENDPOINTS_SCALEINV)
        .value("INNERSEG", LineDistType::INNERSEG);

    m.def("compute_distance_2d", 
        [](const Line2d& l1, const Line2d& l2, const LineDistType& type) {
            return compute_distance<Line2d>(l1, l2, type);
        }
    );
    m.def("compute_distance_3d", 
        [](const Line3d& l1, const Line3d& l2, const LineDistType& type) {
            return compute_distance<Line3d>(l1, l2, type);
        }
    );
    m.def("compute_pairwise_distance_2d", 
        [](const std::vector<Line2d>& lines, const LineDistType& type) {
            return compute_pairwise_distance<Line2d>(lines, type);
        }
    );
    m.def("compute_pairwise_distance_3d", 
        [](const std::vector<Line3d>& lines, const LineDistType& type) {
            return compute_pairwise_distance<Line3d>(lines, type);
        }
    );
}

void bind_line_linker(py::module& m) {
    py::class_<LineLinker2dConfig>(m, "LineLinker2dConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("score_th", &LineLinker2dConfig::score_th)
        .def_readwrite("th_angle", &LineLinker2dConfig::th_angle)
        .def_readwrite("th_overlap", &LineLinker2dConfig::th_overlap)
        .def_readwrite("th_smartangle", &LineLinker2dConfig::th_smartangle)
        .def_readwrite("th_smartoverlap", &LineLinker2dConfig::th_smartoverlap)
        .def_readwrite("th_perp", &LineLinker2dConfig::th_perp)
        .def_readwrite("th_innerseg", &LineLinker2dConfig::th_innerseg)
        .def_readwrite("use_angle", &LineLinker2dConfig::use_angle)
        .def_readwrite("use_overlap", &LineLinker2dConfig::use_overlap)
        .def_readwrite("use_smartangle", &LineLinker2dConfig::use_smartangle)
        .def_readwrite("use_perp", &LineLinker2dConfig::use_perp)
        .def_readwrite("use_innerseg", &LineLinker2dConfig::use_innerseg);

    py::class_<LineLinker3dConfig>(m, "LineLinker3dConfig")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def_readwrite("score_th", &LineLinker3dConfig::score_th)
        .def_readwrite("th_angle", &LineLinker3dConfig::th_angle)
        .def_readwrite("th_overlap", &LineLinker3dConfig::th_overlap)
        .def_readwrite("th_smartangle", &LineLinker3dConfig::th_smartangle)
        .def_readwrite("th_smartoverlap", &LineLinker3dConfig::th_smartoverlap)
        .def_readwrite("th_perp", &LineLinker3dConfig::th_perp)
        .def_readwrite("th_innerseg", &LineLinker3dConfig::th_innerseg)
        .def_readwrite("th_scaleinv", &LineLinker3dConfig::th_scaleinv)
        .def_readwrite("use_angle", &LineLinker3dConfig::use_angle)
        .def_readwrite("use_overlap", &LineLinker3dConfig::use_overlap)
        .def_readwrite("use_smartangle", &LineLinker3dConfig::use_smartangle)
        .def_readwrite("use_perp", &LineLinker3dConfig::use_perp)
        .def_readwrite("use_innerseg", &LineLinker3dConfig::use_innerseg)
        .def_readwrite("use_scaleinv", &LineLinker3dConfig::use_scaleinv);

    py::class_<LineLinker2d>(m, "LineLinker2d")
        .def(py::init<>())
        .def(py::init<const LineLinker2dConfig&>())
        .def(py::init<py::dict>())
        .def_readwrite("config", &LineLinker2d::config)
        .def("check_connection", &LineLinker2d::check_connection)
        .def("compute_score", &LineLinker2d::compute_score);

    py::class_<LineLinker3d>(m, "LineLinker3d")
        .def(py::init<>())
        .def(py::init<const LineLinker3dConfig&>())
        .def(py::init<py::dict>())
        .def_readwrite("config", &LineLinker3d::config)
        .def("check_connection", &LineLinker3d::check_connection)
        .def("compute_score", &LineLinker3d::compute_score);

    py::class_<LineLinker>(m, "LineLinker")
        .def(py::init<>())
        .def(py::init<const LineLinker2d&, const LineLinker3d&>())
        .def(py::init<const LineLinker2dConfig&, const LineLinker3dConfig&>())
        .def(py::init<py::dict, py::dict>())
        .def_readwrite("linker_2d", &LineLinker::linker_2d)
        .def_readwrite("linker_3d", &LineLinker::linker_3d)
        .def("GetLinker2d", &LineLinker::GetLinker2d)
        .def("GetLinker3d", &LineLinker::GetLinker3d)
        .def("check_connection_2d", &LineLinker::check_connection_2d)
        .def("compute_score_2d", &LineLinker::compute_score_2d)
        .def("check_connection_3d", &LineLinker::check_connection_3d)
        .def("compute_score_3d", &LineLinker::compute_score_3d);
}

void bind_camera(py::module& m) {
    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def(py::init<int, const std::vector<double>&, int, std::pair<int, int>>(), py::arg("model_id"), py::arg("params"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1))
        .def(py::init<const std::string&, const std::vector<double>&, int, std::pair<int, int>>(), py::arg("model_name"), py::arg("params"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1))
        .def(py::init<M3D, int, std::pair<int, int>>(), py::arg("K"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1))
        .def(py::init<int, M3D, int, std::pair<int, int>>(), py::arg("model_id"), py::arg("K"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1))
        .def(py::init<const std::string&, M3D, int, std::pair<int, int>>(), py::arg("model_name"), py::arg("K"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1))
        .def(py::init<py::dict>())
        .def(py::init<const Camera&>())
        .def(py::init<int, int, std::pair<int, int>>(), py::arg("model_id"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1)) // empty camera
        .def(py::init<const std::string&, int, std::pair<int, int>>(), py::arg("model_name"), py::arg("cam_id")=-1, py::arg("hw")=std::make_pair<int, int>(-1, -1)) // empty camera
        .def(py::pickle(
            [](const Camera& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return Camera(dict);
            }
        ))
        .def("as_dict", &Camera::as_dict)
        .def("h", &Camera::h)
        .def("w", &Camera::w)
        .def("K", &Camera::K)
        .def("K_inv", &Camera::K_inv)
        .def("cam_id", &Camera::CameraId)
        .def("model_id", &Camera::ModelId)
        .def("params", &Camera::params)
        .def("num_params", &Camera::NumParams)
        .def("resize", &Camera::resize)
        .def("set_max_image_dim", &Camera::set_max_image_dim)
        .def("set_cam_id", &Camera::SetCameraId)
        .def("InitializeParams", &Camera::InitializeParams)
        .def("IsUndistorted", &Camera::IsUndistorted)
        .def("ImageToWorld", &Camera::ImageToWorld)
        .def("WorldToImage", &Camera::WorldToImage)
        .def("IsInitialized", &Camera::IsInitialized);

    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<bool>(), py::arg("initialized")=false)
        .def(py::init<V4D, V3D, bool>(), py::arg("qvec"), py::arg("tvec"), py::arg("initialized")=true)
        .def(py::init<M3D, V3D, bool>(), py::arg("R"), py::arg("tvec"), py::arg("initialized")=true)
        .def(py::init<py::dict>())
        .def(py::init<const CameraPose&>())
        .def(py::pickle(
            [](const CameraPose& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return CameraPose(dict);
            }
        ))
        .def("as_dict", &CameraPose::as_dict)
        .def_readonly("qvec", &CameraPose::qvec)
        .def_readonly("tvec", &CameraPose::tvec)
        .def_readwrite("initialized", &CameraPose::initialized)
        .def("R", &CameraPose::R)
        .def("T", &CameraPose::T)
        .def("center", &CameraPose::center)
        .def("projdepth", &CameraPose::projdepth);

    py::class_<CameraImage>(m, "CameraImage")
        .def(py::init<>())
        .def(py::init<const int&, const std::string&>(), py::arg("cam_id"), py::arg("image_name") = "none") // empty image
        .def(py::init<const Camera&, const std::string&>(), py::arg("camera"), py::arg("image_name") = "none") // empty image
        .def(py::init<const int&, const CameraPose&, const std::string&>(), py::arg("cam_id"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<const Camera&, const CameraPose&, const std::string&>(), py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<py::dict>())
        .def(py::init<const CameraImage&>())
        .def(py::pickle(
            [](const CameraImage& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return CameraImage(dict);
            }
        ))
        .def("as_dict", &CameraImage::as_dict)
        .def_readonly("cam_id", &CameraImage::cam_id)
        .def_readonly("pose", &CameraImage::pose)
        .def("R", &CameraImage::R)
        .def("T", &CameraImage::T)
        .def("set_camera_id", &CameraImage::SetCameraId)
        .def("image_name", &CameraImage::image_name)
        .def("set_image_name", &CameraImage::SetImageName);

    py::class_<CameraView>(m, "CameraView")
        .def(py::init<>())
        .def(py::init<const Camera&, const std::string&>(), py::arg("camera"), py::arg("image_name") = "none") // empty view
        .def(py::init<const Camera&, const CameraPose&, const std::string&>(), py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<py::dict>())
        .def(py::init<const CameraView&>())
        .def(py::pickle(
            [](const CameraView& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return CameraView(dict);
            }
        ))
        .def_readonly("cam", &CameraView::cam)
        .def_readonly("pose", &CameraView::pose)
        .def("as_dict", &CameraView::as_dict)
        .def("read_image", &CameraView::read_image, py::arg("set_gray")=false)
        .def("K", &CameraView::K)
        .def("K_inv", &CameraView::K_inv)
        .def("h", &CameraView::h)
        .def("w", &CameraView::w)
        .def("R", &CameraView::R)
        .def("T", &CameraView::T)
        .def("matrix", &CameraView::matrix)
        .def("projection", &CameraView::projection)
        .def("ray_direction", &CameraView::ray_direction)
        .def("get_direction_from_vp", &CameraView::get_direction_from_vp)
        .def("image_name", &CameraView::image_name)
        .def("set_image_name", &CameraView::SetImageName)
        .def("get_initial_focal_length", &CameraView::get_initial_focal_length);

    py::class_<ImageCollection>(m, "ImageCollection")
        .def(py::init<>())
        .def(py::init<const std::map<int, Camera>&, const std::map<int, CameraImage>&>())
        .def(py::init<const std::vector<Camera>&, const std::map<int, CameraImage>&>())
        .def(py::init<const std::map<int, Camera>&, const std::vector<CameraImage>&>())
        .def(py::init<const std::vector<Camera>&, const std::vector<CameraImage>&>())
        .def(py::init<const std::vector<CameraView>&>())
        .def(py::init<py::dict>())
        .def(py::init<const ImageCollection&>())
        .def(py::pickle(
            [](const ImageCollection& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return ImageCollection(dict);
            }
        ))
        .def("as_dict", &ImageCollection::as_dict)
        .def("subset_by_camera_ids", &ImageCollection::subset_by_camera_ids)
        .def("subset_by_image_ids", &ImageCollection::subset_by_image_ids)
        .def("subset_initialized", &ImageCollection::subset_initialized)
        .def("update_neighbors", &ImageCollection::update_neighbors)
        .def("get_cameras", &ImageCollection::get_cameras)
        .def("get_cam_ids", &ImageCollection::get_cam_ids)
        .def("get_images", &ImageCollection::get_images)
        .def("get_img_ids", &ImageCollection::get_img_ids)
        .def("get_camviews", &ImageCollection::get_camviews)
        .def("get_map_camviews", &ImageCollection::get_map_camviews)
        .def("get_locations", &ImageCollection::get_locations)
        .def("get_map_locations", &ImageCollection::get_map_locations)
        .def("exist_cam", &ImageCollection::exist_cam)
        .def("exist_image", &ImageCollection::exist_image)
        .def("cam", &ImageCollection::cam)
        .def("camimage", &ImageCollection::camimage)
        .def("campose", &ImageCollection::campose)
        .def("camview", &ImageCollection::camview)
        .def("image_name", &ImageCollection::image_name)
        .def("get_image_name_list", &ImageCollection::get_image_name_list)
        .def("get_image_name_dict", &ImageCollection::get_image_name_dict)
        .def("NumCameras", &ImageCollection::NumCameras)
        .def("NumImages", &ImageCollection::NumImages)
        .def("set_max_image_dim", &ImageCollection::set_max_image_dim)
        .def("change_camera", &ImageCollection::change_camera)
        .def("set_camera_pose", &ImageCollection::set_camera_pose)
        .def("get_camera_pose", &ImageCollection::get_camera_pose)
        .def("change_image", &ImageCollection::change_image)
        .def("change_image_name", &ImageCollection::change_image_name)
        .def("IsUndistorted", &ImageCollection::IsUndistorted)
        .def("read_image", &ImageCollection::read_image, py::arg("img_id"), py::arg("set_gray")=false)
        .def("apply_similarity_transform", &ImageCollection::apply_similarity_transform)
        .def("get_first_image_id_by_camera_id", &ImageCollection::get_first_image_id_by_camera_id)
        .def("init_uninitialized_cameras", &ImageCollection::init_uninitialized_cameras)
        .def("uninitialize_poses", &ImageCollection::uninitialize_poses)
        .def("uninitialize_intrinsics", &ImageCollection::uninitialize_intrinsics)
        .def("IsUndistortedCameraModel", &ImageCollection::IsUndistortedCameraModel);
}

void bind_pointtrack(py::module& m) {
    py::class_<Point2d>(m, "Point2d")
        .def(py::init<>())
        .def(py::init<V2D, int>(), py::arg("p"), py::arg("point3D_id")=-1)
        .def("as_dict", &Point2d::as_dict)
        .def(py::pickle(
            [](const Point2d& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return Point2d(dict);
            }
        ))
        .def_readwrite("p", &Point2d::p)
        .def_readwrite("point3D_id", &Point2d::point3D_id);

    py::class_<PointTrack>(m, "PointTrack")
        .def(py::init<>())
        .def(py::init<const PointTrack&>())
        .def(py::init<const V3D&, const std::vector<int>&, const std::vector<int>&, const std::vector<V2D>&>())
        .def(py::init<py::dict>())
        .def("as_dict", &PointTrack::as_dict)
        .def(py::pickle(
            [](const PointTrack& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return PointTrack(dict);
            }
        ))
        .def_readwrite("p", &PointTrack::p)
        .def_readonly("image_id_list", &PointTrack::image_id_list)
        .def_readonly("p2d_id_list", &PointTrack::p2d_id_list)
        .def_readonly("p2d_list", &PointTrack::p2d_list)
        .def("count_images", &PointTrack::count_images);
}

void bind_base(py::module& m) {
    bind_general_structures(m);
    bind_graph(m);
    bind_transforms(m);
    bind_pointtrack(m);
    bind_linebase(m);
    bind_linetrack(m);
    bind_line_dists(m);
    bind_line_linker(m);
    bind_camera(m);

    m.def("get_effective_num_threads", &colmap::GetEffectiveNumThreads);
}

}

