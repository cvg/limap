#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_limap/helpers.h"

#include <colmap/util/threading.h>
#include "util/types.h"

#include "base/graph.h"
#include "base/camera.h"
#include "base/camera_view.h"
#include "base/image_collection.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/line_dists.h"
#include "base/line_linker.h"
#include "base/line_reconstruction.h"
#include "base/featurepatch.h"

namespace limap {

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

void bind_linebase(py::module& m) {
    py::class_<Line2d>(m, "Line2d")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def(py::init<V2D, V2D>(), py::arg("start"), py::arg("end"))
        .def(py::init<V2D, V2D, double>(), py::arg("start"), py::arg("end"), py::kw_only(), py::arg("score"))
        .def_readonly("start", &Line2d::start)
        .def_readonly("end", &Line2d::end)
        .def_readonly("score", &Line2d::score)
        .def("length", &Line2d::length)
        .def("as_array", &Line2d::as_array)
        .def("midpoint", &Line2d::midpoint)
        .def("direction", &Line2d::direction);

    py::class_<Line3d>(m, "Line3d")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>())
        .def(py::init<V3D, V3D>(), py::arg("start"), py::arg("end"))
        .def(py::init<V3D, V3D, double, double, double, double>(), py::arg("start"), py::arg("end"), py::kw_only(), py::arg("score"), py::arg("depth_start"), py::arg("depth_end"), py::arg("uncertainty"))
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
        .def("direction", &Line3d::direction);

    // build initial graph
    m.def("_GetAllLines2D", 
        [](const std::vector<Eigen::MatrixXd>& all_lines_2d) {
            std::vector<std::vector<Line2d>> all_lines;
            GetAllLines2D(all_lines_2d, all_lines);
            return all_lines;
        }
    );
}

void bind_linetrack(py::module& m) {
    py::class_<LineTrack>(m, "LineTrack")
        .def(py::init<>())
        .def(py::init<LineTrack>())
        .def(py::init<const Line3d&, const std::vector<int>&, const std::vector<int>&, const std::vector<Line2d>&>())
        .def_readwrite("line", &LineTrack::line)
        .def_readonly("node_id_list", &LineTrack::node_id_list)
        .def_readonly("image_id_list", &LineTrack::image_id_list)
        .def_readonly("line_id_list", &LineTrack::line_id_list)
        .def_readonly("line3d_list", &LineTrack::line3d_list)
        .def_readonly("line2d_list", &LineTrack::line2d_list)
        .def_readonly("score_list", &LineTrack::score_list)
        .def_readonly("newmerge", &LineTrack::newmerge)
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

void bind_line_reconstruction(py::module& m) {
    py::class_<LineReconstruction>(m, "LineReconstruction")
        .def(py::init<>())
        .def(py::init<const std::vector<LineTrack>&, const ImageCollection&>())
        .def("NumTracks", &LineReconstruction::NumTracks)
        .def("NumCameras", &LineReconstruction::NumCameras)
        .def("GetInitTrack", &LineReconstruction::GetInitTrack)
        .def("GetCameraMap", &LineReconstruction::GetCameraMap)
        .def("NumSupportingLines", &LineReconstruction::NumSupportingLines)
        .def("NumSupportingImages", &LineReconstruction::NumSupportingImages)
        .def("GetImageIds", &LineReconstruction::GetImageIds)
        .def("GetLine2ds", &LineReconstruction::GetLine2ds)
        .def("GetCameras", &LineReconstruction::GetCameras)
        .def("GetLines", &LineReconstruction::GetLines, py::arg("num_outliers") = 2)
        .def("GetTracks", &LineReconstruction::GetTracks, py::arg("num_outliers") = 2);
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
        .def("IsUndistorted", &Camera::IsUndistorted);

    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def(py::init<V4D, V3D>())
        .def(py::init<M3D, V3D>())
        .def(py::init<py::dict>())
        .def(py::init<const CameraPose&>())
        .def("as_dict", &CameraPose::as_dict)
        .def_readonly("qvec", &CameraPose::qvec)
        .def_readonly("tvec", &CameraPose::tvec)
        .def("R", &CameraPose::R)
        .def("T", &CameraPose::T)
        .def("center", &CameraPose::center)
        .def("projdepth", &CameraPose::projdepth);

    py::class_<CameraImage>(m, "CameraImage")
        .def(py::init<>())
        .def(py::init<const int&, const CameraPose&, const std::string&>(), py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<const Camera&, const CameraPose&, const std::string&>(), py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<py::dict>())
        .def(py::init<const CameraImage&>())
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
        .def(py::init<const Camera&, const CameraPose&, const std::string&>(), py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
        .def(py::init<py::dict>())
        .def(py::init<const CameraView&>())
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
        .def("projection", &CameraView::projection)
        .def("ray_direction", &CameraView::ray_direction)
        .def("image_name", &CameraView::image_name)
        .def("set_image_name", &CameraView::SetImageName);

    py::class_<ImageCollection>(m, "ImageCollection")
        .def(py::init<>())
        .def(py::init<const std::vector<Camera>&, const std::vector<CameraImage>&>())
        .def(py::init<const std::map<int, Camera>&, const std::vector<CameraImage>&>())
        .def(py::init<const std::vector<CameraView>&>())
        .def(py::init<py::dict>())
        .def(py::init<const ImageCollection&>())
        .def("as_dict", &ImageCollection::as_dict)
        .def("get_cameras", &ImageCollection::get_cameras)
        .def("get_cam_ids", &ImageCollection::get_cam_ids)
        .def("get_images", &ImageCollection::get_images)
        .def("get_camviews", &ImageCollection::get_camviews)
        .def("cam", &ImageCollection::cam)
        .def("exist_cam", &ImageCollection::exist_cam)
        .def("camimage", &ImageCollection::camimage)
        .def("campose", &ImageCollection::campose)
        .def("camview", &ImageCollection::camview)
        .def("image_name", &ImageCollection::image_name)
        .def("get_image_list", &ImageCollection::get_image_list)
        .def("NumCameras", &ImageCollection::NumCameras)
        .def("NumImages", &ImageCollection::NumImages)
        .def("set_max_image_dim", &ImageCollection::set_max_image_dim)
        .def("change_camera", &ImageCollection::change_camera)
        .def("change_image", &ImageCollection::change_image)
        .def("change_image_name", &ImageCollection::change_image_name)
        .def("IsUndistorted", &ImageCollection::IsUndistorted)
        .def("read_image", &ImageCollection::read_image, py::arg("img_id"), py::arg("set_gray")=false);
}

template <typename DTYPE>
void bind_patchinfo(py::module& m, std::string type_suffix) {
    using PInfo = PatchInfo<DTYPE>;
    py::class_<PInfo>(m, ("PatchInfo" + type_suffix).c_str())
        .def(py::init<>())
        .def(py::init<py::array_t<DTYPE, py::array::c_style>, M2D, V2D, std::pair<int, int>>())
        .def_readwrite("array", &PInfo::array)
        .def_readwrite("R", &PInfo::R)
        .def_readwrite("tvec", &PInfo::tvec)
        .def_readwrite("img_hw", &PInfo::img_hw);
}

void bind_base(py::module& m) {
    bind_graph(m);
    bind_linebase(m);
    bind_linetrack(m);
    bind_line_dists(m);
    bind_line_linker(m);
    bind_line_reconstruction(m);
    bind_camera(m);

    bind_patchinfo<float16>(m, "_f16");
    bind_patchinfo<float>(m, "_f32");
    bind_patchinfo<double>(m, "_f64");

    m.def("get_effective_num_threads", &colmap::GetEffectiveNumThreads);
}

}

