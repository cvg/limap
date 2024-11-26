#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "_limap/helpers.h"

#include "util/kd_tree.h"
#include "util/types.h"
#include <colmap/util/threading.h>

#include "base/camera.h"
#include "base/camera_view.h"
#include "base/graph.h"
#include "base/image_collection.h"
#include "base/infinite_line.h"
#include "base/line_dists.h"
#include "base/line_linker.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"

namespace limap {

void bind_general_structures(py::module &m) {
  py::class_<KDTree>(m, "KDTree")
      .def(py::init<>())
      .def(py::init<const std::vector<V3D> &>())
      .def(py::init<const M3D &>())
      .def("point_distance", &KDTree::point_distance)
      .def("query_nearest", &KDTree::query_nearest)
      .def("save", &KDTree::save)
      .def("load", &KDTree::load);
}

void bind_graph(py::module &m) {
  py::class_<PatchNode>(m, "PatchNode")
      .def(py::init<int, int>())
      .def_readonly("image_idx", &PatchNode::image_idx)
      .def_readonly("line_idx", &PatchNode::line_idx)
      .def_readonly("node_idx", &PatchNode::node_idx)
      .def_readonly("out_edges", &PatchNode::out_edges)
      .def_readonly("in_edges", &PatchNode::in_edges);

  py::class_<Edge>(m, "Edge")
      .def(py::init<size_t, size_t, double>())
      .def_readonly("node_idx1", &Edge::node_idx1)
      .def_readonly("node_idx2", &Edge::node_idx2)
      .def_readonly("edge_idx", &Edge::edge_idx)
      .def_readwrite("similarity", &Edge::sim);

  py::class_<Graph>(m, "Graph")
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
      .def("register_matches",
           [](Graph &self, int im1_idx, int im2_idx,
              py::array_t<size_t, py::array::c_style> &matches,
              py::array_t<double, py::array::c_style> &similarities) {
             py::buffer_info matches_info = matches.request();

             THROW_CHECK_EQ(matches_info.ndim, 2);

             size_t *matches_ptr = static_cast<size_t *>(matches_info.ptr);
             std::vector<ssize_t> matches_shape = matches_info.shape;

             THROW_CHECK_EQ(matches_shape[1], 2);

             size_t n_matches = static_cast<size_t>(matches_shape[0]);
             py::buffer_info sim_info = similarities.request();
             double *sim_ptr = static_cast<double *>(sim_info.ptr);

             assert(n_matches == static_cast<size_t>(sim_info.shape[0]));

             self.RegisterMatches(im1_idx, im2_idx, matches_ptr, sim_ptr,
                                  n_matches);
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
      .def("register_matches",
           [](DirectedGraph &self, int im1_idx, int im2_idx,
              py::array_t<size_t, py::array::c_style> &matches,
              py::array_t<double, py::array::c_style> &similarities) {
             py::buffer_info matches_info = matches.request();

             THROW_CHECK_EQ(matches_info.ndim, 2);

             size_t *matches_ptr = static_cast<size_t *>(matches_info.ptr);
             std::vector<ssize_t> matches_shape = matches_info.shape;

             THROW_CHECK_EQ(matches_shape[1], 2);

             size_t n_matches = static_cast<size_t>(matches_shape[0]);
             py::buffer_info sim_info = similarities.request();
             double *sim_ptr = static_cast<double *>(sim_info.ptr);

             assert(n_matches == static_cast<size_t>(sim_info.shape[0]));

             self.RegisterMatchesDirected(im1_idx, im2_idx, matches_ptr,
                                          sim_ptr, n_matches);
           });

  m.def("compute_track_labels", &ComputeTrackLabels);
  m.def("compute_score_labels", &ComputeScoreLabels);
  m.def("compute_root_labels", &ComputeRootLabels);
  m.def("count_track_edges", &CountTrackEdges);
}

void bind_transforms(py::module &m) {
  py::class_<SimilarityTransform3>(m, "SimilarityTransform3")
      .def(py::init<>())
      .def(py::init<V4D, V3D, double>(), py::arg("qvec"), py::arg("tvec"),
           py::arg("scale") = 1.0)
      .def(py::init<M3D, V3D, double>(), py::arg("R"), py::arg("T"),
           py::arg("scale") = 1.0)
      .def("R", &SimilarityTransform3::R)
      .def("T", &SimilarityTransform3::T)
      .def("s", &SimilarityTransform3::s);
}

void bind_linebase(py::module &m) {
  py::class_<Line2d>(m, "Line2d", "A finite 2D line (segment).")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const Eigen::MatrixXd &>(), R"(
            Constructor from :class:`np.array` of shape (2, 2) stacking the two 2D endpoints
        )",
           py::arg("seg2d"))
      .def(py::init<V2D, V2D>(), R"(
            Constructor from `start` and `end` endpoints, each a :class:`np.array` of shape (2,)
        )",
           py::arg("start"), py::arg("end"))
      .def(py::init<V2D, V2D, double>(), R"(
            Constructor from two endpoints and optionally the score
        )",
           py::arg("start"), py::arg("end"), py::arg("score"))
      .def(py::pickle(
          [](const Line2d &input) { // dump
            return input.as_array();
          },
          [](const Eigen::MatrixXd &arr) { // load
            return Line2d(arr);
          }))
      .def_readwrite("start", &Line2d::start, ":class:`np.array` of shape (2,)")
      .def_readwrite("end", &Line2d::end, ":class:`np.array` of shape (2,)")
      .def_readwrite("score", &Line2d::score, "float")
      .def("length", &Line2d::length, R"(
            Returns:
                float: The length of the 2D line segment
        )")
      .def("coords", &Line2d::coords, R"(
            Returns:
                :class:`np.array` of shape (3,): Normalized homogeneous coordinate of the 2D line
        )")
      .def("as_array", &Line2d::as_array, R"(
            Returns:
                :class:`np.array` of shape (2, 2): Array stacking `start` and `end` endpoints
        )")
      .def("midpoint", &Line2d::midpoint, R"(
            Returns:
                :class:`np.array` of shape (2,): Coordinate of the midpoint of the 2D line segment
        )")
      .def("direction", &Line2d::direction, R"(
            Returns:
                :class:`np.array` of shape (2,): Direction vector of the 2D line from `start` to `end`
        )")
      .def("point_projection", &Line2d::point_projection, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 2D point, of shape (2,)

            Returns:
                :class:`np.array` of shape (2,): Coordinate of the projection of the point `p` on the 2D line segment
        )",
           py::arg("p"))
      .def("point_distance", &Line2d::point_distance, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 2D point, of shape (2,)

            Returns:
                float: Distance from the point `p` to the 2D line segment
        )",
           py::arg("p"));

  py::class_<Line3d>(m, "Line3d", "A finite 3D line (segment).")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const Eigen::MatrixXd &>(), R"(
            Constructor from :class:`np.array` of shape (2, 3) stacking the two 3D endpoints
        )",
           py::arg("seg3d"))
      .def(py::init<V3D, V3D>(), R"(
            Constructor from `start` and `end` endpoints, each a :class:`np.array` of shape (3,)
        )",
           py::arg("start"), py::arg("end"))
      .def(py::init<V3D, V3D, double, double, double, double>(), R"(
            Constructor from two endpoints, and optionally: the score, the start and/or end depth of the 3D segment, and the uncertainty value
        )",
           py::arg("start"), py::arg("end"), py::arg("score"),
           py::arg("depth_start"), py::arg("depth_end"), py::arg("uncertainty"))
      .def(py::pickle(
          [](const Line3d &input) { // dump
            return input.as_array();
          },
          [](const Eigen::MatrixXd &arr) { // load
            return Line3d(arr);
          }))
      .def_readonly("start", &Line3d::start, ":class:`np.array` of shape (3,)")
      .def_readonly("end", &Line3d::end, ":class:`np.array` of shape (3,)")
      .def_readonly("score", &Line3d::score, "float")
      .def_readonly("depths", &Line3d::depths, "float")
      .def_readonly("uncertainty", &Line3d::uncertainty, "float")
      .def("set_uncertainty", &Line3d::set_uncertainty,
           "Setter for the uncertainty value")
      .def("length", &Line3d::length, R"(
            Returns:
                float: The length of the 3D line segment
        )")
      .def("as_array", &Line3d::as_array, R"(
            Returns:
                :class:`np.array` of shape (2, 3): Array stacking `start` and `end` endpoints
        )")
      .def("projection", &Line3d::projection, R"(
            Args:
                view (CameraView): :class:`~limap.base.CameraView` instance used to project the 3D line segment to 2D

            Returns:
                :class:`~limap.base.Line2d`: The 2D line segment projected from the 3D line segment
        )",
           py::arg("view"))
      .def("sensitivity", &Line3d::sensitivity, R"(
            Args:
                view (CameraView): :class:`~limap.base.CameraView` instance

            Returns:
                float: Sensitivity with respect to `view`
        )",
           py::arg("view"))
      .def("computeUncertainty", &Line3d::computeUncertainty, R"(
            Args:
                view (CameraView): :class:`~limap.base.CameraView` instance 
                var2d (float): Variance in 2D

            Returns:
                float: The computed uncertainty value with respect to `view` and `var2d`
        )")
      .def("midpoint", &Line3d::midpoint, R"(
            Returns:
                :class:`np.array` of shape (3,): Coordinate of the midpoint of the 3D line segment
        )")
      .def("direction", &Line3d::direction, R"(
            Returns:
                :class:`np.array` of shape (3,): Direction vector of the 3D line from `start` to `end`
        )")
      .def("point_projection", &Line3d::point_projection, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 3D point, of shape (3,)

            Returns:
                :class:`np.array` of shape (3,): Coordinate of the projection of the point `p` on the 3D line segment
        )",
           py::arg("p"))
      .def("point_distance", &Line3d::point_distance, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 3D point, of shape (3,)

            Returns:
                float: Distance from the point `p` to the 3D line segment
        )",
           py::arg("p"));

  m.def("_GetLine2dVectorFromArray", &GetLine2dVectorFromArray);
  m.def("_GetLine3dVectorFromArray", &GetLine3dVectorFromArray);

  py::class_<InfiniteLine2d>(m, "InfiniteLine2d", "An infinite 2D line.")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const V3D &>(), R"(
            Constructor from homogeneous coordinate (:class:`np.array` of shape (3,))
        )",
           py::arg("coords")) // coords
      .def(py::init<const V2D &, const V2D &>(), R"(
            Constructor from a start point and a direction, both :class:`np.array` of shape (2,)
        )",
           py::arg("p"), py::arg("direc")) // point + direction
      .def(py::init<const Line2d &>(), R"(
            Constructor from a :class:`~limap.base.Line2d`
        )",
           py::arg("line"))
      .def_readonly("coords", &InfiniteLine2d::coords,
                    "Homogeneous coordinate, :class:`np.array` of shape (3,)")
      .def("point", &InfiniteLine2d::point, R"(
            Returns:
                :class:`np.array` of shape (2,): A point on the line (in fact the projection of (0, 0))
        )")
      .def("direction", &InfiniteLine2d::direction, R"(
            Returns:
                :class:`np.array` of shape (2,): The direction of the line
        )")
      .def("point_projection", &InfiniteLine2d::point_projection, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 2D point, of shape (2,)

            Returns:
                :class:`np.array` of shape (2,): Coordinate of the projection of the point `p` on the 2D infinite line
        )",
           py::arg("p"))
      .def("point_distance", &InfiniteLine2d::point_distance, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 2D point, of shape (2,)

            Returns:
                float: Distance from the point `p` to the 2D infinite line
        )",
           py::arg("p"));

  py::class_<InfiniteLine3d>(m, "InfiniteLine3d", "An infinite 3D line.")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<const V3D &, const V3D &, bool>(), R"(
            | Constructor using normal coordinate (a start point and direction) or Plücker coordinate.
            | if `use_normal` is True -> (`a`, `b`) is (`p`, `direc`): normal coordinate with a point and a direction;
            | if `use_normal` is False -> (`a`, `b`) is (`direc`, `m`): Plücker coordinate.
        )",
           py::arg("a"), py::arg("b"), py::arg("use_normal"))
      .def(py::init<const Line3d &>(), R"(
            Constructor from a :class:`~limap.base.Line3d`
        )",
           py::arg("line"))
      .def_readonly("d", &InfiniteLine3d::d,
                    "Direction, :class:`np.array` of shape (3,)")
      .def_readonly("m", &InfiniteLine3d::m,
                    "Moment, :class:`np.array` of shape (3,)")
      .def("point", &InfiniteLine3d::point, R"(
            Returns:
                :class:`np.array` of shape (3,): A point on the line (in fact the projection of (0, 0, 0))
        )")
      .def("direction", &InfiniteLine3d::direction, R"(
            Returns:
                :class:`np.array` of shape (3,): The direction of the line (`d`)
        )")
      .def("matrix", &InfiniteLine3d::matrix, R"(
            Returns:
                :class:`np.array` of shape (4, 4): The `Plücker matrix <https://en.wikipedia.org/wiki/Pl%C3%BCcker_matrix>`_
        )")
      .def("point_projection", &InfiniteLine3d::point_projection, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 3D point, of shape (3,)

            Returns:
                :class:`np.array` of shape (3,): Coordinate of the projection of the point `p` on the 3D infinite line
        )",
           py::arg("p"))
      .def("point_distance", &InfiniteLine3d::point_distance, R"(
            Args:
                p (:class:`np.array`): Coordinate of a 3D point, of shape (3,)

            Returns:
                float: Distance from the point `p` to the 3D infinite line
        )",
           py::arg("p"))
      .def("projection", &InfiniteLine3d::projection, R"(
            Projection from Plücker coordinate to 2D homogeneous line coordinate. 

            Args:
                view (CameraView): :class:`~limap.base.CameraView` instance used to project the 3D infinite line to 2D
            
            Returns:
                :class:`~limap.base.InfiniteLine2D`: The 2D infinite line projected from the 3D infinite line
        )",
           py::arg("view"))
      .def("unprojection", &InfiniteLine3d::unprojection, R"(
            Unproject a 2D point by finding the closest point on the 3D line from the camera ray of the 2D point.

            Args:
                p2d (:class:`np.array`): The 2D point to unproject, of shape (2,)
                view (CameraView): :class:`~limap.base.CameraView` instance to unproject the point
            
            Returns:
                :class:`np.array` of shape (3,): The closest point on the 3D line from the unprojected camera ray of the 2D point
        )",
           py::arg("p2d"), py::arg("view"))
      .def("project_from_infinite_line",
           &InfiniteLine3d::project_from_infinite_line, R"(
            Projection from another infinite 3D line by finding the closest point on this 3D line to the other line.

            Args:
                line (:class:`~limap.base.InfiniteLine3d`): The other infinite line to project from
            
            Returns:
                :class:`np.array` of shape (3,): The projected point on this 3D line from the other line
        )",
           py::arg("line"))
      .def("project_to_infinite_line",
           &InfiniteLine3d::project_to_infinite_line, R"(
            Inverse of the previous function: finding the closest point on the other line to this line.

            Args:
                line (:class:`~limap.base.InfiniteLine3d`): The other infinite line to project to
            
            Returns:
                :class:`np.array` of shape (3,): The projected point on the other line from this line
        )",
           py::arg("line"));

  m.def(
      "_GetLineSegmentFromInfiniteLine3d",
      py::overload_cast<const InfiniteLine3d &, const std::vector<CameraView> &,
                        const std::vector<Line2d> &, const int>(
          &GetLineSegmentFromInfiniteLine3d),
      py::arg("inf_line"), py::arg("camviews"), py::arg("line2ds"),
      py::arg("num_outliers") = 2);
  m.def("_GetLineSegmentFromInfiniteLine3d",
        py::overload_cast<const InfiniteLine3d &, const std::vector<Line3d> &,
                          const int>(&GetLineSegmentFromInfiniteLine3d),
        py::arg("inf_line"), py::arg("line3ds"), py::arg("num_outliers") = 2);
}

void bind_linetrack(py::module &m) {
  py::class_<LineTrack>(m, "LineTrack",
                        "Associated line track across multi-view.")
      .def(py::init<>(), R"(
            Default constructor
        )")
      .def(py::init<LineTrack>(), R"(
            Copy constructor
        )",
           py::arg("track"))
      .def(py::init<const Line3d &, const std::vector<int> &,
                    const std::vector<int> &, const std::vector<Line2d> &>(),
           R"(
            Constructor from a :class:`~limap.base.Line3d`, a list of associated image IDs, a list of supporting line IDs within each image, 
            and a list of associated :class:`~limap.base.Line2d`
        )",
           py::arg("line"), py::arg("image_id_list"), py::arg("line_id_list"),
           py::arg("line2d_list"))
      .def(py::init<py::dict>(), R"(
            Constructor from a Python dict
        )",
           py::arg("dict"))
      .def("as_dict", &LineTrack::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.LineTrack`
        )")
      .def(py::pickle(
          [](const LineTrack &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return LineTrack(dict);
          }))
      .def_readwrite("line", &LineTrack::line,
                     ":class:`~limap.base.Line3d`, the 3D line")
      .def_readonly("image_id_list", &LineTrack::image_id_list,
                    "list[int], the associated image IDs")
      .def_readonly("line_id_list", &LineTrack::line_id_list,
                    "list[int], IDs of supporting 2D lines within each image")
      .def_readonly(
          "line2d_list", &LineTrack::line2d_list,
          "list[:class:`~limap.base.Line2d`], the supporting 2D line segments")
      .def_readonly("active", &LineTrack::active,
                    "bool, active status for recursive merging")
      .def("count_lines", &LineTrack::count_lines, R"(
            Returns:
                int: The number of supporting 2D lines
        )")
      .def("GetSortedImageIds", &LineTrack::GetSortedImageIds, R"(
            Returns:
                list[int]: Sorted (and deduplicated) list of the associated image IDs
        )")
      .def("count_images", &LineTrack::count_images, R"(
            Returns:
                int: Number of unique associated images
        )")
      .def("projection", &LineTrack::projection, R"(
            Project the 3D line to 2D using a list of :class:`~limap.base.CameraView`.

            Args:
                views (list[:class:`~limap.base.CameraView`]): Camera views to project the 3D line
            
            Returns:
                list[:class:`~limap.base.Line2d`]: The 2D projection segments of the 3D line
        )",
           py::arg("views"))
      .def("HasImage", &LineTrack::HasImage, R"(
            Check whether the 3D line has a 2D support from a certain image.

            Args:
                image_id (int): The image ID
            
            Returns:
                bool: True if there is a supporting 2D line from this image
        )",
           py::arg("image_id"))
      .def("Read", &LineTrack::Read, R"(
            Read the line track information from a file.

            Args:
                filename (str): The file to read from
        )",
           py::arg("filename"))
      .def("Write", &LineTrack::Write, R"(
            Write the line track information to a file.

            Args:
                filename (str): The file to write to
        )",
           py::arg("filename"));
}

void bind_line_dists(py::module &m) {
  py::enum_<LineDistType>(m, "LineDistType",
                          "Enum of supported line distance types.")
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
      .value("PERPENDICULAR_SCALEINV_ONEWAY",
             LineDistType::PERPENDICULAR_SCALEINV_ONEWAY)
      .value("PERPENDICULAR_SCALEINV", LineDistType::PERPENDICULAR_SCALEINV)
      .value("ENDPOINTS_SCALEINV_ONEWAY",
             LineDistType::ENDPOINTS_SCALEINV_ONEWAY)
      .value("ENDPOINTS_SCALEINV", LineDistType::ENDPOINTS_SCALEINV)
      .value("INNERSEG", LineDistType::INNERSEG);

  m.def(
      "compute_distance_2d",
      [](const Line2d &l1, const Line2d &l2, const LineDistType &type) {
        return compute_distance<Line2d>(l1, l2, type);
      },
      R"(
            Compute distance between two :class:`~limap.base.Line2d` using the specified line distance type.

            Args:
                l1 (:class:`~limap.base.Line2d`): First 2D line segment
                l2 (:class:`~limap.base.Line2d`): Second 2D line segment
                type (:class:`~limap.base.LineDistType`): Line distance type
            
            Returns:
                `float`: The computed distance
        )",
      py::arg("l1"), py::arg("l2"), py::arg("type"));
  m.def(
      "compute_distance_3d",
      [](const Line3d &l1, const Line3d &l2, const LineDistType &type) {
        return compute_distance<Line3d>(l1, l2, type);
      },
      R"(
            Compute distance between two :class:`~limap.base.Line3d` using the specified line distance type.

            Args:
                l1 (:class:`~limap.base.Line3d`): First 3D line segment
                l2 (:class:`~limap.base.Line3d`): Second 3D line segment
                type (:class:`~limap.base.LineDistType`): Line distance type
            
            Returns:
                `float`: The computed distance
        )",
      py::arg("l1"), py::arg("l2"), py::arg("type"));
  m.def(
      "compute_pairwise_distance_2d",
      [](const std::vector<Line2d> &lines, const LineDistType &type) {
        return compute_pairwise_distance<Line2d>(lines, type);
      },
      R"(
            Compute pairwise distance among a list of :class:`~limap.base.Line2d` using the specified line distance type.

            Args:
                lines (list[:class:`~limap.base.Line2d`]): List of 2D line segments
                type (:class:`~limap.base.LineDistType`): Line distance type
            
            Returns:
                :class:`np.array`: The computed pairwise distance matrix
        )",
      py::arg("lines"), py::arg("type"));
  m.def(
      "compute_pairwise_distance_3d",
      [](const std::vector<Line3d> &lines, const LineDistType &type) {
        return compute_pairwise_distance<Line3d>(lines, type);
      },
      R"(
            Compute pairwise distance among a list of :class:`~limap.base.Line3d` using the specified line distance type.

            Args:
                 lines (list[:class:`~limap.base.Line3d`]): List of 3D line segments
                type (:class:`~limap.base.LineDistType`): Line distance type
            
            Returns:
                :class:`np.array`: The computed pairwise distance matrix
        )",
      py::arg("lines"), py::arg("type"));
}

void bind_line_linker(py::module &m) {
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
      .def(py::init<const LineLinker2dConfig &>())
      .def(py::init<py::dict>())
      .def_readwrite("config", &LineLinker2d::config)
      .def("check_connection", &LineLinker2d::check_connection)
      .def("compute_score", &LineLinker2d::compute_score);

  py::class_<LineLinker3d>(m, "LineLinker3d")
      .def(py::init<>())
      .def(py::init<const LineLinker3dConfig &>())
      .def(py::init<py::dict>())
      .def_readwrite("config", &LineLinker3d::config)
      .def("check_connection", &LineLinker3d::check_connection)
      .def("compute_score", &LineLinker3d::compute_score);

  py::class_<LineLinker>(m, "LineLinker")
      .def(py::init<>())
      .def(py::init<const LineLinker2d &, const LineLinker3d &>())
      .def(py::init<const LineLinker2dConfig &, const LineLinker3dConfig &>())
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

void bind_camera(py::module &m) {
  // TODO: use pycolmap
  py::enum_<colmap::CameraModelId> PyCameraModelId(m, "CameraModelId");
  PyCameraModelId.value("INVALID", colmap::CameraModelId::kInvalid);
  PyCameraModelId.value("SIMPLE_PINHOLE",
                        colmap::CameraModelId::kSimplePinhole);
  PyCameraModelId.value("PINHOLE", colmap::CameraModelId::kPinhole);

  py::class_<Camera>(m, "Camera", R"(
            | Camera model, inherits `COLMAP's camera model <https://colmap.github.io/cameras.html>`_.
            | COLMAP camera models:
            | 0, SIMPLE_PINHOLE
            | 1, PINHOLE
            | 2, SIMPLE_RADIAL
            | 3, RADIAL
            | 4, OPENCV
            | 5, OPENCV_FISHEYE
            | 6, FULL_OPENCV
            | 7, FOV
            | 8, SIMPLE_RADIAL_FISHEYE
            | 9, RADIAL_FISHEYE
            | 10, THIN_PRISM_FISHEYE
        )")
      .def(py::init<>())
      .def(py::init<int, const std::vector<double> &, int,
                    std::pair<int, int>>(),
           py::arg("model_id"), py::arg("params"), py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1))
      .def(py::init<const std::string &, const std::vector<double> &, int,
                    std::pair<int, int>>(),
           py::arg("model_name"), py::arg("params"), py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1))
      .def(py::init<M3D, int, std::pair<int, int>>(), py::arg("K"),
           py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1))
      .def(py::init<int, M3D, int, std::pair<int, int>>(), py::arg("model_id"),
           py::arg("K"), py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1))
      .def(py::init<const std::string &, M3D, int, std::pair<int, int>>(),
           py::arg("model_name"), py::arg("K"), py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1))
      .def(py::init<py::dict>(), py::arg("dict"))
      .def(py::init<const Camera &>(), py::arg("cam"))
      .def(py::init<int, int, std::pair<int, int>>(), py::arg("model_id"),
           py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1)) // empty camera
      .def(py::init<const std::string &, int, std::pair<int, int>>(),
           py::arg("model_name"), py::arg("cam_id") = -1,
           py::arg("hw") = std::make_pair<int, int>(-1, -1)) // empty camera
      .def("__copy__", [](const Camera &self) { return Camera(self); })
      .def("__deepcopy__",
           [](const Camera &self, const py::dict &) { return Camera(self); })
      .def(py::pickle(
          [](const Camera &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return Camera(dict);
          }))
      .def_readwrite("camera_id", &Camera::camera_id,
                     "Unique identifier of the camera.")
      .def_readwrite("model", &Camera::model_id, "Camera model.")
      .def_readwrite("width", &Camera::width, "Width of camera sensor.")
      .def_readwrite("height", &Camera::height, "Height of camera sensor.")
      .def_readwrite("params", &Camera::params, "Camera parameters.")
      .def("as_dict", &Camera::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.Camera`
        )")
      .def("h", &Camera::h, R"(
            Returns:
                int: Image height in pixels
        )")
      .def("w", &Camera::w, R"(
            Returns:
                int: Image width in pixels
        )")
      .def("K", &Camera::K, R"(
            Returns:
                :class:`np.array` of shape (3, 3): Camera's intrinsic matrix
        )")
      .def("K_inv", &Camera::K_inv, R"(
            Returns:
                :class:`np.array` of shape (3, 3): Inverse of the intrinsic matrix
        )")
      .def("resize", &Camera::resize, R"(
            Resize camera's width and height.

            Args:
                width (int)
                height (int)
        )",
           py::arg("width"), py::arg("height"))
      .def("set_max_image_dim", &Camera::set_max_image_dim, R"(
            Set the maximum image dimension, the camera will be resized if the longer dimension of width or height is larger than this value.

            Args:
                val (int)
        )",
           py::arg("val"))
      .def("InitializeParams", &Camera::InitializeParams, R"(
            Initialize the intrinsics using focal length, width, and height

            Args:
                focal_length (double)
                width (int)
                height (int)
        )",
           py::arg("focal_length"), py::arg("width"), py::arg("height"))
      .def("CamFromImg", &Camera::CamFromImg)
      .def("ImgFromCam", &Camera::ImgFromCam)
      .def("IsInitialized", &Camera::IsInitialized, R"(
            Returns:
                bool: True if the camera parameters are initialized
        )")
      .def("IsUndistorted", &Camera::IsUndistorted, R"(
            Returns:
                bool: True if the camera model is without distortion
        )");

  py::class_<CameraPose>(m, "CameraPose",
                         "Representing the world-to-cam pose (R, t) with a "
                         "quaternion and a translation vector. The quaternion "
                         "convention is `(w, x, y, z)` (real part first).")
      .def(py::init<bool>(), R"(
            Default constructor: identity pose
        )",
           py::arg("initialized") = false)
      .def(py::init<const CameraPose &>(), R"(
            Copy constructor
        )",
           py::arg("campose"))
      .def(py::init<V4D, V3D, bool>(), R"(
            Constructor from a quaternion vector and a translation vector
        )",
           py::arg("qvec"), py::arg("tvec"), py::arg("initialized") = true)
      .def(py::init<M3D, V3D, bool>(), R"(
            Constructor from a rotation matrix and a translation vector
        )",
           py::arg("R"), py::arg("tvec"), py::arg("initialized") = true)
      .def(py::init<py::dict>(), R"(
            Constructor from a Python dict
        )",
           py::arg("dict"))
      .def("__copy__", [](const CameraPose &self) { return CameraPose(self); })
      .def("__deepcopy__", [](const CameraPose &self,
                              const py::dict &) { return CameraPose(self); })
      .def(py::pickle(
          [](const CameraPose &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return CameraPose(dict);
          }))
      .def("as_dict", &CameraPose::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.CameraPose`
        )")
      .def_readonly("qvec", &CameraPose::qvec,
                    ":class:`np.array` of shape (4,): The quaternion vector "
                    "`(w, x, y, z)`")
      .def_readonly("tvec", &CameraPose::tvec,
                    ":class:`np.array` of shape (3,): The translation vector")
      .def_readwrite(
          "initialized", &CameraPose::initialized,
          "bool: Flag indicating whether the pose has been initialized")
      .def("R", &CameraPose::R, R"(
            Returns:
                :class:`np.array` of shape (3, 3): The rotation matrix
        )")
      .def("T", &CameraPose::T, R"(
            Returns:
                :class:`np.array` of shape (3,): The translation vector
        )")
      .def("center", &CameraPose::center, R"(
            Returns:
                :class:`np.array` of shape (3,): World-space coordinate of the camera
        )")
      .def("projdepth", &CameraPose::projdepth, R"(
            Args:
                p3d (:class:`np.array`): World-space coordinate of a 3D point
            
            Returns:
                float: The projection depth of the 3D point viewed from this camera pose
        )",
           py::arg("p3d"));

  py::class_<CameraImage>(
      m, "CameraImage",
      "This class associates the ID of a :class:`~limap.base.Camera`, a "
      ":class:`~limap.base.CameraPose`, and an image file")
      .def(py::init<>())
      .def(py::init<const int &, const std::string &>(), py::arg("cam_id"),
           py::arg("image_name") = "none") // empty image
      .def(py::init<const Camera &, const std::string &>(), py::arg("camera"),
           py::arg("image_name") = "none") // empty image
      .def(py::init<const int &, const CameraPose &, const std::string &>(),
           py::arg("cam_id"), py::arg("pose"), py::arg("image_name") = "none")
      .def(py::init<const Camera &, const CameraPose &, const std::string &>(),
           py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
      .def(py::init<py::dict>(), py::arg("dict"))
      .def(py::init<const CameraImage &>(), py::arg("camimage"))
      .def("__copy__",
           [](const CameraImage &self) { return CameraImage(self); })
      .def("__deepcopy__", [](const CameraImage &self,
                              const py::dict &) { return CameraImage(self); })
      .def(py::pickle(
          [](const CameraImage &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return CameraImage(dict);
          }))
      .def("as_dict", &CameraImage::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.CameraImage`
        )")
      .def_readonly("cam_id", &CameraImage::cam_id, "int, the camera ID")
      .def_readonly("pose", &CameraImage::pose,
                    ":class:`~limap.base.CameraPose`, the camera pose")
      .def("R", &CameraImage::R, R"(
            Returns:
                :class:`np.array` of shape (3, 3): The rotation matrix of the camera pose
        )")
      .def("T", &CameraImage::T, R"(
            Returns:
                :class:`np.array` of shape (3,): The translation vector of the camera pose
        )")
      .def("set_camera_id", &CameraImage::SetCameraId, R"(
            Set the camera ID.

            Args:
                cam_id (int)
        )",
           py::arg("cam_id"))
      .def("image_name", &CameraImage::image_name, R"(
            Returns:
                str: The image file name
        )")
      .def("set_image_name", &CameraImage::SetImageName, R"(
            Set the name of the image file.

            Args:
                image_name (str)
        )",
           py::arg("image_name"));

  py::class_<CameraView>(
      m, "CameraView",
      "Inherits :class:`~limap.base.CameraImage`, incorporating the "
      ":class:`~limap.base.Camera` model and its parameters for "
      "projection/unprojection between 2D and 3D.")
      .def(py::init<>())
      .def(py::init<const std::string &>()) // empty view
      .def(py::init<const Camera &, const std::string &>(), py::arg("camera"),
           py::arg("image_name") = "none") // empty view
      .def(py::init<const Camera &, const CameraPose &, const std::string &>(),
           py::arg("camera"), py::arg("pose"), py::arg("image_name") = "none")
      .def(py::init<py::dict>(), py::arg("dict"))
      .def(py::init<const CameraView &>(), py::arg("camview"))
      .def("__copy__", [](const CameraView &self) { return CameraView(self); })
      .def("__deepcopy__", [](const CameraView &self,
                              const py::dict &) { return CameraView(self); })
      .def(py::pickle(
          [](const CameraView &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return CameraView(dict);
          }))
      .def_readonly("cam", &CameraView::cam,
                    ":class:`~limap.base.Camera`, the camera model")
      .def_readonly("pose", &CameraView::pose,
                    ":class:`~limap.base.CameraPose`, the camera pose")
      .def("as_dict", &CameraView::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.CameraView`
        )")
      .def("read_image", &CameraView::read_image, R"(
            Read image data from the image file.

            Args:
                set_gray (bool): Whether to convert the image to gray. Default False.

            Returns:
                :class:`np.array`: The image data matrix
        )",
           py::arg("set_gray") = false)
      .def("K", &CameraView::K, R"(
            Returns:
                :class:`np.array` of shape (3, 3): The intrinsic matrix of the camera
        )")
      .def("K_inv", &CameraView::K_inv, R"(
            Returns:
                :class:`np.array` of shape (3, 3): The inverse of the camera's intrinsic matrix
        )")
      .def("h", &CameraView::h, R"(
            Returns:
                int: Image height in pixels
        )")
      .def("w", &CameraView::w, R"(
            Returns:
                int: Image width in pixels
        )")
      .def("R", &CameraView::R, R"(
            Returns:
                :class:`np.array` of shape (3, 3): The rotation matrix of the camera pose
        )")
      .def("T", &CameraView::T, R"(
            Returns:
                :class:`np.array` of shape (3,): The translation vector of the camera pose
        )")
      .def("matrix", &CameraView::matrix, R"(
            Returns:
                :class:`np.array` of shape (3, 4): The projection matrix `P = K[R|T]`
        )")
      .def("projection", &CameraView::projection, R"(
            Args:
                p3d (:class:`np.array`): World-space coordinate of a 3D point
            
            Returns:
                :class:`np.array` of shape (2,): The 2D pixel-space coordinate of the point's projection on image 
        )",
           py::arg("p3d"))
      .def("ray_direction", &CameraView::ray_direction, R"(
            Args:
                p2d (:class:`np.array`): Pixel-space coordinate of a 2D point on the image
            
            Returns:
                :class:`np.array` of shape (3,): The world-space direction of the camera ray passing the 2D point
        )",
           py::arg("p2d"))
      .def("get_direction_from_vp", &CameraView::get_direction_from_vp, R"(
            Args:
                vp (:class:`np.array`): The coordinate of a vanishing point
            
            Returns:
                :class:`np.array` of shape (3,): The direction from the vanishing point
        )",
           py::arg("vp"))
      .def("image_name", &CameraView::image_name, R"(
            Returns:
                str: The image file name
        )")
      .def("set_image_name", &CameraView::SetImageName, R"(
            Set the name of the image file.

            Args:
                image_name (str)
        )",
           py::arg("image_name"))
      .def("get_initial_focal_length", &CameraView::get_initial_focal_length,
           R"(
            Try to get the focal length information from the image's EXIF data.
            If not available in image EXIF, the focal length is estimated by the max dimension of the image.

            Returns:
                tuple[double, bool]: Initial focal length and a flag indicating if the value is read from image's EXIF data.
        )");

  py::class_<ImageCollection>(m, "ImageCollection", R"(
            A flexible class that consists of cameras and images in a scene or dataset. In each image stores the corresponding ID of the camera, making it easy to extend to single/multiple sequences or unstructured image collections. 
            The constructor arguments `input_cameras` and `input_images` can be either list of :class:`~limap.base.Camera` and :class:`~limap.base.CameraImage` 
            or python dict mapping integer IDs to :class:`~limap.base.Camera` and :class:`~limap.base.CameraImage`.
        )")
      .def(py::init<>())
      .def(py::init<const std::map<int, Camera> &,
                    const std::map<int, CameraImage> &>(),
           py::arg("input_cameras"), py::arg("input_images"))
      .def(py::init<const std::vector<Camera> &,
                    const std::map<int, CameraImage> &>(),
           py::arg("input_cameras"), py::arg("input_images"))
      .def(py::init<const std::map<int, Camera> &,
                    const std::vector<CameraImage> &>(),
           py::arg("input_cameras"), py::arg("input_images"))
      .def(py::init<const std::vector<Camera> &,
                    const std::vector<CameraImage> &>(),
           py::arg("input_cameras"), py::arg("input_images"))
      .def(py::init<const std::vector<CameraView> &>(), py::arg("camviews"))
      .def(py::init<py::dict>(), py::arg("dict"))
      .def(py::init<const ImageCollection &>(), py::arg("imagecols"))
      .def("__copy__",
           [](const ImageCollection &self) { return ImageCollection(self); })
      .def("__deepcopy__",
           [](const ImageCollection &self, const py::dict &) {
             return ImageCollection(self);
           })
      .def(py::pickle(
          [](const ImageCollection &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return ImageCollection(dict);
          }))
      .def("as_dict", &ImageCollection::as_dict, R"(
            Returns:
                dict: Python dict representation of this :class:`~limap.base.ImageCollection`
        )")
      .def("subset_by_camera_ids", &ImageCollection::subset_by_camera_ids, R"(
            Filter the images using camera IDs.

            Args:
                valid_camera_ids (list[int]): Images from camera with these IDs are kept in the filtered subset

            Returns:
                :class:`~limap.base.ImageCollection`: The filtered subset collection
        )",
           py::arg("valid_camera_ids"))
      .def("subset_by_image_ids", &ImageCollection::subset_by_image_ids, R"(
            Filter the images using image IDs.

            Args:
                valid_image_ids (list[int]): IDs of images to be kept in the filtered subset

            Returns:
                :class:`~limap.base.ImageCollection`: The filtered subset collection
        )",
           py::arg("valid_image_ids"))
      .def("subset_initialized", &ImageCollection::subset_initialized, R"(
            Filter the images to create a subset collection containing only images with initialized camera poses.

            Returns:
                :class:`~limap.base.ImageCollection`: The filtered subset collection
        )")
      .def("update_neighbors", &ImageCollection::update_neighbors, R"(
            Update the neighbor information among images (e.g. after filtering). Remove neighboring images that are not in the image collection.

            Args:
                neighbors (dict[int -> list[int]]): The input neighbor information

            Returns:
                dict[int -> list[int]]: Updated neighbor information
        )")
      .def("get_cameras", &ImageCollection::get_cameras, R"(
            Returns:
                list[:class:`~limap.base.Camera`]: All cameras in the collection
        )")
      .def("get_cam_ids", &ImageCollection::get_cam_ids, R"(
            Returns:
                list[int]: IDs of all cameras in the collection
        )")
      .def("get_images", &ImageCollection::get_images, R"(
            Returns:
                list[:class:`~limap.base.CameraImage`]: All images in the collection
        )")
      .def("get_img_ids", &ImageCollection::get_img_ids, R"(
            Returns:
                list[int]: IDs of all images in the collection
        )")
      .def("get_camviews", &ImageCollection::get_camviews, R"(
            Returns:
                list[:class:`~limap.base.CameraView`]: The associated :class:`~limap.base.CameraView` from all the images and their cameras in the collection
        )")
      .def("get_map_camviews", &ImageCollection::get_map_camviews, R"(
            Returns:
                dict[int -> :class:`~limap.base.CameraView`]: Mapping of image IDs to their associated :class:`~limap.base.CameraView`
        )")
      .def("get_locations", &ImageCollection::get_locations, R"(
            Returns:
                list[:class:`np.array`]: The world-space locations of the camera for all images in the collection, each of shape (3, )
        )")
      .def("get_map_locations", &ImageCollection::get_map_locations, R"(
            Returns:
                dict[int -> :class:`np.array`]: Mapping of image IDs to their camera locations in world-space
        )")
      .def("exist_cam", &ImageCollection::exist_cam, R"(
            Args:
                cam_id (int)

            Returns:
                bool: True if the camera with `cam_id` exists in the collection
        )",
           py::arg("cam_id"))
      .def("exist_image", &ImageCollection::exist_image, R"(
            Args:
                img_id (int)

            Returns:
                bool: True if the image with `img_id` exists in the collection
        )",
           py::arg("img_id"))
      .def("cam", &ImageCollection::cam, R"(
            Args:
                cam_id (int)

            Returns:
                :class:`~limap.base.Camera`: The camera with `cam_id`
        )",
           py::arg("cam_id"))
      .def("camimage", &ImageCollection::camimage, R"(
            Args:
                img_id (int)

            Returns:
                :class:`~limap.base.CameraImage`: The image with `img_id`
        )",
           py::arg("img_id"))
      .def("campose", &ImageCollection::campose, R"(
            Args:
                img_id (int)

            Returns:
                :class:`~limap.base.CameraPose`: The camera pose of the image
        )",
           py::arg("img_id"))
      .def("camview", &ImageCollection::camview, R"(
            Args:
                img_id (int)

            Returns:
                :class:`~limap.base.CameraView`: The :class:`~limap.base.CameraView` from the image
        )",
           py::arg("img_id"))
      .def("image_name", &ImageCollection::image_name, R"(
            Args:
                img_id (int)

            Returns:
                str: The file name of the image
        )",
           py::arg("img_id"))
      .def("get_image_name_list", &ImageCollection::get_image_name_list, R"(
            Returns:
                list[str]: All the image file names
        )")
      .def("get_image_name_dict", &ImageCollection::get_image_name_dict, R"(
            Returns:
                dict[int -> str]: Mapping of image IDs to the file names
        )")
      .def("NumCameras", &ImageCollection::NumCameras, R"(
            Returns:
                int: The number of cameras in the collection
        )")
      .def("NumImages", &ImageCollection::NumImages, R"(
            Returns:
                int: The number of images in the collection
        )")
      .def("set_max_image_dim", &ImageCollection::set_max_image_dim, R"(
            Set the maximum image dimension for all cameras using :py:meth:`~limap.base.Camera.set_max_image_dim`.

            Args:
                val (int)
        )",
           py::arg("val"))
      .def("change_camera", &ImageCollection::change_camera, R"(
            Change the camera model of a specific camera.

            Args:
                cam_id (int)
                cam (:class:`~limap.base.Camera`)
        )",
           py::arg("cam_id"), py::arg("cam"))
      .def("set_camera_pose", &ImageCollection::set_camera_pose, R"(
            Set the camera pose for a specific image.

            Args:
                img_id (int)
                pose (:class:`~limap.base.CameraPose`)
        )",
           py::arg("img_id"), py::arg("pose"))
      .def("get_camera_pose", &ImageCollection::get_camera_pose, R"(
            Get the camera pose of a specific image.

            Args:
                img_id (int)

            Returns:
                :class:`~limap.base.CameraPose`
        )",
           py::arg("img_id"))
      .def("change_image", &ImageCollection::change_image, R"(
            Change an image.

            Args:
                img_id (int)
                camimage (:class:`~limap.base.CameraImage`)
        )",
           py::arg("img_id"), py::arg("camimage"))
      .def("change_image_name", &ImageCollection::change_image_name, R"(
            Change the file name of an image.

            Args:
                img_id (int)
                new_name (str)
        )",
           py::arg("img_id"), py::arg("new_name"))
      .def("IsUndistorted", &ImageCollection::IsUndistorted, R"(
            Returns:
                bool: True if all cameras in the collection are without distortion, see :py:meth:`~limap.base.Camera.IsUndistorted`.
        )")
      .def("read_image", &ImageCollection::read_image, R"(
            Read an image, calls :py:meth:`~limap.base.CameraView.read_image`.

            Args:
                img_id (int): The image ID
                set_gray (bool): Whether to convert the image to gray. Default False.

            Returns:
                :class:`np.array`: The image data matrix
        )",
           py::arg("img_id"), py::arg("set_gray") = false)
      .def("apply_similarity_transform",
           &ImageCollection::apply_similarity_transform, R"(
            Apply similarity transform to all image poses.

            Args:
                transform (:class:`limap.base.SimilarityTransform3`)
        )",
           py::arg("transform"))
      .def("get_first_image_id_by_camera_id",
           &ImageCollection::get_first_image_id_by_camera_id, R"(
            Get the ID of the first image captured with a specific camera.

            Args:
                cam_id (int): The camera ID
            
            Return:
                int: The image ID.
        )",
           py::arg("cam_id"))
      .def("init_uninitialized_cameras",
           &ImageCollection::init_uninitialized_cameras, R"(
            Initialize all uninitialized cameras by :func:`~limap.base.Camera.InitializeParams`.
        )")
      .def("uninitialize_poses", &ImageCollection::uninitialize_poses, R"(
            Uninitialize camera poses for all images, set them to identity poses and remove the :attr:`~limap.base.CameraPose.initialized` flag.
        )")
      .def("uninitialize_intrinsics", &ImageCollection::uninitialize_intrinsics,
           R"(
            Uninitialize intrinsics for all cameras.
        )")
      .def("IsUndistortedCameraModel",
           &ImageCollection::IsUndistortedCameraModel, R"(
            Returns:
                bool: True if all camera models are undistorted.
        )");

  m.def("pose_similarity_transform", &pose_similarity_transform);
}

void bind_pointtrack(py::module &m) {
  py::class_<Point2d>(m, "Point2d")
      .def(py::init<>())
      .def(py::init<V2D, int>(), py::arg("p"), py::arg("point3D_id") = -1)
      .def("as_dict", &Point2d::as_dict)
      .def(py::pickle(
          [](const Point2d &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return Point2d(dict);
          }))
      .def_readwrite("p", &Point2d::p)
      .def_readwrite("point3D_id", &Point2d::point3D_id);

  py::class_<PointTrack>(m, "PointTrack")
      .def(py::init<>())
      .def(py::init<const PointTrack &>())
      .def(py::init<const V3D &, const std::vector<int> &,
                    const std::vector<int> &, const std::vector<V2D> &>())
      .def(py::init<py::dict>())
      .def("as_dict", &PointTrack::as_dict)
      .def(py::pickle(
          [](const PointTrack &input) { // dump
            return input.as_dict();
          },
          [](const py::dict &dict) { // load
            return PointTrack(dict);
          }))
      .def_readwrite("p", &PointTrack::p)
      .def_readonly("image_id_list", &PointTrack::image_id_list)
      .def_readonly("p2d_id_list", &PointTrack::p2d_id_list)
      .def_readonly("p2d_list", &PointTrack::p2d_list)
      .def("count_images", &PointTrack::count_images);
}

void bind_base(py::module &m) {
  bind_general_structures(m);
  bind_graph(m);
  bind_transforms(m);
  bind_pointtrack(m);
  bind_camera(m);
  bind_linebase(m);
  bind_linetrack(m);
  bind_line_dists(m);
  bind_line_linker(m);

  m.def("get_effective_num_threads", &colmap::GetEffectiveNumThreads);
}

} // namespace limap
