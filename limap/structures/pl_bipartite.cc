#include "structures/pl_bipartite.h"

#include "base/graph.h"
#include "base/infinite_line.h"
#include "util/kd_tree.h"

namespace limap {

namespace structures {

py::dict PL_Bipartite2d::as_dict() const {
  py::dict output;
  std::map<int, py::dict> dict_points;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    dict_points.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["points_"] = dict_points;
  std::map<int, Eigen::MatrixXd> dict_lines;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    dict_lines.insert(std::make_pair(it->first, it->second.as_array()));
  }
  output["lines_"] = dict_lines;
  output["np2l_"] = np2l_;
  output["nl2p_"] = nl2p_;
  return output;
}

PL_Bipartite2d::PL_Bipartite2d(py::dict dict) {
  // load points
  std::map<int, py::dict> dict_points;
  if (dict.contains("points_"))
    dict_points = dict["points_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"points_\" does not exist!");
  for (auto it = dict_points.begin(); it != dict_points.end(); ++it) {
    points_.insert(std::make_pair(it->first, Point2d(it->second)));
  }

  // load lines
  std::map<int, Eigen::MatrixXd> dict_lines;
  if (dict.contains("lines_"))
    dict_lines = dict["lines_"].cast<std::map<int, Eigen::MatrixXd>>();
  else
    throw std::runtime_error("Error! Key \"lines_\" does not exist!");
  for (auto it = dict_lines.begin(); it != dict_lines.end(); ++it) {
    lines_.insert(std::make_pair(it->first, Line2d(it->second)));
  }

  // load connections
#define TMPMAPTYPE std::map<int, std::set<int>>
  ASSIGN_PYDICT_ITEM(dict, np2l_, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, nl2p_, TMPMAPTYPE)
#undef TMPMAPTYPE
}

void PL_Bipartite2d::add_keypoint(const Point2d &p, int point_id) {
  if (point_id == -1)
    point_id = get_default_new_point_id();
  std::vector<int> neighbors;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    double dist = it->second.point_distance(p.p);
    if (dist > config_.threshold_keypoints)
      continue;
    neighbors.push_back(it->first);
  }
  add_point(p, point_id, neighbors);
}

void PL_Bipartite2d::add_keypoints_with_point3D_ids(
    const std::vector<V2D> &points, const std::vector<int> &point3D_ids,
    const std::vector<int> &ids) {
  THROW_CHECK_EQ(points.size(), point3D_ids.size());
  if (!ids.empty())
    THROW_CHECK_EQ(points.size(), ids.size());
  size_t n_points = points.size();
  for (size_t i = 0; i < n_points; ++i) {
    int point_id = -1;
    if (!ids.empty())
      point_id = ids[i];
    add_keypoint(Point2d(points[i], point3D_ids[i]), point_id);
  }
}

void PL_Bipartite2d::compute_intersection() {
  std::vector<V2D> points;
  for (auto it = points_.begin(); it != points_.end(); ++it)
    points.push_back(it->second.p);
  compute_intersection_with_points(points);
}

void PL_Bipartite2d::compute_intersection_with_points(
    const std::vector<V2D> &points) {
  std::vector<V3D> tmp_points;
  for (auto it = points.begin(); it != points.end(); ++it) {
    V2D p2d = *it;
    tmp_points.push_back(V3D(p2d(0), p2d(1), 0.0));
  }
  KDTree tree(tmp_points);
  std::vector<int> line_ids = get_line_ids();

  // add all endpoints
  std::vector<Junction<V2D>> intersections;
  for (size_t i = 0; i < count_lines(); ++i) {
    int line_id = line_ids[i];
    std::vector<int> ids;
    ids.push_back(line_id);
    intersections.push_back(Junction<V2D>(line(line_id).start, ids));
    intersections.push_back(Junction<V2D>(line(line_id).end, ids));
  }

  // exhaustive intersecting pairs
  for (size_t i = 0; i < count_lines() - 1; ++i) {
    int line_id1 = line_ids[i];
    for (size_t j = i + 1; j < count_lines(); ++j) {
      int line_id2 = line_ids[j];
      auto res = intersect(line(line_id1), line(line_id2));
      if (!res.first)
        continue;
      std::vector<int> ids;
      ids.push_back(line_id1);
      ids.push_back(line_id2);
      intersections.push_back(Junction<V2D>(res.second, ids));
    }
  }

  // exhaustive merge junctions
  size_t n_inters = intersections.size();
  std::vector<int> parents(n_inters, -1);
  for (size_t i = 0; i < n_inters - 1; ++i) {
    for (size_t j = i + 1; j < n_inters; ++j) {
      int rooti = union_find_get_root(i, parents);
      int rootj = union_find_get_root(j, parents);
      if (rooti == rootj)
        continue;
      double dist = (intersections[i].p - intersections[j].p).norm();
      if (dist > config_.threshold_merge_junctions) // 1 pixel
        continue;
      if (i < j)
        parents[j] = i;
      else
        parents[i] = j;
    }
  }
  std::map<int, std::vector<Junction<V2D>>> m;
  for (size_t i = 0; i < n_inters; ++i) {
    int root = union_find_get_root(i, parents);
    if (m.find(root) == m.end()) {
      m.insert(std::make_pair(root, std::vector<Junction<V2D>>()));
    }
    m[root].push_back(intersections[i]);
  }
  for (auto it = m.begin(); it != m.end(); ++it) {
    Junction<V2D> junc = merge_junctions(it->second);
    // test if it overlaps existing junctions
    if (!tree.empty()) {
      V2D p2d = junc.p;
      V3D p3d = V3D(p2d(0), p2d(1), 0.0);
      double dist = tree.point_distance(p3d);
      if (dist < config_.threshold_merge_junctions)
        continue;
    }
    add_junction(Junction2d(Point2d(junc.p), junc.line_ids));
  }
}

std::pair<bool, V2D> PL_Bipartite2d::intersect(const Line2d &l1,
                                               const Line2d &l2) const {
  double threshold = config_.threshold_intersection;
  // test endpoints
  if ((l1.start - l2.start).norm() <= threshold)
    return std::make_pair(true, (l1.start + l2.start) / 2.0);
  if ((l1.end - l2.start).norm() <= threshold)
    return std::make_pair(true, (l1.end + l2.start) / 2.0);
  if ((l1.start - l2.end).norm() <= threshold)
    return std::make_pair(true, (l1.start + l2.end) / 2.0);
  if ((l1.end - l2.end).norm() <= threshold)
    return std::make_pair(true, (l1.end + l2.end) / 2.0);

  // intersect
  V3D coor1 = l1.coords();
  V3D coor2 = l2.coords();
  V3D junc_homo = coor1.cross(coor2);
  junc_homo = junc_homo.normalized();
  double px = junc_homo(0) / (junc_homo(2) + EPS);
  double py = junc_homo(1) / (junc_homo(2) + EPS);
  V2D junc = V2D(px, py);

  // test if it is on the line
  double proj1 = (junc - l1.start).dot(l1.direction());
  double error1 = 0.0;
  if (proj1 < 0.0)
    error1 = -proj1;
  if (proj1 > l1.length())
    error1 = proj1 - l1.length();
  double proj2 = (junc - l2.start).dot(l2.direction());
  double error2 = 0.0;
  if (proj2 < 0.0)
    error2 = -proj2;
  if (proj2 > l2.length())
    error2 = proj2 - l2.length();
  if (error1 + error2 > threshold)
    return std::make_pair(false, V2D(0.0, 0.0));
  return std::make_pair(true, junc);
}

Junction<V2D>
PL_Bipartite2d::merge_junctions(const std::vector<Junction<V2D>> juncs) const {
  int n_juncs = juncs.size();
  V2D p(0.0, 0.0);
  std::set<int> s;
  for (int i = 0; i < n_juncs; ++i) {
    p += juncs[i].p;
    for (auto it = juncs[i].line_ids.begin(); it != juncs[i].line_ids.end();
         ++it)
      s.insert(*it);
  }
  p /= n_juncs;
  std::vector<int> line_ids;
  for (auto it = s.begin(); it != s.end(); ++it) {
    line_ids.push_back(*it);
  }
  return Junction<V2D>(p, line_ids);
}

py::dict PL_Bipartite3d::as_dict() const {
  py::dict output;
  std::map<int, py::dict> dict_points;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    dict_points.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["points_"] = dict_points;
  std::map<int, py::dict> dict_lines;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    dict_lines.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["lines_"] = dict_lines;
  output["np2l_"] = np2l_;
  output["nl2p_"] = nl2p_;
  return output;
}

PL_Bipartite3d::PL_Bipartite3d(py::dict dict) {
  // load points
  std::map<int, py::dict> dict_points;
  if (dict.contains("points_"))
    dict_points = dict["points_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"points_\" does not exist!");
  for (auto it = dict_points.begin(); it != dict_points.end(); ++it) {
    points_.insert(std::make_pair(it->first, PointTrack(it->second)));
  }

  // load lines
  std::map<int, py::dict> dict_lines;
  if (dict.contains("lines_"))
    dict_lines = dict["lines_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"lines_\" does not exist!");
  for (auto it = dict_lines.begin(); it != dict_lines.end(); ++it) {
    lines_.insert(std::make_pair(it->first, LineTrack(it->second)));
  }

  // load connections
#define TMPMAPTYPE std::map<int, std::set<int>>
  ASSIGN_PYDICT_ITEM(dict, np2l_, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, nl2p_, TMPMAPTYPE)
#undef TMPMAPTYPE
}

std::vector<V3D> PL_Bipartite3d::get_point_cloud() const {
  std::vector<V3D> output;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    output.push_back(it->second.p);
  }
  return output;
}

std::vector<Line3d> PL_Bipartite3d::get_line_cloud() const {
  std::vector<Line3d> output;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    output.push_back(it->second.line);
  }
  return output;
}

} // namespace structures

} // namespace limap
