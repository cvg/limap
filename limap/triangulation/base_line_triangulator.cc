#include "triangulation/base_line_triangulator.h"
#include "triangulation/functions.h"

#include <algorithm>
#include <colmap/util/logging.h>
#include <iostream>
#include <third-party/progressbar.hpp>

namespace limap {

namespace triangulation {

void BaseLineTriangulator::offsetHalfPixel() {
  std::vector<int> image_ids = imagecols_->get_img_ids();
  for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
    int img_id = *it;
    for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
      auto &line = all_lines_2d_[img_id][line_id];
      line.start = line.start + V2D(0.5, 0.5);
      line.end = line.end + V2D(0.5, 0.5);
    }
  }
}

void BaseLineTriangulator::Init(
    const std::map<int, std::vector<Line2d>> &all_2d_segs,
    const ImageCollection *imagecols) {
  all_lines_2d_ = all_2d_segs;
  THROW_CHECK_EQ(imagecols->IsUndistorted(), true);
  imagecols_ = imagecols;
  if (config_.add_halfpix)
    offsetHalfPixel();

  // initialize empty containers
  for (const int &img_id : imagecols->get_img_ids()) {
    size_t n_lines = all_2d_segs.at(img_id).size();
    neighbors_.insert(std::make_pair(img_id, std::vector<int>()));
    edges_.insert(std::make_pair(img_id, std::vector<std::vector<LineNode>>()));
    edges_.at(img_id).resize(n_lines);
    tris_.insert(std::make_pair(img_id, std::vector<std::vector<TriTuple>>()));
    tris_.at(img_id).resize(n_lines);
  }
}

void BaseLineTriangulator::Init(
    const std::map<int, std::vector<Line2d>> &all_2d_segs,
    const ImageCollection &imagecols) {
  return Init(all_2d_segs, &imagecols);
}

void BaseLineTriangulator::TriangulateImage(
    const int img_id, const std::map<int, Eigen::MatrixXi> &matches) {
  neighbors_[img_id].clear();
  for (auto it = matches.begin(); it != matches.end(); ++it) {
    int ng_img_id = it->first;
    const auto &match_info = it->second;
    neighbors_[img_id].push_back(ng_img_id);
    if (match_info.rows() != 0) {
      THROW_CHECK_EQ(match_info.cols(), 2);
    }
    size_t n_matches = match_info.rows();
    std::vector<size_t> matches;
    for (int k = 0; k < n_matches; ++k) {
      // good match exists
      int line_id = match_info(k, 0);
      int ng_line_id = match_info(k, 1);
      if (line_id >= edges_[img_id].size()) {
        throw std::runtime_error(
            "IndexError! Out-of-index matches exist between image (img_id = " +
            std::to_string(img_id) +
            ") and neighbor image (img_id = " + std::to_string(ng_img_id) +
            "). Please make sure you are reusing the correct descriptors and "
            "matches when using the --skip_exists option.");
      }
      edges_[img_id][line_id].push_back(std::make_pair(ng_img_id, ng_line_id));
    }
    for (size_t line_id = 0; line_id < CountLines(img_id); ++line_id) {
      triangulateOneNode(img_id, line_id);
      clearEdgesOneNode(img_id, line_id);
    }
    edges_[img_id].clear();
    edges_[img_id].resize(CountLines(img_id));
  }
  ScoringCallback(img_id);
  if (!config_.debug_mode) {
    tris_[img_id].clear();
    tris_[img_id].resize(CountLines(img_id));
  }
}

void BaseLineTriangulator::TriangulateImageExhaustiveMatch(
    const int img_id, const std::vector<int> &neighbors) {
  neighbors_[img_id].clear();
  neighbors_[img_id] = neighbors;
  size_t n_neighbors = neighbors.size();
  for (size_t neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id) {
    int ng_img_id = neighbors[neighbor_id];
    int n_lines_ng = all_lines_2d_[ng_img_id].size();
    size_t n_lines = CountLines(img_id);
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
      for (int ng_line_id = 0; ng_line_id < n_lines_ng; ++ng_line_id) {
        edges_[img_id][line_id].push_back(
            std::make_pair(ng_img_id, ng_line_id));
      }
      triangulateOneNode(img_id, line_id);
      clearEdgesOneNode(img_id, line_id);
    }
    edges_[img_id].clear();
    edges_[img_id].resize(CountLines(img_id));
  }
  ScoringCallback(img_id);
  if (!config_.debug_mode) {
    tris_[img_id].clear();
    tris_[img_id].resize(CountLines(img_id));
  }
}

void BaseLineTriangulator::clearEdgesOneNode(const int img_id,
                                             const int line_id) {
  edges_[img_id][line_id].clear();
}

void BaseLineTriangulator::clearEdges() {
  for (const int &img_id : imagecols_->get_img_ids()) {
    for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
      clearEdgesOneNode(img_id, line_id);
    }
  }
}

int BaseLineTriangulator::countEdges() const {
  int counter = 0;
  for (const int &img_id : imagecols_->get_img_ids()) {
    for (int line_id = 0; line_id < CountLines(img_id); ++line_id) {
      counter += edges_.at(img_id)[line_id].size();
    }
  }
  return counter;
}

void BaseLineTriangulator::triangulateOneNode(const int img_id,
                                              const int line_id) {
  THROW_CHECK_EQ(imagecols_->exist_image(img_id), true);
  auto &connections = edges_[img_id][line_id];
  const Line2d &l1 = all_lines_2d_[img_id][line_id];
  if (l1.length() <= config_.min_length_2d)
    return;
  const CameraView &view1 = imagecols_->camview(img_id);
  size_t n_conns = connections.size();
  std::vector<std::vector<TriTuple>> results(n_conns);

#pragma omp parallel for
  for (size_t conn_id = 0; conn_id < n_conns; ++conn_id) {
    int ng_img_id = connections[conn_id].first;
    int ng_line_id = connections[conn_id].second;
    const Line2d &l2 = all_lines_2d_[ng_img_id][ng_line_id];
    if (l2.length() <= config_.min_length_2d)
      continue;
    const CameraView &view2 = imagecols_->camview(ng_img_id);

    // Step 1.1: many points: connect points
    // Step 1.2: one point: point-based triangulation
    if (use_pointsfm_ && (!config_.disable_many_points_triangulation ||
                          !config_.disable_one_point_triangulation)) {
      std::map<int, Point2d> points1;
      std::set<int> set1;
      std::map<int, std::pair<V2D, V2D>> points_info;
      for (const int &point_id :
           all_bpt2ds_->at(img_id).neighbor_points(line_id)) {
        auto p = all_bpt2ds_->at(img_id).point(point_id);
        set1.insert(p.point3D_id);
        points1.insert(std::make_pair(p.point3D_id, p));
      }
      for (const int &point_id :
           all_bpt2ds_->at(ng_img_id).neighbor_points(ng_line_id)) {
        auto p = all_bpt2ds_->at(ng_img_id).point(point_id);
        if (set1.find(p.point3D_id) != set1.end()) {
          V2D p1 = points1.at(p.point3D_id).p;
          points_info.insert(
              std::make_pair(p.point3D_id, std::pair<V2D, V2D>(p1, p.p)));
        }
      }
      // triangulate points
      std::vector<V3D> points;
      for (auto it = points_info.begin(); it != points_info.end(); ++it) {
        if (sfm_points_.empty()) {
          auto res = triangulate_point(it->second.first, view1,
                                       it->second.second, view2);
          if (res.second)
            points.push_back(res.first);
        } else
          points.push_back(sfm_points_.at(it->first));
      }

      // Step 1.1: many points: connect points
      // points.size() >= 2 -> fit line
      // fit lines with total least square
      if (!config_.disable_many_points_triangulation && points.size() >= 2) {
        V3D center(0.0, 0.0, 0.0);
        for (size_t i = 0; i < points.size(); ++i) {
          center += points[i];
        }
        center /= points.size();
        Eigen::MatrixXd epoints;
        epoints.resize(points.size(), 3);
        for (size_t i = 0; i < points.size(); ++i) {
          epoints.row(i) = points[i] - center;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(epoints, Eigen::ComputeThinV);
        V3D direc = svd.matrixV().col(0).normalized();
        InfiniteLine3d inf_line = InfiniteLine3d(center, direc);
        Line3d line = triangulate_line_with_infinite_line(l1, view1, inf_line);
        if (line.score > 0) {
          double u1 = line.computeUncertainty(view1, config_.var2d);
          double u2 = line.computeUncertainty(view2, config_.var2d);
          line.uncertainty = std::min(u1, u2);
          results[conn_id].push_back(std::make_tuple(
              line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
        }
      }

      // Step 1.2 one point triangulation
      if (!config_.disable_one_point_triangulation && !points.empty()) {
        for (const V3D &p : points) {
          Line3d line =
              triangulate_line_with_one_point(l1, view1, l2, view2, p);
          if (line.score > 0) {
            double u1 = line.computeUncertainty(view1, config_.var2d);
            double u2 = line.computeUncertainty(view2, config_.var2d);
            line.uncertainty = std::min(u1, u2);
            results[conn_id].push_back(std::make_tuple(
                line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
          }
        }
      }
    }

    // Step 2: triangulation with VPs
    if (config_.use_vp && !config_.disable_vp_triangulation) {
      // vp1
      if (vpresults_[img_id].HasVP(line_id)) {
        V3D direc =
            getDirectionFromVP(vpresults_[img_id].GetVP(line_id), view1);
        Line3d line =
            triangulate_line_with_direction(l1, view1, l2, view2, direc);
        if (line.score > 0) {
          double u1 = line.computeUncertainty(view1, config_.var2d);
          double u2 = line.computeUncertainty(view2, config_.var2d);
          line.uncertainty = std::min(u1, u2);
          results[conn_id].push_back(std::make_tuple(
              line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
        }
      }
      // vp2
      if (vpresults_[ng_img_id].HasVP(ng_line_id)) {
        V3D direc =
            getDirectionFromVP(vpresults_[ng_img_id].GetVP(ng_line_id), view1);
        Line3d line =
            triangulate_line_with_direction(l1, view1, l2, view2, direc);
        if (line.score > 0) {
          double u1 = line.computeUncertainty(view1, config_.var2d);
          double u2 = line.computeUncertainty(view2, config_.var2d);
          line.uncertainty = std::min(u1, u2);
          results[conn_id].push_back(std::make_tuple(
              line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
        }
      }
    }

    // Step 3: line triangulation
    if (!config_.disable_algebraic_triangulation) {
      // test degeneracy by ray-plane angles
      V3D n2 = getNormalDirection(l2, view2);
      V3D ray1_start = view1.ray_direction(l1.start);
      double angle_start =
          90 - acos(std::abs(n2.dot(ray1_start))) * 180.0 / M_PI;
      if (angle_start < config_.line_tri_angle_threshold)
        continue;
      V3D ray1_end = view1.ray_direction(l1.end);
      double angle_end = 90 - acos(std::abs(n2.dot(ray1_end))) * 180.0 / M_PI;
      if (angle_end < config_.line_tri_angle_threshold)
        continue;

      // test weak epipolar constraints
      double IoU = compute_epipolar_IoU(l1, view1, l2, view2);
      if (IoU < config_.IoU_threshold)
        continue;

      // triangulation with weak epipolar constraints test
      Line3d line;
      if (!config_.use_endpoints_triangulation)
        line = triangulate_line(l1, view1, l2, view2);
      else
        line = triangulate_line_by_endpoints(l1, view1, l2, view2);
      if (line.sensitivity(view1) > config_.sensitivity_threshold &&
          line.sensitivity(view2) > config_.sensitivity_threshold)
        line.score = -1;
      if (line.score > 0) {
        double u1 = line.computeUncertainty(view1, config_.var2d);
        double u2 = line.computeUncertainty(view2, config_.var2d);
        line.uncertainty = std::min(u1, u2);
        results[conn_id].push_back(
            std::make_tuple(line, -1.0, std::make_pair(ng_img_id, ng_line_id)));
      }
    }
  }
  for (int conn_id = 0; conn_id < n_conns; ++conn_id) {
    auto &result = results[conn_id];
    for (auto it = result.begin(); it != result.end(); ++it) {
      if (ranges_flag_) {
        if (!test_line_inside_ranges(std::get<0>(*it), ranges_))
          continue;
      }
      tris_[img_id][line_id].push_back(*it);
    }
  }
}

} // namespace triangulation

} // namespace limap
