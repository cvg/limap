#ifndef LIMAP_BASE_LINE_DISTS_H_
#define LIMAP_BASE_LINE_DISTS_H_

#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>

#include "base/linebase.h"

namespace limap {

enum class LineDistType {
  ANGULAR = 0,
  ANGULAR_DIST,
  ENDPOINTS,
  MIDPOINT,
  MIDPOINT_PERPENDICULAR,
  OVERLAP,
  BIOVERLAP,
  OVERLAP_DIST,
  PERPENDICULAR_ONEWAY,
  PERPENDICULAR,
  PERPENDICULAR_SCALEINV_LINE3DPP_ONEWAY,
  PERPENDICULAR_SCALEINV_LINE3DPP,
  PERPENDICULAR_SCALEINV_ONEWAY,
  PERPENDICULAR_SCALEINV,
  ENDPOINTS_SCALEINV_ONEWAY,
  ENDPOINTS_SCALEINV,
  INNERSEG
};

template <typename LineType>
double compute_distance(const LineType &l1, const LineType &l2,
                        const LineDistType &type);

template <typename LineType>
Eigen::MatrixXd compute_pairwise_distance(const std::vector<LineType> lines,
                                          const LineDistType &type) {
  size_t n = lines.size();
  Eigen::MatrixXd D(n, n);
  for (int i = 0; i < n; ++i) {
    D(i, i) = 0;
    for (int j = i + 1; j < n; ++j) {
      double dist = compute_distance<LineType>(lines[i], lines[j], type);
      D(i, j) = D(j, i) = dist;
    }
  }
  return D;
}

template <typename LineType>
double cosine(const LineType &l1, const LineType &l2) {
  return std::abs(l1.direction().dot(l2.direction()));
}

template <typename LineType>
double dist_angular(const LineType &l1, const LineType &l2) {
  return 1 - cosine<LineType>(l1, l2);
}

template <typename LineType>
double compute_angle(const LineType &l1, const LineType &l2) {
  double cos_val = cosine<LineType>(l1, l2);
  return acos(cos_val) * 180.0 / M_PI;
}

template <typename LineType>
double dist_midpoint(const LineType &l1, const LineType &l2) {
  return (l1.midpoint() - l2.midpoint()).norm();
}

template <typename LineType>
double dist_endpoints(const LineType &l1, const LineType &l2) {
  double d1 = (l1.start - l2.start).norm() + (l1.end - l2.end).norm();
  double d2 = (l1.start - l2.end).norm() + (l1.end - l2.start).norm();
  return std::min(d1, d2);
}

template <typename LineType>
double dist_midpoint_perpendicular(const LineType &l1, const LineType &l2) {
  auto mid1 = l1.midpoint();
  auto mid2 = l2.midpoint();
  auto v1 = l1.direction();
  auto v2 = l2.direction();

  double d12squared =
      (mid1 - l2.start).squaredNorm() - pow((mid1 - l2.start).dot(v2), 2);
  double d12 = sqrt(std::max(d12squared, 0.0));
  double d21squared =
      (mid2 - l1.start).squaredNorm() - pow((mid2 - l1.start).dot(v1), 2);
  double d21 = sqrt(std::max(d21squared, 0.0));

  double dist = 0.5 * (d12 + d21);
  return dist;
}

template <typename LineType>
std::pair<double, double>
dists_endpoints_perpendicular_oneway(const LineType &l1, const LineType &l2) {
  // l1 endpoints projected to l2
  auto v2 = l2.direction();

  auto disps = l1.start - l2.start;
  double d12s_squared = disps.squaredNorm() - pow(disps.dot(v2), 2);
  double d12s = sqrt(std::max(d12s_squared, 0.0));
  auto dispe = l1.end - l2.start;
  double d12e_squared = dispe.squaredNorm() - pow(dispe.dot(v2), 2);
  double d12e = sqrt(std::max(d12e_squared, 0.0));
  return std::make_pair(d12s, d12e);
}

template <typename LineType>
double dist_endpoints_perpendicular_oneway(const LineType &l1,
                                           const LineType &l2) {
  std::pair<double, double> dists =
      dists_endpoints_perpendicular_oneway<LineType>(l1, l2);
  return std::max(dists.first, dists.second);
}

template <typename LineType>
std::vector<double> dists_endpoints_perpendicular(const LineType &l1,
                                                  const LineType &l2) {
  auto dists12 = dists_endpoints_perpendicular_oneway(l1, l2);
  auto dists21 = dists_endpoints_perpendicular_oneway(l2, l1);
  return {dists12.first, dists12.second, dists21.first, dists21.second};
}

template <typename LineType>
double dist_endpoints_perpendicular(const LineType &l1, const LineType &l2) {
  std::vector<double> dists = dists_endpoints_perpendicular<LineType>(l1, l2);
  return *std::max_element(dists.begin(), dists.end());
}

double dist_endpoints_scaleinv_oneway(const Line3d &l1, const Line3d &l2);
double dist_endpoints_scaleinv(const Line3d &l1, const Line3d &l2);
double dist_endpoints_perpendicular_scaleinv_line3dpp_oneway(const Line3d &l1,
                                                             const Line3d &l2);
double dist_endpoints_perpendicular_scaleinv_oneway(const Line3d &l1,
                                                    const Line3d &l2);
double dist_endpoints_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                      const Line3d &l2);
double dist_endpoints_perpendicular_scaleinv(const Line3d &l1,
                                             const Line3d &l2);

// minimum point on l1 to l2
double dist_minpoint_2d_oneway(const Line2d &l1, const Line2d &l2);
double dist_minpoint_3d_oneway(const Line3d &l1, const Line3d &l2);
template <typename LineType>
double dist_minpoint_oneway(const LineType &l1, const LineType &l2);
template <typename LineType>
double dist_minpoint(const LineType &l1, const LineType &l2) {
  double mindist1 = dist_minpoint_oneway<LineType>(l1, l2);
  double mindist2 = dist_minpoint_oneway<LineType>(l2, l1);
  return std::min(mindist1, mindist2);
}

// innerseg
template <typename LineType>
bool get_innerseg(const LineType &l1, const LineType &l2, LineType &innerseg) {
  // unproject the two endpoints of l1 to l2 and select the inner seg along l2
  // return false if there is no overlap between the unprojection and l2
  auto l1_dir = l1.direction();
  double denom = (l2.end - l2.start).dot(l1_dir);
  double nume_start = (l1.start - l2.start).dot(l1_dir);
  double t1 = nume_start / (denom + EPS);
  double nume_end = (l1.end - l2.start).dot(l1_dir);
  double t2 = nume_end / (denom + EPS);
  if (t1 > t2)
    std::swap(t1, t2);
  if (t1 >= 1.0 || t2 <= 0.0)
    return false;
  innerseg.start = l2.start + (l2.end - l2.start) * std::max(t1, 0.0);
  innerseg.end = l2.start + (l2.end - l2.start) * std::min(t2, 1.0);
  return true;
}

template <typename LineType>
double dist_innerseg(const LineType &l1, const LineType &l2) {
  double MAX_DIST = std::numeric_limits<double>::max();
  LineType l1_innerseg, l2_innerseg;
  if (!get_innerseg<LineType>(l2, l1, l1_innerseg))
    return MAX_DIST;
  if (!get_innerseg<LineType>(l1, l2, l2_innerseg))
    return MAX_DIST;
  // return dist_minpoint<Line3d>(l1_innerseg, l2_innerseg);
  return dist_endpoints_perpendicular<LineType>(l1_innerseg, l2_innerseg);
}

template <typename LineType>
double compute_overlap(const LineType &l1, const LineType &l2) {
  // project l1 onto l2 and compute ratio of intersection
  double len = l2.length();
  auto v = l2.direction();
  double p1 = (l1.start - l2.start).dot(v) / len;
  double p2 = (l1.end - l2.start).dot(v) / len;
  if (p1 > p2)
    std::swap(p1, p2);
  double val_i = std::min(p2, 1.0) - std::max(p1, 0.0);
  return val_i;
}

template <typename LineType>
double compute_bioverlap(const LineType &l1, const LineType &l2) {
  // take the maximum of both directions
  double val1 = compute_overlap<LineType>(l1, l2);
  double val2 = compute_overlap<LineType>(l2, l1);
  return std::max(val1, val2);
}

template <typename LineType>
double dist_overlap(const LineType &l1, const LineType &l2) {
  return 1 - compute_bioverlap<LineType>(l1, l2);
}

double infinite_dist_perpendicular(const Line3d &l1, const Line3d &l2);
double infinite_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                const Line3d &l2);
double infinite_dist_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                     const Line3d &l2);

} // namespace limap

#endif
