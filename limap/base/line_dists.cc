#include "base/line_dists.h"
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>

namespace limap {

double dist_endpoints_perpendicular_scaleinv_line3dpp_oneway(const Line3d &l1,
                                                             const Line3d &l2) {
  assert(l1.depths[0] != -1 && l1.depths[1] != -1);
  assert(l2.depths[0] != -1 && l2.depths[1] != -1);
  std::pair<double, double> dists =
      dists_endpoints_perpendicular_oneway<Line3d>(l1, l2);
  return std::max(dists.first / (l1.depths[0] + EPS),
                  dists.second / (l1.depths[1] + EPS));
}

double dist_endpoints_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                      const Line3d &l2) {
  double dist1 = dist_endpoints_perpendicular_scaleinv_line3dpp_oneway(l1, l2);
  double dist2 = dist_endpoints_perpendicular_scaleinv_line3dpp_oneway(l2, l1);
  return std::max(dist1, dist2);
}

double dist_endpoints_perpendicular_scaleinv_oneway(const Line3d &l1,
                                                    const Line3d &l2) {
  assert(l1.depths[0] != -1 && l1.depths[1] != -1);
  assert(l2.depths[0] != -1 && l2.depths[1] != -1);
  std::pair<double, double> dists =
      dists_endpoints_perpendicular_oneway<Line3d>(l1, l2);
  double dist_start = dists.first;
  double dist_end = dists.second;

  V3D dir_l2 = l2.direction();
  double length_l2 = l2.length();
  double alpha_start = (l1.start - l2.start).dot(dir_l2) / length_l2;
  double depth_start =
      l2.depths[0] + alpha_start * (l2.depths[1] - l2.depths[0]);
  double alpha_end = (l1.end - l2.start).dot(dir_l2) / length_l2;
  double depth_end = l2.depths[0] + alpha_end * (l2.depths[1] - l2.depths[0]);
  double MAX_DIST = std::numeric_limits<double>::max();
  if (alpha_start < 100 * EPS || alpha_end < 100 * EPS)
    return MAX_DIST;
  return std::max(dist_start / depth_start, dist_end / depth_end);
}

double dist_endpoints_perpendicular_scaleinv(const Line3d &l1,
                                             const Line3d &l2) {
  double dist1 = dist_endpoints_perpendicular_scaleinv_oneway(l1, l2);
  double dist2 = dist_endpoints_perpendicular_scaleinv_oneway(l2, l1);
  return std::max(dist1, dist2);
}

double dist_endpoints_scaleinv_oneway(const Line3d &l1, const Line3d &l2) {
  double dist_start = (l1.start - l2.start).norm();
  double dist_end = (l1.end - l2.end).norm();
  return std::max(dist_start / (l1.depths[0] + EPS),
                  dist_end / (l1.depths[1] + EPS));
}

double dist_endpoints_scaleinv(const Line3d &l1, const Line3d &l2) {
  double dist1 = dist_endpoints_scaleinv_oneway(l1, l2);
  double dist2 = dist_endpoints_scaleinv_oneway(l2, l1);
  return std::max(dist1, dist2);
}

double infinite_dist_perpendicular(const Line3d &l1, const Line3d &l2) {
  // compute the minimum distance between two 3D lines
  // min |C0 + Cp * p + Cq * q|^2
  // return (C0 + Cp * p + Cq * q).norm()
  V3D C0 = l1.start - l2.start;
  V3D Cp = l1.end - l1.start;
  V3D Cq = l2.start - l2.end;

  double A11, A12, A21, A22, B1, B2;
  A11 = Cp.dot(Cp);
  A22 = Cq.dot(Cq);
  A12 = A21 = Cp.dot(Cq);
  B1 = -C0.dot(Cp);
  B2 = -C0.dot(Cq);
  double det = A11 * A22 - A12 * A21;
  double p, q;
  if (det < EPS) // l1 and l2 nearly collinear
  {
    p = B1 / (A11 + EPS);
    q = 0;
  } else {
    p = (B1 * A22 - B2 * A12) / det;
    q = (A11 * B2 - A21 * B1) / det;
  }
  double dist = (C0 + Cp * p + Cq * q).norm();
  return dist;
}

double infinite_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                const Line3d &l2) {
  // compute the scale invariance distance from l1 to l2
  // min [(d(p(l1, z), l2)) / z]^2
  // return d(p(l1, z*), l2) / z*
  assert(l1.depths[0] != -1 && l1.depths[1] != -1);
  double z1, z2;
  z1 = l1.depths[0];
  z2 = l1.depths[1];

  // min_z {[|Ck + Cz * z|^2 - |(Ck + Cz * z)^T v|^2] / z^2}
  // -->
  // min_k {|Ck * k + Cz|^2 - |(Ck * k + Cz)^T v|^2}, k = 1 / z
  V3D vec2 = l2.end - l2.start;
  V3D v = vec2 / vec2.norm();
  V3D Ck = l1.start - (l1.end - l1.start) * z1 / (z2 - z1) - l2.start;
  V3D Cz = (l1.end - l1.start) / (z2 - z1);

  // take derivative: Ak + B = 0
  double CkTv = Ck.dot(v);
  double A = Ck.dot(Ck) - pow(CkTv, 2);
  double B = Ck.dot(Cz) - CkTv * Cz.dot(v);
  double k = -B / (A + EPS);

  double distsquared =
      (Ck * k + Cz).squaredNorm() - pow((Ck * k + Cz).dot(v), 2);
  double dist = sqrt(distsquared);
  return dist;
}

double infinite_dist_perpendicular_scaleinv_line3dpp(const Line3d &l1,
                                                     const Line3d &l2) {
  // take the mininum of both directions
  double val1 = infinite_perpendicular_scaleinv_line3dpp(l1, l2);
  double val2 = infinite_perpendicular_scaleinv_line3dpp(l2, l1);
  return std::min(val1, val2);
}

double dist_minpoint_2d_oneway(const Line2d &l1, const Line2d &l2) {
  V2D v1 = l1.direction();
  V2D v2 = l2.direction();

  // get basis
  V2D start_vec = (l2.start - l1.start) - (l2.start - l1.start).dot(v2) * v2;
  if (start_vec.norm() < EPS)
    return 0.0;
  double val = start_vec.norm();
  double beta1 = v1.dot(start_vec.normalized());
  if (beta1 <= 0)
    return val;
  else
    return std::max(0.0, val - beta1 * l1.length());
}

double dist_minpoint_3d_oneway(const Line3d &l1, const Line3d &l2) {
  V3D v1 = l1.direction();
  V3D v2 = l2.direction();

  // get basis
  V3D start_vec = (l2.start - l1.start) - (l2.start - l1.start).dot(v2) * v2;
  if (start_vec.norm() < EPS)
    return 0.0;
  double val = start_vec.norm();
  double beta1 = v1.dot(start_vec.normalized());
  if (beta1 <= 0)
    return val;
  double beta2 = v1.dot(v2);
  double beta3 = sqrt(1 - beta1 * beta1 - beta2 * beta2);

  // min (val - beta1 * x)^2 + (beta3 * x)^2
  double peakloc = (beta1 * val) / (beta1 * beta1 + beta3 * beta3);
  if (peakloc < l1.length())
    return V2D(val - beta1 * peakloc, beta3 * peakloc).norm();
  else
    return V2D(val - beta1 * l1.length(), beta3 * l1.length()).norm();
}

template <>
double dist_minpoint_oneway<Line2d>(const Line2d &l1, const Line2d &l2) {
  return dist_minpoint_2d_oneway(l1, l2);
}

template <>
double dist_minpoint_oneway<Line3d>(const Line3d &l1, const Line3d &l2) {
  return dist_minpoint_3d_oneway(l1, l2);
}

template <>
double compute_distance<Line2d>(const Line2d &l1, const Line2d &l2,
                                const LineDistType &type) {
  switch (type) {
  case LineDistType::ANGULAR:
    return compute_angle<Line2d>(l1, l2);
  case LineDistType::ANGULAR_DIST:
    return dist_angular<Line2d>(l1, l2);
  case LineDistType::ENDPOINTS:
    return dist_endpoints<Line2d>(l1, l2);
  case LineDistType::MIDPOINT:
    return dist_midpoint<Line2d>(l1, l2);
  case LineDistType::MIDPOINT_PERPENDICULAR:
    return dist_midpoint_perpendicular<Line2d>(l1, l2);
  case LineDistType::OVERLAP:
    return compute_overlap<Line2d>(l1, l2);
  case LineDistType::BIOVERLAP:
    return compute_bioverlap<Line2d>(l1, l2);
  case LineDistType::OVERLAP_DIST:
    return dist_overlap<Line2d>(l1, l2);
  case LineDistType::PERPENDICULAR_ONEWAY:
    return dist_endpoints_perpendicular_oneway<Line2d>(l1, l2);
  case LineDistType::PERPENDICULAR:
    return dist_endpoints_perpendicular<Line2d>(l1, l2);
  case LineDistType::INNERSEG:
    return dist_innerseg<Line2d>(l1, l2);
  case LineDistType::PERPENDICULAR_SCALEINV_ONEWAY:
    throw std::runtime_error("Type error. Scale invariance perpendicular "
                             "distance is not supported for Line2d.");
  case LineDistType::PERPENDICULAR_SCALEINV:
    throw std::runtime_error("Type error. Scale invariance perpendicular "
                             "distance is not supported for Line2d.");
  case LineDistType::ENDPOINTS_SCALEINV_ONEWAY:
    throw std::runtime_error(
        "Type error. Scale invariance distance is not supported for Line2d.");
  case LineDistType::ENDPOINTS_SCALEINV:
    throw std::runtime_error(
        "Type error. Scale invariance distance is not supported for Line2d.");
  }
  return -1.0;
}

template <>
double compute_distance<Line3d>(const Line3d &l1, const Line3d &l2,
                                const LineDistType &type) {
  switch (type) {
  case LineDistType::ANGULAR:
    return compute_angle<Line3d>(l1, l2);
  case LineDistType::ANGULAR_DIST:
    return dist_angular<Line3d>(l1, l2);
  case LineDistType::ENDPOINTS:
    return dist_endpoints<Line3d>(l1, l2);
  case LineDistType::MIDPOINT:
    return dist_midpoint<Line3d>(l1, l2);
  case LineDistType::MIDPOINT_PERPENDICULAR:
    return dist_midpoint_perpendicular<Line3d>(l1, l2);
  case LineDistType::OVERLAP:
    return compute_overlap<Line3d>(l1, l2);
  case LineDistType::BIOVERLAP:
    return compute_bioverlap<Line3d>(l1, l2);
  case LineDistType::OVERLAP_DIST:
    return dist_overlap<Line3d>(l1, l2);
  case LineDistType::PERPENDICULAR_ONEWAY:
    return dist_endpoints_perpendicular_oneway<Line3d>(l1, l2);
  case LineDistType::PERPENDICULAR:
    return dist_endpoints_perpendicular<Line3d>(l1, l2);
  case LineDistType::INNERSEG:
    return dist_innerseg<Line3d>(l1, l2);
  case LineDistType::PERPENDICULAR_SCALEINV_LINE3DPP_ONEWAY:
    return dist_endpoints_perpendicular_scaleinv_line3dpp_oneway(l1, l2);
  case LineDistType::PERPENDICULAR_SCALEINV_LINE3DPP:
    return dist_endpoints_perpendicular_scaleinv_line3dpp(l1, l2);
  case LineDistType::PERPENDICULAR_SCALEINV_ONEWAY:
    return dist_endpoints_perpendicular_scaleinv_oneway(l1, l2);
  case LineDistType::PERPENDICULAR_SCALEINV:
    return dist_endpoints_perpendicular_scaleinv(l1, l2);
  case LineDistType::ENDPOINTS_SCALEINV_ONEWAY:
    return dist_endpoints_scaleinv_oneway(l1, l2);
  case LineDistType::ENDPOINTS_SCALEINV:
    return dist_endpoints_scaleinv(l1, l2);
  }
  return -1.0;
}

} // namespace limap
