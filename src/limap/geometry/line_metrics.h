#pragma once

#include <cmath>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"

namespace limap {

enum class LineMetricType {
  ANGULAR = 0,
  ENDPOINTS,
  MIDPOINT,
  MIDPOINT_PERPENDICULAR,
  OVERLAP,
  BIOVERLAP,
  PERPENDICULAR_ONEWAY,
  PERPENDICULAR,
  INNERSEG
};

template <typename LineType>
double ComputeDistance(const LineType &l1, const LineType &l2,
                       const LineMetricType &type);

template <typename LineType>
Eigen::MatrixXd ComputePairwiseDistance(const std::vector<LineType> lines,
                                        const LineMetricType &type) {
  size_t n = lines.size();
  Eigen::MatrixXd D(n, n);
  for (int i = 0; i < n; ++i) {
    D(i, i) = 0;
    for (int j = i + 1; j < n; ++j) {
      double dist = ComputeDistance<LineType>(lines[i], lines[j], type);
      D(i, j) = D(j, i) = dist;
    }
  }
  return D;
}

template <typename LineType>
double Cosine(const LineType &l1, const LineType &l2) {
  return std::abs(l1.Direction().dot(l2.Direction()));
}

template <typename LineType>
double ComputeAngle(const LineType &l1, const LineType &l2) {
  double cosVal = Cosine<LineType>(l1, l2);
  return acos(cosVal) * 180.0 / M_PI;
}

template <typename LineType>
double MidpointDistance(const LineType &l1, const LineType &l2) {
  return (l1.Midpoint() - l2.Midpoint()).norm();
}

template <typename LineType>
double EndpointsDistance(const LineType &l1, const LineType &l2) {
  double d1 = (l1.start - l2.start).norm() + (l1.end - l2.end).norm();
  double d2 = (l1.start - l2.end).norm() + (l1.end - l2.start).norm();
  return std::min(d1, d2);
}

template <typename LineType>
double MidpointPerpendicularDistance(const LineType &l1, const LineType &l2) {
  auto mid1 = l1.Midpoint();
  auto mid2 = l2.Midpoint();
  auto v1 = l1.Direction();
  auto v2 = l2.Direction();

  double d12Squared =
      (mid1 - l2.start).squaredNorm() - pow((mid1 - l2.start).dot(v2), 2);
  double d12 = sqrt(std::max(d12Squared, 0.0));
  double d21Squared =
      (mid2 - l1.start).squaredNorm() - pow((mid2 - l1.start).dot(v1), 2);
  double d21 = sqrt(std::max(d21Squared, 0.0));

  double dist = 0.5 * (d12 + d21);
  return dist;
}

template <typename LineType>
std::pair<double, double>
EndpointsPerpendicularDistancesOneway(const LineType &l1, const LineType &l2) {
  auto v2 = l2.Direction();

  auto disps = l1.start - l2.start;
  double d12sSquared = disps.squaredNorm() - pow(disps.dot(v2), 2);
  double d12s = sqrt(std::max(d12sSquared, 0.0));
  auto dispe = l1.end - l2.start;
  double d12eSquared = dispe.squaredNorm() - pow(dispe.dot(v2), 2);
  double d12e = sqrt(std::max(d12eSquared, 0.0));
  return std::make_pair(d12s, d12e);
}

template <typename LineType>
double EndpointsPerpendicularDistanceOneway(const LineType &l1,
                                            const LineType &l2) {
  std::pair<double, double> dists =
      EndpointsPerpendicularDistancesOneway<LineType>(l1, l2);
  return std::max(dists.first, dists.second);
}

template <typename LineType>
std::vector<double> EndpointsPerpendicularDistances(const LineType &l1,
                                                    const LineType &l2) {
  auto dists12 = EndpointsPerpendicularDistancesOneway(l1, l2);
  auto dists21 = EndpointsPerpendicularDistancesOneway(l2, l1);
  return {dists12.first, dists12.second, dists21.first, dists21.second};
}

template <typename LineType>
double EndpointsPerpendicularDistance(const LineType &l1, const LineType &l2) {
  std::vector<double> dists = EndpointsPerpendicularDistances<LineType>(l1, l2);
  return *std::max_element(dists.begin(), dists.end());
}

template <typename LineType>
bool GetInnerseg(const LineType &l1, const LineType &l2, LineType &innerseg) {
  auto l1Dir = l1.Direction();
  double denom = (l2.end - l2.start).dot(l1Dir);
  double numeStart = (l1.start - l2.start).dot(l1Dir);
  if (std::abs(denom) < 1e-8) {
    return false;
  }
  double t1 = numeStart / denom;
  double numeEnd = (l1.end - l2.start).dot(l1Dir);
  double t2 = numeEnd / denom;
  if (t1 > t2) {
    std::swap(t1, t2);
  }
  if (t1 >= 1.0 || t2 <= 0.0) {
    return false;
  }
  innerseg.start = l2.start + (l2.end - l2.start) * std::max(t1, 0.0);
  innerseg.end = l2.start + (l2.end - l2.start) * std::min(t2, 1.0);
  return true;
}

template <typename LineType>
double InnersegDistance(const LineType &l1, const LineType &l2) {
  double MAX_DIST = std::numeric_limits<double>::max();
  LineType l1Innerseg, l2Innerseg;
  if (!GetInnerseg<LineType>(l2, l1, l1Innerseg))
    return MAX_DIST;
  if (!GetInnerseg<LineType>(l1, l2, l2Innerseg))
    return MAX_DIST;
  return EndpointsPerpendicularDistance<LineType>(l1Innerseg, l2Innerseg);
}

template <typename LineType>
double ComputeOverlap(const LineType &l1, const LineType &l2) {
  double len = l2.Length();
  auto v = l2.Direction();
  double p1 = (l1.start - l2.start).dot(v) / len;
  double p2 = (l1.end - l2.start).dot(v) / len;
  if (p1 > p2)
    std::swap(p1, p2);
  double val_i = std::min(p2, 1.0) - std::max(p1, 0.0);
  return val_i;
}

template <typename LineType>
double ComputeBioverlap(const LineType &l1, const LineType &l2) {
  double val1 = ComputeOverlap<LineType>(l1, l2);
  double val2 = ComputeOverlap<LineType>(l2, l1);
  return std::max(val1, val2);
}

template <typename LineType>
double OverlapDistance(const LineType &l1, const LineType &l2) {
  return 1 - ComputeBioverlap<LineType>(l1, l2);
}

double InfinitePerpendicularDistance(const Line3d &l1, const Line3d &l2);

} // namespace limap
