#include "limap/geometry/line_metrics.h"

namespace limap {

double InfinitePerpendicularDistance(const Line3d &l1, const Line3d &l2) {
  // compute the minimum distance between two 3D lines
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
  if (det < 1e-12) {
    p = B1 / A11;
    q = 0;
  } else {
    p = (B1 * A22 - B2 * A12) / det;
    q = (A11 * B2 - A21 * B1) / det;
  }
  double dist = (C0 + Cp * p + Cq * q).norm();
  return dist;
}

template <>
double ComputeDistance<Line2d>(const Line2d &l1, const Line2d &l2,
                               const LineMetricType &type) {
  switch (type) {
  case LineMetricType::ANGULAR:
    return ComputeAngle<Line2d>(l1, l2);
  case LineMetricType::ENDPOINTS:
    return EndpointsDistance<Line2d>(l1, l2);
  case LineMetricType::MIDPOINT:
    return MidpointDistance<Line2d>(l1, l2);
  case LineMetricType::MIDPOINT_PERPENDICULAR:
    return MidpointPerpendicularDistance<Line2d>(l1, l2);
  case LineMetricType::OVERLAP:
    return ComputeOverlap<Line2d>(l1, l2);
  case LineMetricType::BIOVERLAP:
    return ComputeBioverlap<Line2d>(l1, l2);
  case LineMetricType::PERPENDICULAR_ONEWAY:
    return EndpointsPerpendicularDistanceOneway<Line2d>(l1, l2);
  case LineMetricType::PERPENDICULAR:
    return EndpointsPerpendicularDistance<Line2d>(l1, l2);
  case LineMetricType::INNERSEG:
    return InnersegDistance<Line2d>(l1, l2);
  default:
    throw std::runtime_error("Distance type not supported for Line2d.");
  }
}

template <>
double ComputeDistance<Line3d>(const Line3d &l1, const Line3d &l2,
                               const LineMetricType &type) {
  switch (type) {
  case LineMetricType::ANGULAR:
    return ComputeAngle<Line3d>(l1, l2);
  case LineMetricType::ENDPOINTS:
    return EndpointsDistance<Line3d>(l1, l2);
  case LineMetricType::MIDPOINT:
    return MidpointDistance<Line3d>(l1, l2);
  case LineMetricType::MIDPOINT_PERPENDICULAR:
    return MidpointPerpendicularDistance<Line3d>(l1, l2);
  case LineMetricType::OVERLAP:
    return ComputeOverlap<Line3d>(l1, l2);
  case LineMetricType::BIOVERLAP:
    return ComputeBioverlap<Line3d>(l1, l2);
  case LineMetricType::PERPENDICULAR_ONEWAY:
    return EndpointsPerpendicularDistanceOneway<Line3d>(l1, l2);
  case LineMetricType::PERPENDICULAR:
    return EndpointsPerpendicularDistance<Line3d>(l1, l2);
  case LineMetricType::INNERSEG:
    return InnersegDistance<Line3d>(l1, l2);
  default:
    throw std::runtime_error("Distance type not supported for Line3d.");
  }
}

} // namespace limap
