#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/image_utils.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {

InfiniteLine3d::InfiniteLine3d(const V3D &a, const V3D &b, bool use_normal) {
#define THROW_CHECK_NEAR(val, target, tol)                                     \
  THROW_CHECK_LT(std::abs((val) - (target)), (tol))

  if (use_normal) {
    THROW_CHECK_NEAR(b.norm(), 1.0, 1e-8);
    d = b;
    m = a.cross(b);
  } else {
    THROW_CHECK_NEAR(a.norm(), 1.0, 1e-8);
    d = a;
    m = b;
  }

#undef THROW_CHECK_NEAR
}

InfiniteLine3d::InfiniteLine3d(const Line3d &line) {
  CHECK_GT(line.Length(), 0.0);
  d = line.Direction();
  m = line.start.cross(d);
}

V3D InfiniteLine3d::PointProjection(const V3D &q) const {
  // Reference: page 4 at
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  V3D m_q = m + d.cross(q);
  return q + d.cross(m_q);
}

double InfiniteLine3d::PointDistance(const V3D &q) const {
  return (q - PointProjection(q)).norm();
}

V3D InfiniteLine3d::Point() const { return PointProjection(V3D(0., 0., 0.)); }

V3D InfiniteLine3d::Direction() const { return d; }

M4D InfiniteLine3d::Matrix() const {
  // Plücker matrix from geometric form
  // [LINK] https://en.wikipedia.org/wiki/Pl%C3%BCcker_matrix
  M4D L = M4D::Zero();
  L(0, 3) = d[0];
  L(3, 0) = -d[0];
  L(1, 3) = d[1];
  L(3, 1) = -d[1];
  L(2, 3) = d[2];
  L(3, 2) = -d[2];
  L(1, 2) = -m[0];
  L(2, 1) = m[0];
  L(0, 2) = m[1];
  L(2, 0) = -m[1];
  L(0, 1) = -m[2];
  L(1, 0) = m[2];
  return L;
}

InfiniteLine2d InfiniteLine3d::Projection(const colmap::Image &image) const {
  // Projection from Plücker coordinate to 2D homogeneous line coordinate
  // [LINK]
  // https://math.stackexchange.com/questions/1811665/how-to-project-a-3d-line-represented-in-pl%C3%BCcker-coordinates-into-2d-image
  M4D L = Matrix();
  Eigen::MatrixXd P = ImageUtils::ProjectionMatrix(image);
  M3D l_skew = P * L * P.transpose();
  V3D coords;
  coords(0) = l_skew(2, 1);
  coords(1) = l_skew(0, 2);
  coords(2) = l_skew(1, 0);
  coords = coords.normalized();
  InfiniteLine2d inf_line = InfiniteLine2d(coords);
  return inf_line;
}

V3D InfiniteLine3d::Unprojection(const V2D &p2d,
                                 const colmap::Image &image) const {
  // take the closest point along the 3D lines towards the camera ray
  // min |C0 + C1 * t1 + C2 * t2|^2, C0 = p1 - p2
  V3D p1, p2;
  V3D C0, C1, C2;
  p1 = Point();
  p2 = Inverse(image.CamFromWorld()).translation();
  C0 = p1 - p2;
  C1 = Direction();
  C2 = ImageUtils::RayDirection(image, p2d);
  C1 = C1.normalized();
  C2 = C2.normalized();

  double A11, A12, A21, A22, B1, B2;
  A11 = A22 = 1.0;
  A12 = A21 = C1.dot(C2);
  B1 = -C0.dot(C1);
  B2 = -C0.dot(C2);
  double det = A11 * A22 - A12 * A21;
  double t;
  if (det < 1e-12) // l1 and l2 nearly collinear
    t = B1 / A11;
  else
    t = (B1 * A22 - B2 * A12) / det;
  return p1 + t * C1;
}

V3D InfiniteLine3d::ProjectFromInfiniteLine(const InfiniteLine3d &line) const {
  // [LINK] Section 4.2
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  V3D l1, m1, l2, m2;
  l1 = d;
  m1 = m;
  l2 = line.d;
  m2 = line.m;
  V3D p = (-1) * (m1.cross(l2.cross(l1.cross(l2)))) + m2.dot(l1.cross(l2)) * l1;
  p /= (l1.cross(l2)).squaredNorm();
  return p;
}

V3D InfiniteLine3d::ProjectToInfiniteLine(const InfiniteLine3d &line) const {
  return line.ProjectFromInfiniteLine(*this);
}

Line3d GetLineSegmentFromInfiniteLine3d(
    const InfiniteLine3d &inf_line, const std::vector<colmap::Image> &images,
    const std::vector<Line2d> &line2ds, const int num_outliers) {
  THROW_CHECK_EQ(images.size(), line2ds.size());
  THROW_CHECK_GT(images.size(), num_outliers);
  int n_lines = line2ds.size();
  V3D dir = inf_line.Direction();
  V3D p_ref = inf_line.Point();

  std::vector<double> values;
  for (int i = 0; i < n_lines; ++i) {
    const auto &image = images[i];
    const Line2d &line2d = line2ds[i];
    InfiniteLine2d inf_line2d_proj = inf_line.Projection(image);
    // project the two 2D endpoints to the 2d projection of the infinite line
    V2D pstart_2d = inf_line2d_proj.PointProjection(line2d.start);
    V3D pstart_3d = inf_line.Unprojection(pstart_2d, image);
    double tstart = (pstart_3d - p_ref).dot(dir);
    V2D pend_2d = inf_line2d_proj.PointProjection(line2d.end);
    V3D pend_3d = inf_line.Unprojection(pend_2d, image);
    double tend = (pend_3d - p_ref).dot(dir);

    values.push_back(tstart);
    values.push_back(tend);
  }
  std::sort(values.begin(), values.end());
  Line3d final_line;
  final_line.start = p_ref + dir * values[num_outliers];
  final_line.end = p_ref + dir * values[n_lines * 2 - 1 - num_outliers];
  return final_line;
}

Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d &inf_line,
                                        const std::vector<Line3d> &line3ds,
                                        const int num_outliers) {
  THROW_CHECK_GT(line3ds.size(), num_outliers);
  int n_lines = line3ds.size();
  V3D dir = inf_line.Direction();
  V3D p_ref =
      inf_line.PointProjection(line3ds[0].start); // get a point on the line

  std::vector<double> values;
  for (int i = 0; i < n_lines; ++i) {
    const Line3d &line3d = line3ds[i];
    double tstart = (line3d.start - p_ref).dot(dir);
    double tend = (line3d.end - p_ref).dot(dir);
    values.push_back(tstart);
    values.push_back(tend);
  }
  std::sort(values.begin(), values.end());
  Line3d final_line;
  final_line.start = p_ref + dir * values[num_outliers];
  final_line.end = p_ref + dir * values[n_lines * 2 - 1 - num_outliers];
  return final_line;
}

} // namespace limap
