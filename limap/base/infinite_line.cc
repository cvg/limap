#include "base/infinite_line.h"
#include "base/pose.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {

InfiniteLine2d::InfiniteLine2d(const V2D &p, const V2D &direc) {
  THROW_CHECK_LT(std::abs(direc.norm() - 1.0), EPS);
  V3D coor;
  coor[0] = direc[1];
  coor[1] = (-1) * direc[0];
  coor[2] = (-1) * direc[1] * p[0] + direc[0] * p[1];
  coords = coor.normalized();
}

InfiniteLine2d::InfiniteLine2d(const Line2d &line) {
  CHECK_GT(line.length(), 0.0);
  coords = line.coords();
}

V2D InfiniteLine2d::point_projection(const V2D &q) const {
  V2D direc = direction();
  InfiniteLine2d inf_line_perp = InfiniteLine2d(q, V2D(direc[1], -direc[0]));
  V3D p_homo = coords.cross(inf_line_perp.coords);
  THROW_CHECK_GT(p_homo(2), EPS);
  return dehomogeneous(p_homo);
}

double InfiniteLine2d::point_distance(const V2D &q) const {
  return (q - point_projection(q)).norm();
}

V2D InfiniteLine2d::point() const { return point_projection(V2D(0., 0.)); }

V2D InfiniteLine2d::direction() const {
  return V2D(coords[1], -coords[0]).normalized();
}

std::pair<V2D, bool> Intersect_InfiniteLine2d(const InfiniteLine2d &l1,
                                              const InfiniteLine2d &l2) {
  V3D coor1 = l1.coords;
  V3D coor2 = l2.coords;
  V3D p_homo = coor1.cross(coor2).normalized();
  V2D p(0, 0);
  if (std::abs(p_homo(2)) < EPS)
    return std::make_pair(p, false);
  else {
    p = dehomogeneous(p_homo);
    return std::make_pair(p, true);
  }
}

InfiniteLine3d::InfiniteLine3d(const V3D &a, const V3D &b, bool use_normal) {
  if (use_normal) {
    THROW_CHECK_LT(std::abs(b.norm() - 1.0), EPS);
    d = b;
    m = a.cross(b);
  } else {
    THROW_CHECK_LT(std::abs(a.norm() - 1.0), EPS);
    d = a;
    m = b;
  }
}

InfiniteLine3d::InfiniteLine3d(const Line3d &line) {
  CHECK_GT(line.length(), 0.0);
  d = line.direction();
  m = line.start.cross(d);
}

V3D InfiniteLine3d::point_projection(const V3D &q) const {
  // Reference: page 4 at
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  V3D m_q = m + d.cross(q);
  return q + d.cross(m_q);
}

double InfiniteLine3d::point_distance(const V3D &q) const {
  return (q - point_projection(q)).norm();
}

V3D InfiniteLine3d::point() const { return point_projection(V3D(0., 0., 0.)); }

V3D InfiniteLine3d::direction() const { return d; }

M4D InfiniteLine3d::matrix() const {
  // Plucker matrix from geometric form
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

InfiniteLine2d InfiniteLine3d::projection(const CameraView &view) const {
  // Projection from Plucker coordinate to 2D homogeneous line coordinate
  // [LINK]
  // https://math.stackexchange.com/questions/1811665/how-to-project-a-3d-line-represented-in-pl%C3%BCcker-coordinates-into-2d-image
  M4D L = matrix();
  Eigen::MatrixXd P = view.matrix();
  M3D l_skew = P * L * P.transpose();
  V3D coords;
  coords(0) = l_skew(2, 1);
  coords(1) = l_skew(0, 2);
  coords(2) = l_skew(1, 0);
  coords = coords.normalized();
  InfiniteLine2d inf_line = InfiniteLine2d(coords);
  return inf_line;
}

V3D InfiniteLine3d::unprojection(const V2D &p2d, const CameraView &view) const {
  // take the closest point along the 3D lines towards the camera ray
  // min |C0 + C1 * t1 + C2 * t2|^2, C0 = p1 - p2
  V3D p1, p2;
  V3D C0, C1, C2;
  p1 = point();
  p2 = view.pose.center();
  C0 = p1 - p2;
  C1 = direction();
  C2 = view.ray_direction(p2d);
  C1 = C1.normalized();
  C2 = C2.normalized();

  double A11, A12, A21, A22, B1, B2;
  A11 = A22 = 1.0;
  A12 = A21 = C1.dot(C2);
  B1 = -C0.dot(C1);
  B2 = -C0.dot(C2);
  double det = A11 * A22 - A12 * A21;
  double t;
  if (det < EPS) // l1 and l2 nearly collinear
  {
    t = B1 / (A11 + EPS);
  } else
    t = (B1 * A22 - B2 * A12) / det;
  return p1 + t * C1;
}

V3D InfiniteLine3d::project_from_infinite_line(
    const InfiniteLine3d &line) const {
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

V3D InfiniteLine3d::project_to_infinite_line(const InfiniteLine3d &line) const {
  return line.project_from_infinite_line(*this);
}

MinimalInfiniteLine3d::MinimalInfiniteLine3d(
    const std::vector<double> &values) {
  THROW_CHECK_EQ(values.size(), 6);
  uvec[0] = values[0];
  uvec[1] = values[1];
  uvec[2] = values[2];
  uvec[3] = values[3];
  wvec[0] = values[4];
  wvec[1] = values[5];
}

MinimalInfiniteLine3d::MinimalInfiniteLine3d(const InfiniteLine3d &inf_line) {
  // [LINK]
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  // [LINK] https://hal.archives-ouvertes.fr/hal-00092589/document

  // get Plucker coordinate
  V3D a = inf_line.d;
  V3D b = inf_line.m;

  // orthonormal representation
  // SO(2)
  double w1, w2;
  w1 = 1.0;
  w2 = b.norm();
  double denom = V2D(w1, w2).norm();
  wvec(0) = w1 / denom;
  wvec(1) = w2 / denom;

  // SO(3)
  M3D Q;
  Q.col(0) = a / a.norm();
  if (b.norm() > EPS) {
    Q.col(1) = b / b.norm();
    V3D axb = a.cross(b);
    Q.col(2) = axb / axb.norm();
  } else {
    int best_index = 0;
    if (std::abs(a(1)) > std::abs(a(0)))
      best_index = 1;
    if (std::abs(a(2)) > std::abs(a(best_index)))
      best_index = 2;
    int i1 = (best_index + 1) % 3;
    int i2 = (best_index + 2) % 3;
    V3D bprime;
    bprime(i1) = 1.0;
    bprime(i2) = 1.0;
    bprime(best_index) =
        -(a(i1) * bprime(i1) + a(i2) * bprime(i2)) / a(best_index);
    Q.col(1) = bprime / bprime.norm();
    V3D axb = a.cross(bprime);
    Q.col(2) = axb / axb.norm();
  }
  uvec = RotationMatrixToQuaternion(Q);
}

InfiniteLine3d MinimalInfiniteLine3d::GetInfiniteLine() const {
  // get plucker coordinate
  M3D Q = QuaternionToRotationMatrix(uvec);
  V3D d = Q.col(0);
  V3D m = std::abs(wvec(1)) / std::abs(wvec(0)) * Q.col(1);
  return InfiniteLine3d(d, m, false);
}

Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d &inf_line,
                                        const std::vector<CameraView> &views,
                                        const std::vector<Line2d> &line2ds,
                                        const int num_outliers) {
  THROW_CHECK_EQ(views.size(), line2ds.size());
  int n_lines = line2ds.size();
  V3D dir = inf_line.direction();
  V3D p_ref = inf_line.point();

  std::vector<double> values;
  for (int i = 0; i < n_lines; ++i) {
    const auto &view = views[i];
    const Line2d &line2d = line2ds[i];
    InfiniteLine2d inf_line2d_proj = inf_line.projection(view);
    // project the two 2D endpoints to the 2d projection of the infinite line
    V2D pstart_2d = inf_line2d_proj.point_projection(line2d.start);
    V3D pstart_3d = inf_line.unprojection(pstart_2d, view);
    double tstart = (pstart_3d - p_ref).dot(dir);
    V2D pend_2d = inf_line2d_proj.point_projection(line2d.end);
    V3D pend_3d = inf_line.unprojection(pend_2d, view);
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
  THROW_CHECK_GT(line3ds.size(), 0);
  int n_lines = line3ds.size();
  V3D dir = inf_line.direction();
  V3D p_ref =
      inf_line.point_projection(line3ds[0].start); // get a point on the line

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
