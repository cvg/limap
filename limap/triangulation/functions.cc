#include "triangulation/functions.h"
#include "solvers/triangulation/triangulate_line_with_one_point.h"

namespace limap {

namespace triangulation {

bool test_line_inside_ranges(const Line3d &line,
                             const std::pair<V3D, V3D> &ranges) {
  // test start
  if (line.start[0] < ranges.first[0] || (line.start[0] > ranges.second[0]))
    return false;
  if (line.start[1] < ranges.first[1] || (line.start[1] > ranges.second[1]))
    return false;
  if (line.start[2] < ranges.first[2] || (line.start[2] > ranges.second[2]))
    return false;

  // test end
  if (line.end[0] < ranges.first[0] || (line.end[0] > ranges.second[0]))
    return false;
  if (line.end[1] < ranges.first[1] || (line.end[1] > ranges.second[1]))
    return false;
  if (line.end[2] < ranges.first[2] || (line.end[2] > ranges.second[2]))
    return false;
  return true;
}

V3D getNormalDirection(const Line2d &l, const CameraView &view) {
  const M3D K_inv = view.K_inv();
  const M3D R = view.R();
  V3D c_start = R.transpose() * K_inv * V3D(l.start[0], l.start[1], 1);
  V3D c_end = R.transpose() * K_inv * V3D(l.end[0], l.end[1], 1);
  V3D n = c_start.cross(c_end);
  return n.normalized();
}

V3D getDirectionFromVP(const V3D &vp, const CameraView &view) {
  const M3D K_inv = view.K_inv();
  const M3D R = view.R();
  V3D direc = R.transpose() * K_inv * vp;
  return direc.normalized();
}

M3D compute_essential_matrix(const CameraView &view1, const CameraView &view2) {
  const M3D R1 = view1.R();
  const V3D T1 = view1.T();
  const M3D R2 = view2.R();
  const V3D T2 = view2.T();

  // compose relative pose
  M3D relR = R2 * R1.transpose();
  V3D relT = T2 - relR * T1;

  // essential matrix and fundamental matrix
  M3D tskew;
  tskew(0, 0) = 0.0;
  tskew(0, 1) = -relT[2];
  tskew(0, 2) = relT[1];
  tskew(1, 0) = relT[2];
  tskew(1, 1) = 0.0;
  tskew(1, 2) = -relT[0];
  tskew(2, 0) = -relT[1];
  tskew(2, 1) = relT[0];
  tskew(2, 2) = 0.0;
  M3D E = tskew * relR;
  return E;
}

M3D compute_fundamental_matrix(const CameraView &view1,
                               const CameraView &view2) {
  M3D E = compute_essential_matrix(view1, view2);
  M3D F = view2.K_inv().transpose() * E * view1.K_inv();
  return F;
}

double compute_epipolar_IoU(const Line2d &l1, const CameraView &view1,
                            const Line2d &l2, const CameraView &view2) {
  // fundamental matrix
  M3D F = compute_fundamental_matrix(view1, view2);

  // epipolar lines
  V3D coor_l2 = l2.coords();
  V3D coor_epline_start = (F * V3D(l1.start[0], l1.start[1], 1)).normalized();
  V3D homo_c_start = coor_l2.cross(coor_epline_start);
  V2D c_start = dehomogeneous(homo_c_start);
  V3D coor_epline_end = (F * V3D(l1.end[0], l1.end[1], 1)).normalized();
  V3D homo_c_end = coor_l2.cross(coor_epline_end);
  V2D c_end = dehomogeneous(homo_c_end);

  // compute IoU
  double c1 = (c_start - l2.start).dot(l2.direction()) / l2.length();
  double c2 = (c_end - l2.start).dot(l2.direction()) / l2.length();
  if (c1 > c2)
    std::swap(c1, c2);
  double IoU = (std::min(c2, 1.0) - std::max(c1, 0.0)) /
               (std::max(c2, 1.0) - std::min(c1, 0.0));
  return IoU;
}

std::pair<V3D, bool> triangulate_point(const V2D &p1, const CameraView &view1,
                                       const V2D &p2, const CameraView &view2) {
  V3D C1 = view1.pose.center();
  V3D C2 = view2.pose.center();
  V3D n1e = view1.ray_direction(p1);
  V3D n2e = view2.ray_direction(p2);
  M2D A;
  A << n1e.dot(n1e), -n1e.dot(n2e), -n2e.dot(n1e), n2e.dot(n2e);
  V2D b;
  b(0) = n1e.dot(C2 - C1);
  b(1) = n2e.dot(C1 - C2);
  V2D res = A.ldlt().solve(b);
  V3D point = 0.5 * (n1e * res[0] + C1 + n2e * res[1] + C2);
  // cheirality test
  if (view1.pose.projdepth(point) < EPS || view2.pose.projdepth(point) < EPS)
    return std::make_pair(V3D(0., 0., 0.), false);
  return std::make_pair(point, true);
}

M3D point_triangulation_covariance(const V2D &p1, const CameraView &view1,
                                   const V2D &p2, const CameraView &view2,
                                   const Eigen::Matrix4d &covariance) {
  V3D C1 = view1.pose.center();
  V3D C2 = view2.pose.center();
  V3D n1e = view1.ray_direction(p1);
  V3D n2e = view2.ray_direction(p2);

  std::pair<V3D, V3D> n1e_grad = view1.ray_direction_gradient(p1);
  std::pair<V3D, V3D> n2e_grad = view2.ray_direction_gradient(p2);

  M2D A;
  A << n1e.dot(n1e), -n1e.dot(n2e), -n2e.dot(n1e), n2e.dot(n2e);
  M2D A_inv = A.inverse();
  V2D b;
  b(0) = n1e.dot(C2 - C1);
  b(1) = n2e.dot(C1 - C2);
  V2D res = A.ldlt().solve(b);

  std::vector<M2D> dAsdx(4, M2D::Zero());
  dAsdx[0](0, 0) = 2 * n1e_grad.first.dot(n1e);
  dAsdx[0](0, 1) = dAsdx[0](1, 0) = (-1) * n1e_grad.first.dot(n2e);
  dAsdx[1](0, 0) = 2 * n1e_grad.second.dot(n1e);
  dAsdx[1](0, 1) = dAsdx[1](1, 0) = (-1) * n1e_grad.second.dot(n2e);
  dAsdx[2](1, 1) = 2 * n2e_grad.first.dot(n2e);
  dAsdx[2](0, 1) = dAsdx[2](1, 0) = (-1) * n2e_grad.first.dot(n1e);
  dAsdx[3](1, 1) = 2 * n2e_grad.second.dot(n2e);
  dAsdx[3](0, 1) = dAsdx[3](1, 0) = (-1) * n2e_grad.second.dot(n1e);

  Eigen::MatrixXd dbdx = Eigen::MatrixXd::Zero(2, 4);
  dbdx(0, 0) = n1e_grad.first.dot(C2 - C1);
  dbdx(0, 1) = n1e_grad.second.dot(C2 - C1);
  dbdx(1, 2) = n2e_grad.first.dot(C1 - C2);
  dbdx(1, 3) = n2e_grad.second.dot(C1 - C2);

  Eigen::MatrixXd J_res = Eigen::MatrixXd::Zero(2, 4);
  for (size_t i = 0; i < 4; ++i) {
    J_res.col(i) = (-1) * (A_inv * dAsdx[i] * A_inv) * b + A_inv * dbdx.col(i);
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 4);
  J.col(0) =
      0.5 * (n1e * J_res(0, 0) + n2e * J_res(1, 0) + n1e_grad.first * res[0]);
  J.col(1) =
      0.5 * (n1e * J_res(0, 1) + n2e * J_res(1, 1) + n1e_grad.second * res[0]);
  J.col(2) =
      0.5 * (n1e * J_res(0, 2) + n2e * J_res(1, 2) + n2e_grad.first * res[1]);
  J.col(3) =
      0.5 * (n1e * J_res(0, 3) + n2e * J_res(1, 3) + n2e_grad.second * res[1]);
  return J * covariance * J.transpose();
}

// Triangulating endpoints for triangulation
Line3d triangulate_line_by_endpoints(const Line2d &l1, const CameraView &view1,
                                     const Line2d &l2,
                                     const CameraView &view2) {
  // start point
  auto res_start = triangulate_point(l1.start, view1, l2.start, view2);
  if (!res_start.second)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  V3D pstart = res_start.first;
  // end point
  auto res_end = triangulate_point(l1.end, view1, l2.end, view2);
  if (!res_end.second)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  V3D pend = res_end.first;

  // construct line
  double z_start = view1.pose.projdepth(pstart);
  double z_end = view1.pose.projdepth(pend);
  return Line3d(pstart, pend, 1.0, z_start, z_end);
}

// Asymmetric perspective to (view1, l1)
// Triangulation by plane intersection
std::pair<Line3d, bool> line_triangulation(const Line2d &l1,
                                           const CameraView &view1,
                                           const Line2d &l2,
                                           const CameraView &view2) {
  V3D c1_start = view1.ray_direction(l1.start);
  V3D c1_end = view1.ray_direction(l1.end);
  V3D c2_start = view2.ray_direction(l2.start);
  V3D c2_end = view2.ray_direction(l2.end);
  V3D B = view2.pose.center() - view1.pose.center();

  // start point
  M3D A_start;
  A_start << c1_start, -c2_start, -c2_end;
  auto res_start = A_start.inverse() * B;
  V3D l3d_start = c1_start * res_start[0] + view1.pose.center();
  double z_start = view1.pose.projdepth(l3d_start);

  // end point
  M3D A_end;
  A_end << c1_end, -c2_start, -c2_end;
  auto res_end = A_end.inverse() * B;
  V3D l3d_end = c1_end * res_end[0] + view1.pose.center();
  double z_end = view1.pose.projdepth(l3d_end);

  // check
  if (z_start < EPS || z_end < EPS)
    return std::make_pair(Line3d(), false);
  double d21, d22;
  d21 = view2.pose.projdepth(l3d_start);
  d22 = view2.pose.projdepth(l3d_end);
  if (d21 < EPS || d22 < EPS)
    return std::make_pair(Line3d(), false);

  // check nan
  if (std::isnan(l3d_start[0]) || std::isnan(l3d_end[0]))
    return std::make_pair(Line3d(), false);

  Line3d line = Line3d(l3d_start, l3d_end, 1.0, z_start, z_end);
  return std::make_pair(line, true);
}

M6D line_triangulation_covariance(const Line2d &l1, const CameraView &view1,
                                  const Line2d &l2, const CameraView &view2,
                                  const M8D &covariance) {
  // compute matrix form again
  V3D c1_start = view1.ray_direction(l1.start);
  V3D c1_end = view1.ray_direction(l1.end);
  V3D c2_start = view2.ray_direction(l2.start);
  V3D c2_end = view2.ray_direction(l2.end);
  V3D B = view2.pose.center() - view1.pose.center();
  M3D A_start;
  A_start << c1_start, -c2_start, -c2_end;
  auto res_start = A_start.inverse() * B;
  M3D A_end;
  A_end << c1_end, -c2_start, -c2_end;
  auto res_end = A_end.inverse() * B;

  // compute first-order gradient
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, 8);

  // gradient
  std::pair<V3D, V3D> c1_start_grad = view1.ray_direction_gradient(l1.start);
  std::pair<V3D, V3D> c1_end_grad = view1.ray_direction_gradient(l1.end);
  std::pair<V3D, V3D> c2_start_grad = view2.ray_direction_gradient(l2.start);
  std::pair<V3D, V3D> c2_end_grad = view2.ray_direction_gradient(l2.end);

  // start point
  M3D A_start_inv = A_start.inverse();
  std::vector<M3D> dAsdx(8, M3D::Zero());
  dAsdx[0].col(0) = c1_start_grad.first;
  dAsdx[1].col(0) = c1_start_grad.second;
  dAsdx[4].col(1) = -c2_start_grad.first;
  dAsdx[5].col(1) = -c2_start_grad.second;
  dAsdx[6].col(1) = -c2_end_grad.first;
  dAsdx[7].col(1) = -c2_end_grad.second;
  for (size_t i = 0; i < 8; ++i) {
    double dzdx = (-1) * (A_start_inv * dAsdx[i] * A_start_inv).row(0) * B;
    J.block<3, 1>(0, i) = c1_start * dzdx;
  }
  J.block<3, 1>(0, 0) += c1_start_grad.first * res_start[0];
  J.block<3, 1>(0, 1) += c1_start_grad.second * res_start[0];
  // end point
  M3D A_end_inv = A_end.inverse();
  std::vector<M3D> dAedx(8, M3D::Zero());
  dAedx[2].col(0) = c1_end_grad.first;
  dAedx[3].col(0) = c1_end_grad.second;
  dAedx[4].col(1) = -c2_start_grad.first;
  dAedx[5].col(1) = -c2_start_grad.second;
  dAedx[6].col(1) = -c2_end_grad.first;
  dAedx[7].col(1) = -c2_end_grad.second;
  for (size_t i = 0; i < 8; ++i) {
    double dzdx = (-1) * (A_end_inv * dAedx[i] * A_end_inv).row(0) * B;
    J.block<3, 1>(3, i) = c1_end * dzdx;
  }
  J.block<3, 1>(3, 2) += c1_end_grad.first * res_end[0];
  J.block<3, 1>(3, 3) += c1_end_grad.second * res_end[0];
  // covariance propagation
  return J * covariance * J.transpose();
}

// Algebraic line triangulation
Line3d triangulate_line(const Line2d &l1, const CameraView &view1,
                        const Line2d &l2, const CameraView &view2) {
  // triangulate line
  auto res = line_triangulation(l1, view1, l2, view2);
  if (!res.second)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  Line3d line = res.first;
  return line;
}

// unproject endpoints with known infinite line
Line3d triangulate_line_with_infinite_line(const Line2d &l1,
                                           const CameraView &view1,
                                           const InfiniteLine3d &inf_line) {
  InfiniteLine3d ray1_start =
      InfiniteLine3d(view1.pose.center(), view1.ray_direction(l1.start));
  V3D pstart = inf_line.project_to_infinite_line(ray1_start);
  double z_start = view1.pose.projdepth(pstart);
  InfiniteLine3d ray1_end =
      InfiniteLine3d(view1.pose.center(), view1.ray_direction(l1.end));
  V3D pend = inf_line.project_to_infinite_line(ray1_end);
  double z_end = view1.pose.projdepth(pend);

  if (z_start < EPS || z_end < EPS) // cheriality test
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  return Line3d(pstart, pend, 1.0, z_start, z_end);
}

// Asymmetric perspective to (view1, l1)
// Triangulation with a known point
Line3d triangulate_line_with_one_point(const Line2d &l1,
                                       const CameraView &view1,
                                       const Line2d &l2,
                                       const CameraView &view2,
                                       const V3D &point) {
  // project point onto plane 1
  V3D n1 = getNormalDirection(l1, view1);
  V3D C1 = view1.pose.center();
  V3D p = point - n1.dot(point - C1) * n1;
  V3D v1s = view1.ray_direction(l1.start);
  V3D v1e = view1.ray_direction(l1.end);

  // get plane 2
  V3D n2 = getNormalDirection(l2, view2);
  double alpha = (-1) * n2.dot(view2.pose.center());

  // construct transformation
  M3D R;
  R.col(0) = v1s;
  R.col(1) = (v1e - v1s.dot(v1e) * v1s).normalized();
  R.col(2) = (R.col(0).cross(R.col(1))).normalized();
  V3D t = C1;

  // apply inverse transform
  V3D v1_transformed = V3D(1., 0., 0.);
  V3D v2_transformed = R.transpose() * v1e;
  V3D p_transformed = R.transpose() * (p - C1);
  V3D n2_transformed = R.transpose() * n2;
  double alpha_transformed = alpha + n2.dot(t);

  // generate input and apply the quartic polynomial solver
  V4D input_plane = V4D(n2_transformed[0], n2_transformed[1], n2_transformed[2],
                        alpha_transformed);
  V2D input_p = V2D(p_transformed[0], p_transformed[1]);
  V2D input_v1 = V2D(v1_transformed[0], v1_transformed[1]).normalized();
  V2D input_v2 = V2D(v2_transformed[0], v2_transformed[1]).normalized();
  std::pair<double, double> res =
      solvers::triangulation::triangulate_line_with_one_point(
          input_plane, input_p, input_v1, input_v2);
  if (res.first < 0 || res.second < 0) {
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  }
  V2D lstart_2d = input_v1 * res.first;
  V3D lstart = R * V3D(lstart_2d[0], lstart_2d[1], 0.0) + t;
  V2D lend_2d = input_v2 * res.second;
  V3D lend = R * V3D(lend_2d[0], lend_2d[1], 0.0) + t;

  // cheirality check
  double z_start = view1.pose.projdepth(lstart);
  double z_end = view1.pose.projdepth(lend);
  if (z_start < EPS || z_end < EPS) {
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  }
  double d21, d22;
  d21 = view2.pose.projdepth(lstart);
  d22 = view2.pose.projdepth(lend);
  if (d21 < EPS || d22 < EPS) {
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  }
  return Line3d(lstart, lend, 1.0, z_start, z_end);
}

// Asymmetric perspective to (view1, l1)
// Triangulation with known direction
Line3d triangulate_line_with_direction(const Line2d &l1,
                                       const CameraView &view1,
                                       const Line2d &l2,
                                       const CameraView &view2,
                                       const V3D &direction) {
  // Step 1: project direction onto plane 1
  V3D n1 = getNormalDirection(l1, view1);
  V3D direc = direction - (n1.dot(direction)) * n1;
  if (direc.norm() < EPS)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  direc = direc.normalized();

  // Step 2: parameterize on plane 1 (a1s * d1s - a1e * d1e = 0)
  V3D perp_direc = n1.cross(direc);
  V3D v1s = view1.ray_direction(l1.start);
  double a1s = v1s.dot(perp_direc);
  V3D v1e = view1.ray_direction(l1.end);
  double a1e = v1e.dot(perp_direc);
  const double MIN_VALUE = 0.001;
  if (a1s < 0) {
    a1s *= -1;
    a1e *= -1;
  }
  if (a1s < MIN_VALUE || a1e < MIN_VALUE)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);

  // Step 3: min [(c1s * d1s - b)^2 + (c1e * d1e - b)^2]
  V3D C1 = view1.pose.center();
  V3D C2 = view2.pose.center();
  V3D n2 = getNormalDirection(l2, view2);
  double c1s = n2.dot(v1s);
  double c1e = n2.dot(v1e);
  double b = n2.dot(C2 - C1);

  // Optimal solution
  double c1 = c1s;
  double c2 = c1e * a1s / a1e;
  double d1s_num = (c1 + c2) * b;
  double d1s_denom = (c1 * c1 + c2 * c2);
  double d1s = d1s_num / d1s_denom;
  double d1e = d1s * a1s / a1e;

  // check
  V3D lstart = d1s * v1s + C1;
  V3D lend = d1e * v1e + C1;
  double z_start = view1.pose.projdepth(lstart);
  double z_end = view1.pose.projdepth(lend);
  if (z_start < EPS || z_end < EPS)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  double d21, d22;
  d21 = view2.pose.projdepth(lstart);
  d22 = view2.pose.projdepth(lend);
  if (d21 < EPS || d22 < EPS)
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  if (std::isnan(lstart[0]) || std::isnan(lend[0]))
    return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
  return Line3d(lstart, lend, 1.0, z_start, z_end);
}

} // namespace triangulation

} // namespace limap
