#include "limap/estimators/triangulation/functions.h"
#include "limap/estimators/triangulation/triangulate_line_with_one_point.h"

#include "limap/geometry/image_utils.h"

namespace limap {

namespace estimators {

namespace triangulation {

bool TestLineInsideRanges(const Line3d &line,
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

V3D GetNormalDirection(const Line2d &l, const colmap::Image &image) {
  const M3D K_inv = image.CameraPtr()->CalibrationMatrix().inverse();
  const colmap::Rigid3d world_from_cam = Inverse(image.CamFromWorld());

  // Compute back-projected rays in camera coordinates
  V3D c_start = K_inv * V3D(l.start[0], l.start[1], 1);
  V3D c_end = K_inv * V3D(l.end[0], l.end[1], 1);

  // Transform rays into world coordinates
  V3D w_start = world_from_cam.rotation() * c_start;
  V3D w_end = world_from_cam.rotation() * c_end;

  // Compute normal
  V3D n = w_start.cross(w_end);
  return n.normalized();
}

V3D GetDirectionFromVP(const V3D &vp, const colmap::Image &image) {
  const M3D K_inv = image.CameraPtr()->CalibrationMatrix().inverse();
  const colmap::Rigid3d world_from_cam = Inverse(image.CamFromWorld());

  V3D direc = world_from_cam.rotation() * K_inv * vp;
  return direc.normalized();
}

M3D ComputeEssentialMatrix(const colmap::Rigid3d &cam1_from_world,
                           const colmap::Rigid3d &cam2_from_world) {
  // Extract world->camera rotations/translations
  const M3D R1 = cam1_from_world.rotation().toRotationMatrix();
  const V3D t1 = cam1_from_world.translation();
  const M3D R2 = cam2_from_world.rotation().toRotationMatrix();
  const V3D t2 = cam2_from_world.translation();

  // Relative pose from cam1 to cam2: x2 = relR * x1 + relT
  const M3D relR = R2 * R1.transpose();
  const V3D relT = t2 - relR * t1;

  // Skew-symmetric matrix of relT
  M3D tskew;
  tskew << 0.0, -relT.z(), relT.y(), relT.z(), 0.0, -relT.x(), -relT.y(),
      relT.x(), 0.0;

  // Essential matrix
  return tskew * relR;
}

M3D ComputeFundamentalMatrix(const colmap::Image &image1,
                             const colmap::Image &image2) {
  M3D E = ComputeEssentialMatrix(image1.CamFromWorld(), image2.CamFromWorld());
  M3D F = image2.CameraPtr()->CalibrationMatrix().inverse().transpose() * E *
          image1.CameraPtr()->CalibrationMatrix().inverse();
  return F;
}

double ComputeEpipolarIoU(const Line2d &l1, const colmap::Image &image1,
                          const Line2d &l2, const colmap::Image &image2) {
  // fundamental matrix
  M3D F = ComputeFundamentalMatrix(image1, image2);

  // epipolar lines
  V3D coor_l2 = l2.Coords();
  V3D coor_epline_start = (F * V3D(l1.start[0], l1.start[1], 1)).normalized();
  V3D homo_c_start = coor_l2.cross(coor_epline_start);
  V2D c_start = homo_c_start.hnormalized();
  V3D coor_epline_end = (F * V3D(l1.end[0], l1.end[1], 1)).normalized();
  V3D homo_c_end = coor_l2.cross(coor_epline_end);
  V2D c_end = homo_c_end.hnormalized();

  // compute IoU
  double c1 = (c_start - l2.start).dot(l2.Direction()) / l2.Length();
  double c2 = (c_end - l2.start).dot(l2.Direction()) / l2.Length();
  if (c1 > c2)
    std::swap(c1, c2);
  double IoU = (std::min(c2, 1.0) - std::max(c1, 0.0)) /
               (std::max(c2, 1.0) - std::min(c1, 0.0));
  return IoU;
}

std::optional<V3D> TriangulatePoint(const V2D &p1, const colmap::Image &image1,
                                    const V2D &p2,
                                    const colmap::Image &image2) {
  colmap::Rigid3d cam1_from_world = image1.CamFromWorld();
  colmap::Rigid3d cam2_from_world = image2.CamFromWorld();

  V3D C1 = Inverse(cam1_from_world).translation();
  V3D C2 = Inverse(cam2_from_world).translation();
  V3D n1e = ImageUtils::RayDirection(image1, p1);
  V3D n2e = ImageUtils::RayDirection(image2, p2);
  M2D A;
  A << n1e.dot(n1e), -n1e.dot(n2e), -n2e.dot(n1e), n2e.dot(n2e);
  V2D b;
  b(0) = n1e.dot(C2 - C1);
  b(1) = n2e.dot(C1 - C2);
  V2D res = A.ldlt().solve(b);
  V3D point = 0.5 * (n1e * res[0] + C1 + n2e * res[1] + C2);
  // cheirality test
  if ((cam1_from_world * point).z() <= 0 || (cam2_from_world * point).z() <= 0)
    return std::nullopt;
  return point;
}

M3D PointTriangulationCovariance(const V2D &p1, const colmap::Image &image1,
                                 const V2D &p2, const colmap::Image &image2,
                                 const Eigen::Matrix4d &covariance) {
  V3D C1 = Inverse(image1.CamFromWorld()).translation();
  V3D C2 = Inverse(image2.CamFromWorld()).translation();
  V3D n1e = ImageUtils::RayDirection(image1, p1);
  V3D n2e = ImageUtils::RayDirection(image2, p2);

  std::pair<V3D, V3D> n1e_grad = ImageUtils::RayDirectionGradient(image1, p1);
  std::pair<V3D, V3D> n2e_grad = ImageUtils::RayDirectionGradient(image2, p2);

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
std::optional<Line3d> TriangulateLineByEndpoints(const Line2d &l1,
                                                 const colmap::Image &image1,
                                                 const Line2d &l2,
                                                 const colmap::Image &image2) {
  // start point
  auto res_start = TriangulatePoint(l1.start, image1, l2.start, image2);
  if (!res_start.has_value()) {
    return std::nullopt;
  }
  V3D pstart = *res_start;
  // end point
  auto res_end = TriangulatePoint(l1.end, image1, l2.end, image2);
  if (!res_end.has_value()) {
    return std::nullopt;
  }
  V3D pend = *res_end;

  // construct line
  return Line3d(pstart, pend);
}

// Asymmetric perspective to (image1, l1)
// Triangulation by plane intersection
std::optional<Line3d> TriangulateLine(const Line2d &l1,
                                      const colmap::Image &image1,
                                      const Line2d &l2,
                                      const colmap::Image &image2) {
  V3D c1_start = ImageUtils::RayDirection(image1, l1.start);
  V3D c1_end = ImageUtils::RayDirection(image1, l1.end);
  V3D c2_start = ImageUtils::RayDirection(image2, l2.start);
  V3D c2_end = ImageUtils::RayDirection(image2, l2.end);
  colmap::Rigid3d cam1_from_world = image1.CamFromWorld();
  colmap::Rigid3d cam2_from_world = image2.CamFromWorld();

  V3D p1 = Inverse(cam1_from_world).translation();
  V3D p2 = Inverse(cam2_from_world).translation();
  V3D B = p2 - p1;

  // start point
  M3D A_start;
  A_start << c1_start, -c2_start, -c2_end;
  auto res_start = A_start.inverse() * B;
  V3D l3d_start = c1_start * res_start[0] + p1;
  double z_start = (cam1_from_world * l3d_start).z();

  // end point
  M3D A_end;
  A_end << c1_end, -c2_start, -c2_end;
  auto res_end = A_end.inverse() * B;
  V3D l3d_end = c1_end * res_end[0] + p1;
  double z_end = (cam1_from_world * l3d_end).z();

  // check
  if (z_start <= 0 || z_end <= 0)
    return std::nullopt;
  double d21, d22;
  d21 = (cam2_from_world * l3d_start).z();
  d22 = (cam2_from_world * l3d_end).z();
  if (d21 <= 0 || d22 <= 0) {
    return std::nullopt;
  }

  // check nan
  if (std::isnan(l3d_start[0]) || std::isnan(l3d_end[0])) {
    return std::nullopt;
  }

  Line3d line = Line3d(l3d_start, l3d_end);
  return line;
}

M6D LineTriangulationCovariance(const Line2d &l1, const colmap::Image &image1,
                                const Line2d &l2, const colmap::Image &image2,
                                const M8D &covariance) {
  // compute matrix form again
  V3D c1_start = ImageUtils::RayDirection(image1, l1.start);
  V3D c1_end = ImageUtils::RayDirection(image1, l1.end);
  V3D c2_start = ImageUtils::RayDirection(image2, l2.start);
  V3D c2_end = ImageUtils::RayDirection(image2, l2.end);
  colmap::Rigid3d cam1_from_world = image1.CamFromWorld();
  colmap::Rigid3d cam2_from_world = image2.CamFromWorld();

  V3D p1 = Inverse(cam1_from_world).translation();
  V3D p2 = Inverse(cam2_from_world).translation();
  V3D B = p2 - p1;

  M3D A_start;
  A_start << c1_start, -c2_start, -c2_end;
  auto res_start = A_start.inverse() * B;
  M3D A_end;
  A_end << c1_end, -c2_start, -c2_end;
  auto res_end = A_end.inverse() * B;

  // compute first-order gradient
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, 8);

  // gradient
  std::pair<V3D, V3D> c1_start_grad =
      ImageUtils::RayDirectionGradient(image1, l1.start);
  std::pair<V3D, V3D> c1_end_grad =
      ImageUtils::RayDirectionGradient(image1, l1.end);
  std::pair<V3D, V3D> c2_start_grad =
      ImageUtils::RayDirectionGradient(image2, l2.start);
  std::pair<V3D, V3D> c2_end_grad =
      ImageUtils::RayDirectionGradient(image2, l2.end);

  // start point
  M3D A_start_inv = A_start.inverse();
  std::vector<M3D> dAsdx(8, M3D::Zero());
  dAsdx[0].col(0) = c1_start_grad.first;
  dAsdx[1].col(0) = c1_start_grad.second;
  dAsdx[4].col(1) = -c2_start_grad.first;
  dAsdx[5].col(1) = -c2_start_grad.second;
  dAsdx[6].col(2) = -c2_end_grad.first;
  dAsdx[7].col(2) = -c2_end_grad.second;
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
  dAedx[6].col(2) = -c2_end_grad.first;
  dAedx[7].col(2) = -c2_end_grad.second;
  for (size_t i = 0; i < 8; ++i) {
    double dzdx = (-1) * (A_end_inv * dAedx[i] * A_end_inv).row(0) * B;
    J.block<3, 1>(3, i) = c1_end * dzdx;
  }
  J.block<3, 1>(3, 2) += c1_end_grad.first * res_end[0];
  J.block<3, 1>(3, 3) += c1_end_grad.second * res_end[0];
  // covariance propagation
  return J * covariance * J.transpose();
}

// unproject endpoints with known infinite line
std::optional<Line3d>
TriangulateLineWithInfiniteLine3d(const Line2d &l1, const colmap::Image &image1,
                                  const InfiniteLine3d &inf_line) {
  colmap::Rigid3d cam_from_world = image1.CamFromWorld();
  V3D center = Inverse(cam_from_world).translation();

  InfiniteLine3d ray1_start =
      InfiniteLine3d(center, ImageUtils::RayDirection(image1, l1.start));
  V3D pstart = inf_line.ProjectToInfiniteLine(ray1_start);
  double z_start = (cam_from_world * pstart).z();
  InfiniteLine3d ray1_end =
      InfiniteLine3d(center, ImageUtils::RayDirection(image1, l1.end));
  V3D pend = inf_line.ProjectToInfiniteLine(ray1_end);
  double z_end = (cam_from_world * pend).z();

  if (z_start <= 0 || z_end <= 0) {
    return std::nullopt;
  }
  return Line3d(pstart, pend);
}

// Asymmetric perspective to (image1, l1)
// Triangulation with a known point
std::optional<Line3d> TriangulateLineWithOnePoint(const Line2d &l1,
                                                  const colmap::Image &image1,
                                                  const Line2d &l2,
                                                  const colmap::Image &image2,
                                                  const V3D &point) {
  colmap::Rigid3d cam1_from_world = image1.CamFromWorld();
  colmap::Rigid3d cam2_from_world = image2.CamFromWorld();
  V3D C1 = Inverse(cam1_from_world).translation();
  V3D C2 = Inverse(cam2_from_world).translation();

  // project point onto plane 1
  V3D n1 = GetNormalDirection(l1, image1);
  V3D p = point - n1.dot(point - C1) * n1;
  V3D v1s = ImageUtils::RayDirection(image1, l1.start);
  V3D v1e = ImageUtils::RayDirection(image1, l1.end);

  // get plane 2
  V3D n2 = GetNormalDirection(l2, image2);
  double alpha = (-1) * n2.dot(C2);

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
      TriangulateLineWithOnePoint(input_plane, input_p, input_v1, input_v2);
  if (res.first < 0 || res.second < 0) {
    return std::nullopt;
  }
  V2D lstart_2d = input_v1 * res.first;
  V3D lstart = R * V3D(lstart_2d[0], lstart_2d[1], 0.0) + t;
  V2D lend_2d = input_v2 * res.second;
  V3D lend = R * V3D(lend_2d[0], lend_2d[1], 0.0) + t;

  // cheirality check
  double z_start = (cam1_from_world * lstart).z();
  double z_end = (cam1_from_world * lend).z();
  if (z_start <= 0 || z_end <= 0) {
    return std::nullopt;
  }
  double d21, d22;
  d21 = (cam2_from_world * lstart).z();
  d22 = (cam2_from_world * lend).z();
  if (d21 <= 0 || d22 <= 0) {
    return std::nullopt;
  }
  return Line3d(lstart, lend);
}

// Asymmetric perspective to (image1, l1)
// Triangulation with known direction
std::optional<Line3d> TriangulateLineWithDirection(const Line2d &l1,
                                                   const colmap::Image &image1,
                                                   const Line2d &l2,
                                                   const colmap::Image &image2,
                                                   const V3D &direction) {
  colmap::Rigid3d cam1_from_world = image1.CamFromWorld();
  colmap::Rigid3d cam2_from_world = image2.CamFromWorld();

  // Step 1: project direction onto plane 1
  V3D n1 = GetNormalDirection(l1, image1);
  V3D direc = direction - (n1.dot(direction)) * n1;
  if (direc.norm() < 1e-6) {
    return std::nullopt;
  }
  direc = direc.normalized();

  // Step 2: parameterize on plane 1 (a1s * d1s - a1e * d1e = 0)
  V3D perp_direc = n1.cross(direc);
  V3D v1s = ImageUtils::RayDirection(image1, l1.start);
  double a1s = v1s.dot(perp_direc);
  V3D v1e = ImageUtils::RayDirection(image1, l1.end);
  double a1e = v1e.dot(perp_direc);
  const double MIN_VALUE = 0.001;
  if (a1s < 0) {
    a1s *= -1;
    a1e *= -1;
  }
  if (a1s < MIN_VALUE || a1e < MIN_VALUE) {
    return std::nullopt;
  }

  // Step 3: min [(c1s * d1s - b)^2 + (c1e * d1e - b)^2]
  V3D C1 = Inverse(cam1_from_world).translation();
  V3D C2 = Inverse(cam2_from_world).translation();
  V3D n2 = GetNormalDirection(l2, image2);
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
  double z_start = (cam1_from_world * lstart).z();
  double z_end = (cam1_from_world * lend).z();
  if (z_start <= 0 || z_end <= 0) {
    return std::nullopt;
  }
  double d21 = (cam2_from_world * lstart).z();
  double d22 = (cam2_from_world * lend).z();
  if (d21 <= 0 || d22 <= 0) {
    return std::nullopt;
  }
  if (std::isnan(lstart[0]) || std::isnan(lend[0])) {
    return std::nullopt;
  }
  return Line3d(lstart, lend);
}

} // namespace triangulation

} // namespace estimators

} // namespace limap
