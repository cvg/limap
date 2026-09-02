#include "limap/geometry/line3d.h"
#include "limap/geometry/camera_utils.h"
#include "limap/geometry/image_utils.h"
#include "limap/geometry/inf_line3d.h"
#include <cmath>

namespace limap {

Line3d::Line3d(V3D start_, V3D end_, double uncertainty_) {
  start = start_;
  end = end_;
  uncertainty = uncertainty_;
}

Line3d::Line3d(const Eigen::MatrixXd &seg) {
  THROW_CHECK_EQ(seg.rows(), 2);
  THROW_CHECK_EQ(seg.cols(), 3);
  start = V3D(seg(0, 0), seg(0, 1), seg(0, 2));
  end = V3D(seg(1, 0), seg(1, 1), seg(1, 2));
}

void Line3d::SetUncertainty(double val) { uncertainty = val; }

double Line3d::Length() const {
  // Euclidean length between endpoints
  return (start - end).norm();
}

V3D Line3d::Midpoint() const {
  // Average of endpoints
  return 0.5 * (start + end);
}

V3D Line3d::Direction() const {
  // Normalized direction vector from start to end
  return (end - start).normalized();
}

V3D Line3d::PointProjection(const V3D &p) const {
  // Project onto the supporting line, then clamp to the segment
  const V3D dir = Direction();
  const double proj = (p - start).dot(dir);

  if (proj <= 0.0) {
    return start; // before start
  }

  const double len = Length();
  if (proj >= len) {
    return end; // beyond end
  }

  return start + proj * dir; // between start and end
}

double Line3d::PointDistance(const V3D &p) const {
  // Euclidean distance from a point to the segment
  const V3D p_proj = PointProjection(p);
  return (p - p_proj).norm();
}

Eigen::MatrixXd Line3d::AsArray() const {
  // Return endpoints as a 2x3 array
  Eigen::MatrixXd arr(2, 3);
  arr(0, 0) = start[0];
  arr(0, 1) = start[1];
  arr(0, 2) = start[2];
  arr(1, 0) = end[0];
  arr(1, 1) = end[1];
  arr(1, 2) = end[2];
  return arr;
}

std::optional<Line2d> Line3d::Projection(const colmap::Image &image) const {
  // Ensure that the corresponding camera is undistorted
  if (image.CameraPtr()->model_id != colmap::CameraModelId::kSimplePinhole &&
      image.CameraPtr()->model_id != colmap::CameraModelId::kPinhole) {
    LOG(FATAL) << "Error! The camera needs to be undistorted.";
  }

  // Project 3D endpoints into the image
  Line2d line2d;
  auto pstart_image = image.CamFromWorld() * start;
  line2d.start =
      (image.CameraPtr()->CalibrationMatrix() * pstart_image).hnormalized();
  auto pend_image = image.CamFromWorld() * end;
  line2d.end =
      (image.CameraPtr()->CalibrationMatrix() * pend_image).hnormalized();

  // If both endpoints are behind the camera, return std::nullopt
  if (pstart_image.z() <= 0 && pend_image.z() <= 0) {
    return std::nullopt;
  }
  return line2d;
}

Line2d Line3d::AlgProjection(const colmap::Image &image) const {
  // Ensure that the corresponding camera is undistorted
  if (image.CameraPtr()->model_id != colmap::CameraModelId::kSimplePinhole &&
      image.CameraPtr()->model_id != colmap::CameraModelId::kPinhole) {
    LOG(FATAL) << "Error! The camera needs to be undistorted.";
  }

  // Project 3D endpoints into the image
  Line2d line2d;
  auto pstart_image = image.CamFromWorld() * start;
  line2d.start =
      (image.CameraPtr()->CalibrationMatrix() * pstart_image).hnormalized();
  auto pend_image = image.CamFromWorld() * end;
  line2d.end =
      (image.CameraPtr()->CalibrationMatrix() * pend_image).hnormalized();
  return line2d;
}

double Line3d::Sensitivity(const colmap::Image &image) const {
  // Angle between line direction and the camera ray through the 2D midpoint.
  //  - If parallel (cos≈1), angle≈0 → high sensitivity (collapsing).
  //  - If perpendicular (cos≈0), angle≈90 → low sensitivity (good).
  const Line2d line2d = AlgProjection(image);
  const V3D dir3d = ImageUtils::RayDirection(image, line2d.Midpoint());
  const double cos_val = std::abs(Direction().dot(dir3d));
  const double angle_deg = std::acos(cos_val) * 180.0 / M_PI;
  const double sensitivity = 90.0 - angle_deg;
  return sensitivity;
}

double Line3d::ComputeUncertainty(const colmap::Image &image,
                                  double var2d) const {
  // Use average projective depth and camera model to estimate uncertainty
  const colmap::Rigid3d &T_cw = image.CamFromWorld();

  const Eigen::Vector3d Xc1 = T_cw * start; // camera-frame point
  const Eigen::Vector3d Xc2 = T_cw * end;

  const double d1 = Xc1.z();
  const double d2 = Xc2.z();
  const double d = 0.5 * (d1 + d2);

  THROW_CHECK(image.HasCameraPtr());
  const double u = CameraUtils::Uncertainty(*image.CameraPtr(), d, var2d);
  return u;
}

std::vector<Line3d>
GetLine3dVectorFromArray(const std::vector<Eigen::MatrixXd> &segs3d) {
  std::vector<Line3d> lines;
  for (size_t seg_id = 0; seg_id < segs3d.size(); ++seg_id) {
    const Eigen::MatrixXd &seg = segs3d[seg_id];
    lines.push_back(Line3d(seg));
  }
  return lines;
}

std::optional<Line2d> ProjectionLine3d(const Line3d &line3d,
                                       const colmap::Image &image) {
  return line3d.Projection(image);
}

Line3d UnprojectionLine2d(const Line2d &line2d, const colmap::Image &image,
                          const std::pair<double, double> &depths) {
  Line3d line3d;

  // Precompute once
  THROW_CHECK(image.HasCameraPtr());
  const Eigen::Matrix3d Kinv = image.CameraPtr()->CalibrationMatrix().inverse();

  // Backproject endpoints into the CAMERA frame
  const V3D xs_h = line2d.start.homogeneous();
  const V3D xe_h = line2d.end.homogeneous();
  const V3D Xc_s = Kinv * xs_h * depths.first;  // camera coords
  const V3D Xc_e = Kinv * xe_h * depths.second; // camera coords

  // Transform CAMERA → WORLD with the inverse pose
  const colmap::Rigid3d T_wc = Inverse(image.CamFromWorld());
  line3d.start = T_wc * Xc_s; // world coords
  line3d.end = T_wc * Xc_e;   // world coords
  return line3d;
}

Line3d AggregateLine3dList(const std::vector<Line3d> &lines, int num_outliers) {
  const int n_lines = static_cast<int>(lines.size());
  THROW_CHECK_GE(num_outliers, 0);
  // Clamp num_outliers to keep at least 2 lines for aggregation
  num_outliers = std::min(num_outliers, std::max(0, n_lines - 2));

  // Total least squares on endpoints.
  V3D center = V3D::Zero();
  for (int i = 0; i < n_lines; ++i) {
    center += lines[i].start;
    center += lines[i].end;
  }
  center /= (2.0 * static_cast<double>(n_lines));

  Eigen::MatrixXd endpoints(n_lines * 2, 3);
  for (int i = 0; i < n_lines; ++i) {
    endpoints.row(2 * i) = (lines[i].start - center).transpose();
    endpoints.row(2 * i + 1) = (lines[i].end - center).transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(endpoints, Eigen::ComputeThinV);
  V3D direc = svd.matrixV().col(0).normalized();
  InfiniteLine3d inf_line3d(center, direc);
  Line3d out =
      GetLineSegmentFromInfiniteLine3d(inf_line3d, lines, num_outliers);

  // Uncertainty = min uncertainty in the cluster.
  double min_uncertainty = std::numeric_limits<double>::max();
  for (int i = 0; i < n_lines; ++i) {
    min_uncertainty = std::min(min_uncertainty, lines[i].uncertainty);
  }
  out.uncertainty = min_uncertainty;
  return out;
}

} // namespace limap
