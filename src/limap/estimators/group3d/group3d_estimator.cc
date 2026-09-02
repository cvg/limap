#include "limap/estimators/group3d/group3d_estimator.h"

#include <colmap/util/logging.h>

#include <PoseLib/types.h>

namespace limap {
namespace estimators {
namespace group3d {

namespace {
poselib::RansacOptions MakeRansacOptions() {
  poselib::RansacOptions options;
  options.max_iterations = 1000;
  return options;
}
} // namespace

std::optional<std::vector<double>>
InitializeGroupParams(GroupType type, const std::vector<V3D> &points,
                      const std::vector<Line3d> &lines,
                      double init_ransac_thresh) {
  auto group_impl = GetGroup(type);
  if (!group_impl) {
    LOG(WARNING)
        << "InitializeGroupParams: Cannot get implementation for group type";
    return std::nullopt;
  }

  double thresh = init_ransac_thresh > 0 ? init_ransac_thresh
                                         : group_impl->GetDefaultThreshold();
  auto ransac_options = MakeRansacOptions();

  switch (type) {
  case GroupType::VP: {
    if (lines.empty()) {
      LOG(WARNING) << "VP: No 3D lines provided";
      return std::nullopt;
    }
    auto result = EstimateVPRobust(lines, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    V3D vp = result.value();
    return std::vector<double>{vp[0], vp[1], vp[2]};
  }
  case GroupType::PLANE: {
    if (points.empty() && lines.empty()) {
      LOG(WARNING) << "Plane: No 3D points or lines provided";
      return std::nullopt;
    }
    auto result = EstimatePlaneRobust(points, lines, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    V4D p = result.value();
    return std::vector<double>{p[0], p[1], p[2], p[3]};
  }
  case GroupType::SPHERE: {
    if (points.size() < 4) {
      LOG(WARNING) << "Sphere: Need at least 4 points, got " << points.size();
      return std::nullopt;
    }
    auto result = EstimateSphereRobust(points, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    const V4D &s = result.value();
    return std::vector<double>{s[0], s[1], s[2], s[3]};
  }
  case GroupType::CYLINDER: {
    if (points.size() < 5) {
      LOG(WARNING) << "Cylinder: Need at least 5 points, got " << points.size();
      return std::nullopt;
    }
    auto result = EstimateCylinderRobust(points, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    return result;
  }
  case GroupType::ELLIPSOID: {
    if (points.size() < 9) {
      LOG(WARNING) << "Ellipsoid: Need at least 9 points, got "
                   << points.size();
      return std::nullopt;
    }
    auto result = EstimateEllipsoidRobust(points, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    return result;
  }
  case GroupType::CUBOID: {
    std::vector<V3D> all_points = points;
    for (const auto &line : lines) {
      all_points.push_back(line.start);
      all_points.push_back(line.end);
    }
    if (all_points.size() < 6) {
      LOG(WARNING) << "Cuboid: Need at least 6 points, got "
                   << all_points.size();
      return std::nullopt;
    }
    auto result = EstimateCuboidRobust(all_points, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    return result;
  }
  case GroupType::CONE: {
    if (points.size() < 6) {
      LOG(WARNING) << "Cone: Need at least 6 points, got " << points.size();
      return std::nullopt;
    }
    auto result = EstimateConeRobust(points, thresh, ransac_options);
    if (!result.has_value())
      return std::nullopt;
    return result;
  }
  case GroupType::INVALID:
    LOG(WARNING) << "InitializeGroupParams: INVALID group type";
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace group3d
} // namespace estimators
} // namespace limap
