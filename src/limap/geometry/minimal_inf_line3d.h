#pragma once

#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <ceres/sphere_manifold.h>

#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

namespace limap {

// Minimal Plücker coordinate for optimization (continuous 6D form)
class MinimalInfiniteLine3d {
public:
  MinimalInfiniteLine3d() = default;
  MinimalInfiniteLine3d(const Line3d &line)
      : MinimalInfiniteLine3d(InfiniteLine3d(line)) {}
  explicit MinimalInfiniteLine3d(const InfiniteLine3d &inf_line);
  explicit MinimalInfiniteLine3d(const std::vector<double> &values);

  InfiniteLine3d GetInfiniteLine() const; // recover full Plücker line

  // Access SO(3) and SO(2) components directly as Eigen maps
  Eigen::Map<const Eigen::Quaterniond> uvec() const {
    return Eigen::Map<const Eigen::Quaterniond>(data.data());
  }
  Eigen::Map<Eigen::Quaterniond> uvec() {
    return Eigen::Map<Eigen::Quaterniond>(data.data());
  }

  Eigen::Map<const Eigen::Vector2d> wvec() const {
    return Eigen::Map<const Eigen::Vector2d>(data.data() + 4);
  }
  Eigen::Map<Eigen::Vector2d> wvec() {
    return Eigen::Map<Eigen::Vector2d>(data.data() + 4);
  }

  V6D data; // [uvec (x,y,z,w), wvec(0), wvec(1)] continuous parameter block
};

// Product manifold for SO(3) × SO(2)
using MinimalInfiniteLine3dManifold =
    ceres::ProductManifold<ceres::EigenQuaternionManifold,
                           ceres::SphereManifold<2>>;

} // namespace limap
