#pragma once

#include "limap/util/eigen_types.h"
#include <colmap/scene/camera.h>

namespace limap {

// Some augmented functions on top of colmap::Camera
class CameraUtils {
public:
  /**
   * @brief Propagate 2D variance to depth-scaled metric uncertainty.
   * @param cam   COLMAP camera (for focal length).
   * @param depth Projective depth (camera Z) at the point/segment.
   * @param var2d Variance of 2D detection (default 5.0).
   * @return var2d * depth / f, where f is average focal length.
   */
  static double Uncertainty(const colmap::Camera &cam, double depth,
                            double var2d = 5.0);

private:
  // @brief Average focal length f from (fx, fy) if available, else single f.
  static double AverageFocalLength(const colmap::Camera &cam);
};

} // namespace limap
