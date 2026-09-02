#include "limap/geometry/camera_utils.h"

namespace limap {

double CameraUtils::Uncertainty(const colmap::Camera &cam, double depth,
                                double var2d) {
  const double f = AverageFocalLength(cam);
  return var2d * depth / f;
}

double CameraUtils::AverageFocalLength(const colmap::Camera &cam) {
  const auto &idxs = cam.FocalLengthIdxs();
  if (idxs.size() == 1) {
    return cam.FocalLength();
  } else if (idxs.size() == 2) {
    const double fx = cam.FocalLengthX();
    const double fy = cam.FocalLengthY();
    return 0.5 * (fx + fy);
  }
  throw std::runtime_error("CameraUtils::AverageFocalLength: FocalLengthIdxs() "
                           "must have size 1 or 2.");
}

} // namespace limap
