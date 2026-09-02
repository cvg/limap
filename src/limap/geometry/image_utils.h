#pragma once

#include "limap/util/eigen_types.h"
#include <colmap/scene/image.h>

namespace limap {

// Some augmented functions on top of colmap::Image
class ImageUtils {
public:
  /**
   * @brief World-space unit ray direction through image point p2d.
   * @param image  colmap::Image with pose (CamFromWorld) and camera pointer.
   * @param p2d    Pixel coordinates (x, y).
   * @return Unit direction in WORLD coordinates.
   */
  static V3D RayDirection(const colmap::Image &image, const V2D &p2d);

  /**
   * @brief Gradients of the world-space unit ray wrt pixel x and y.
   *
   * If d = normalize(v), then ∂d/∂x = ((I - d dᵀ) / ||v||) * ∂v/∂x.
   * Here v = R_wc * K_inv * [x, y, 1]ᵀ, so:
   *   ∂v/∂x = R_wc * K_inv * [1, 0, 0]ᵀ,
   *   ∂v/∂y = R_wc * K_inv * [0, 1, 0]ᵀ.
   *
   * @return pair {∂d/∂x, ∂d/∂y}, both 3D vectors in WORLD coordinates.
   */
  static std::pair<V3D, V3D> RayDirectionGradient(const colmap::Image &image,
                                                  const V2D &p2d);

  static Eigen::Matrix3x4d ProjectionMatrix(const colmap::Image &image);

private:
  static inline Eigen::Matrix3d
  WorldFromCameraRotation(const colmap::Image &image) {
    // T_wc = (CamFromWorld)^(-1);  R_wc is its rotation
    const colmap::Rigid3d T_wc = Inverse(image.CamFromWorld());
    return T_wc.rotation().toRotationMatrix();
  }

  static inline const colmap::Camera &
  RequireCamera(const colmap::Image &image) {
    THROW_CHECK(image.HasCameraPtr());
    return *image.CameraPtr();
  }
};

} // namespace limap
