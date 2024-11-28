#ifndef LIMAP_BASE_CAMERA_MODELS_H_
#define LIMAP_BASE_CAMERA_MODELS_H_

#include <colmap/sensor/models.h>

namespace limap {

// modified from COLMAP
// [Link] https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
#ifndef LIMAP_UNDISTORTED_CAMERA_MODEL_CASES
#define LIMAP_UNDISTORTED_CAMERA_MODEL_CASES                                   \
  CAMERA_MODEL_CASE(colmap::SimplePinholeCameraModel)                          \
  CAMERA_MODEL_CASE(colmap::PinholeCameraModel)
#endif

#ifndef LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#define LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES                            \
  LIMAP_UNDISTORTED_CAMERA_MODEL_CASES                                         \
  default:                                                                     \
    LIMAP_CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION                                \
    break;
#endif

#define LIMAP_CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION                            \
  throw std::domain_error("Camera model does not exist");

// Get the 4-dimensional kvec [fx, fy, cx, cy] from the following colmap camera
// model colmap camera models: (0, SIMPLE_PINHOLE) (1, PINHOLE)
template <typename T>
void ParamsToKvec(const colmap::CameraModelId model_id, const T *params,
                  T *kvec) {
  if (model_id == colmap::CameraModelId::kSimplePinhole) { // SIMPLE_PINHOLE
    kvec[0] = params[0];
    kvec[1] = params[0];
    kvec[2] = params[1];
    kvec[3] = params[2];
  } else if (model_id == colmap::CameraModelId::kPinhole) { // PINHOLE
    kvec[0] = params[0];
    kvec[1] = params[1];
    kvec[2] = params[2];
    kvec[3] = params[3];
  } else
    throw std::runtime_error(
        "Error! Limap optimization does not support non-pinhole models.");
}

} // namespace limap

#endif
