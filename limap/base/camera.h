#ifndef LIMAP_BASE_CAMERA_H_
#define LIMAP_BASE_CAMERA_H_

#include <cmath>
#include <fstream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "_limap/helpers.h"
#include "base/pose.h"
#include "util/types.h"

#include <colmap/scene/camera.h>
#include <colmap/sensor/models.h>

namespace limap {

// colmap camera models:
// (0, SIMPLE_PINHOLE)
// (1, PINHOLE)
// (2, SIMPLE_RADIAL)
// (3, RADIAL)
// (4, OPENCV)
// (5, OPENCV_FISHEYE)
// (6, FULL_OPENCV)
// (7, FOV)
// (8, SIMPLE_RADIAL_FISHEYE)
// (9, RADIAL_FISHEYE)
// (10, THIN_PRISM_FISHEYE)

class Camera : public colmap::Camera {
public:
  Camera() {
    camera_id = -1;
    model_id = colmap::CameraModelId::kPinhole;
    height = -1;
    width = -1;
  } // default

  Camera(const colmap::Camera &cam);
  Camera(int model, const std::vector<double> &params, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1, -1));
  Camera(const std::string &model_name, const std::vector<double> &params,
         int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1, -1));
  // initialize with intrinsics
  Camera(M3D K, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1, -1));
  Camera(int model, M3D K, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1, -1));
  Camera(const std::string &model_name, M3D K, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1, -1));
  Camera(py::dict dict);
  Camera(const Camera &cam);
  Camera &operator=(const Camera &cam);
  Camera(int model, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1,
                                                           -1)); // empty camera
  Camera(const std::string &model_name, int cam_id = -1,
         std::pair<int, int> hw = std::make_pair<int, int>(-1,
                                                           -1)); // empty camera
  bool operator==(const Camera &);

  py::dict as_dict() const;
  void resize(const size_t width, const size_t height) {
    Rescale(width, height);
  }
  void set_max_image_dim(const int &val);
  M3D K() const { return CalibrationMatrix(); }
  M3D K_inv() const { return K().inverse(); }
  int h() const { return height; }
  int w() const { return width; }

  double uncertainty(double depth, double var2d = 5.0) const;

  // initialized
  void SetModelId(int model);
  void SetModelIdFromName(const std::string &model_name);
  void SetParams(const std::vector<double> &params);
  void InitializeParams(double focal_length, int width, int height);
  std::vector<bool> initialized;
  bool IsInitialized() const;
};

class CameraPose {
public:
  CameraPose(bool initialized = false) : initialized(initialized) {}
  CameraPose(V4D qvec, V3D tvec, bool initialized = true)
      : qvec(qvec.normalized()), tvec(tvec), initialized(initialized) {}
  CameraPose(M3D R, V3D T, bool initiallized = true)
      : tvec(T), initialized(initialized) {
    qvec = RotationMatrixToQuaternion(R);
  }
  CameraPose(py::dict dict);
  CameraPose(const CameraPose &campose)
      : qvec(campose.qvec), tvec(campose.tvec),
        initialized(campose.initialized) {}

  V4D qvec = V4D(1., 0., 0., 0.);
  V3D tvec = V3D::Zero();
  bool initialized = false;

  py::dict as_dict() const;
  M3D R() const { return QuaternionToRotationMatrix(qvec); }
  V3D T() const { return tvec; }

  V3D center() const { return -R().transpose() * T(); }
  double projdepth(const V3D &p3d) const;
  void SetInitFlag(bool flag) { initialized = flag; }
};

} // namespace limap

#endif
