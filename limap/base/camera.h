#ifndef LIMAP_BASE_CAMERA_H_
#define LIMAP_BASE_CAMERA_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cmath>
#include <fstream>

namespace py = pybind11;

#include "util/types.h"
#include "_limap/helpers.h"

#include <colmap/base/camera.h>
#include <colmap/base/camera_models.h>
#include <colmap/base/pose.h>

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

class Camera: public colmap::Camera {
public:
    Camera() {}
    Camera(const colmap::Camera& cam);
    Camera(int model_id, const std::vector<double>& params, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(const std::string& model_name, const std::vector<double>& params, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    // initialize with intrinsics
    Camera(M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(int model_id, M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(const std::string& model_name, M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(py::dict dict);
    Camera(const Camera& cam);
    Camera(int model_id, int cam_id, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1)); // empty camera
    Camera(const std::string& model_name, int cam_id, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1)); // empty camera
    bool operator ==(const Camera&);

    py::dict as_dict() const;
    void resize(const size_t width, const size_t height) { Rescale(width, height); }
    void set_max_image_dim(const int& val);
    M3D K() const { return CalibrationMatrix(); }
    M3D K_inv() const { return K().inverse(); }
    int h() const { return Height(); }
    int w() const { return Width(); }
    std::vector<double> params() const { return Params(); }
    
    double uncertainty(double depth, double var2d = 5.0) const;

    // initialized
    void SetModelId(const int model_id); // override
    void SetModelIdFromName(const std::string& model_name); // override
    void SetParams(const std::vector<double>& params); // override
    void InitializeParams(const double focal_length, const int width, const int height);
    std::vector<bool> initialized;
    bool IsInitialized() const;
};

class CameraPose {
public:
    CameraPose(bool initialized = false): initialized(initialized) {}
    CameraPose(V4D qvec, V3D tvec, bool initialized = true): qvec(qvec.normalized()), tvec(tvec), initialized(initialized) {}
    CameraPose(M3D R, V3D T, bool initiallized = true): tvec(T), initialized(initialized) { qvec = colmap::RotationMatrixToQuaternion(R); }
    CameraPose(py::dict dict);
    CameraPose(const CameraPose& campose): qvec(campose.qvec), tvec(campose.tvec), initialized(campose.initialized) {}

    V4D qvec = V4D(1., 0., 0., 0.);
    V3D tvec = V3D::Zero();
    bool initialized = false;

    py::dict as_dict() const;
    M3D R() const { return colmap::QuaternionToRotationMatrix(qvec); }
    V3D T() const { return tvec; }

    V3D center() const { return -R().transpose() * T(); }
    double projdepth(const V3D& p3d) const;
    void SetInitFlag(bool flag) { initialized = flag; }
};

} // namespace limap

#endif

