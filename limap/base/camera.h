#ifndef LIMAP_BASE_CAMERA_H
#define LIMAP_BASE_CAMERA_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <fstream>

namespace py = pybind11;

#include "util/types.h"
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
    Camera(int model_id, const std::vector<double>& params, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(const std::string& model_name, const std::vector<double>& params, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    // initialize with intrinsics
    Camera(M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(int model_id, M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));
    Camera(const std::string& model_name, M3D K, int cam_id=-1, std::pair<int, int> hw=std::make_pair<int, int>(-1, -1));

    void resize(const size_t width, const size_t height) { Rescale(width, height); }
    void set_max_image_dim(const int& val);
    M3D K() const { return CalibrationMatrix(); }
    M3D K_inv() const { return K().inverse(); }
    double h() const { return Height(); }
    double w() const { return Width(); }
    std::vector<double> params() const { return Params(); }
    
    double uncertainty(double depth, double var2d = 5);
};

class CameraPose {
public:
    CameraPose() {}
    CameraPose(V4D qqvec, V3D ttvec): qvec(qqvec), tvec(ttvec) {}
    CameraPose(M3D R, V3D T): tvec(T) { qvec = colmap::RotationMatrixToQuaternion(R); }

    V4D qvec;
    V3D tvec;

    M3D R() const { return colmap::QuaternionToRotationMatrix(qvec); }
    V3D T() const { return tvec; }

    V3D center() const { return -R().transpose() * T(); }
    double projdepth(const V3D& p3d) const;
};

class View {
public:
    View() {}
    View(const Camera& input_cam, const CameraPose& input_pose): cam(input_cam), pose(input_pose) {}

    Camera cam;
    CameraPose pose;
    
    M3D K() const { return cam.K(); }
    M3D K_inv() const { return cam.K_inv(); }
    M3D R() const { return pose.R(); }
    V3D T() const { return pose.T(); }

    V2D projection(const V3D& p3d) const;
    V3D ray_direction(const V2D& p2d) const;
};

class PinholeCamera {
public:
    PinholeCamera() {}
    PinholeCamera(const PinholeCamera& camera);
    PinholeCamera(M3F K_, M3F R_, V3F T_);
    PinholeCamera(M3F K_, M3F R_, V3F T_, const std::pair<int, int>& img_hw): PinholeCamera(K_, R_, T_) {set_hw(img_hw.first, img_hw.second);}
    PinholeCamera(M3D K_, M3D R_, V3D T_); 
    PinholeCamera(M3D K_, M3D R_, V3D T_, const std::pair<int, int>& img_hw): PinholeCamera(K_, R_, T_) {set_hw(img_hw.first, img_hw.second);}
    PinholeCamera(M3F K_, M3F R_, V3F T_, const std::vector<double>& dist_coeffs_); // input distortion coeffs with lengths 5 or 8
    PinholeCamera(M3F K_, M3F R_, V3F T_, const std::vector<double>& dist_coeffs_, const std::pair<int, int>& img_hw): PinholeCamera(K_, R_, T_, dist_coeffs_) {set_hw(img_hw.first, img_hw.second);} 
    PinholeCamera(M3F K_, M3F R_, V3F T_, const std::pair<int, int>& img_hw, const std::vector<double>& dist_coeffs_): PinholeCamera(K_, R_, T_, dist_coeffs_, img_hw) {} 
    PinholeCamera(M3D K_, M3D R_, V3D T_, const std::vector<double>& dist_coeffs_); // input distortion coeffs with lengths 5 or 8
    PinholeCamera(M3D K_, M3D R_, V3D T_, const std::vector<double>& dist_coeffs_, const std::pair<int, int>& img_hw): PinholeCamera(K_, R_, T_, dist_coeffs_) {set_hw(img_hw.first, img_hw.second);} 
    PinholeCamera(M3D K_, M3D R_, V3D T_, const std::pair<int, int>& img_hw, const std::vector<double>& dist_coeffs_): PinholeCamera(K_, R_, T_, dist_coeffs_, img_hw) {} 

    void set_hw(const int& height_, const int& width_) { height = height_; width = width_; }
    void setZeroDistCoeffs() { std::fill(dist_coeffs.begin(), dist_coeffs.end(), 0.0); }
    void set_max_image_dim(const int& val);
    bool checkUndistorted() const;
    double computeUncertainty(const double& depth, const double var2d=5) const;

    V3D GetPosition() const;
    V3D GetCameraRay(const V2D& p2d) const;
    double projdepth(const V3D& p3d) const;
    V2D projection(const V3D& p3d) const;

    void Read(const std::string& filename);
    void Write(const std::string& filename) const;

    M3D K, K_inv;
    M3D R;
    V3D T;
    std::vector<double> dist_coeffs = std::vector<double>(8, 0);
    int height = -1;
    int width = -1;
};

// interchanging with colmap camera (without extrinsics in its definition)
colmap::Camera cam_ours2colmap(const PinholeCamera& camera);

PinholeCamera cam_colmap2ours(const colmap::Camera& camera);

// used for optimization
class MinimalPinholeCamera {
public:
    MinimalPinholeCamera() {}
    MinimalPinholeCamera(const PinholeCamera& camera);
    PinholeCamera GetCamera() const;

    V4D kvec; // [f1, f2, c1, c2]
    V4D qvec;
    V3D tvec;
    int height, width;
};

// interchanging between PinholeCamera and MinimalPinholeCamera
MinimalPinholeCamera cam2minimalcam(const PinholeCamera& camera);

PinholeCamera minimalcam2cam(const MinimalPinholeCamera& camera);

} // namespace limap

#endif

