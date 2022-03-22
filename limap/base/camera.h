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

namespace limap {

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

