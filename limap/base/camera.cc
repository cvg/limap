#include "base/camera.h"

#include <iomanip>
#include <colmap/base/pose.h>

namespace limap {

PinholeCamera::PinholeCamera(const PinholeCamera& cam) {
    K = cam.K; K_inv = cam.K_inv;
    R = cam.R; T = cam.T;
    dist_coeffs = cam.dist_coeffs;
    height = cam.height; width = cam.width;
}

PinholeCamera::PinholeCamera(M3F K_, M3F R_, V3F T_) { 
    K = K_.cast<double> ();
    R = R_.cast<double> ();
    T = T_.cast<double> ();
    K_inv = K.inverse();
    width = K(0, 2) * 2;
    height = K(1, 2) * 2;
}

PinholeCamera::PinholeCamera(M3D K_, M3D R_, V3D T_): K(K_), R(R_), T(T_) { 
    K_inv = K.inverse(); 
    width = K(0, 2) * 2;
    height = K(1, 2) * 2;
}

PinholeCamera::PinholeCamera(M3F K_, M3F R_, V3F T_, const std::vector<double>& dist_coeffs_) { 
    K = K_.cast<double> ();
    R = R_.cast<double> ();
    T = T_.cast<double> ();
    K_inv = K.inverse();  
    if (dist_coeffs_.size() != 5)
        THROW_CHECK_EQ(dist_coeffs_.size(), 8);
    for (int i = 0; i < dist_coeffs_.size(); ++i) { dist_coeffs[i] = dist_coeffs_[i]; }
    width = K(0, 2) * 2;
    height = K(1, 2) * 2;
}

PinholeCamera::PinholeCamera(M3D K_, M3D R_, V3D T_, const std::vector<double>& dist_coeffs_): K(K_), R(R_), T(T_) { 
    K_inv = K.inverse();  
    if (dist_coeffs_.size() != 5)
        THROW_CHECK_EQ(dist_coeffs_.size(), 8);
    for (int i = 0; i < dist_coeffs_.size(); ++i) { dist_coeffs[i] = dist_coeffs_[i]; }
    width = K(0, 2) * 2;
    height = K(1, 2) * 2;
}

void PinholeCamera::set_max_image_dim(const int& val) {
    THROW_CHECK_EQ(K(0, 1), 0);
    THROW_CHECK_EQ(K(1, 0), 0);
    THROW_CHECK_EQ(K(2, 0), 0);
    THROW_CHECK_EQ(K(2, 1), 0);
    THROW_CHECK_EQ(checkUndistorted(), true);

    double ratio = double(val) / double(std::max(height, width));
    if (ratio < 1.0) {
        int new_width = int(round(ratio * width));
        int new_height = int(round(ratio * height));
        K(0, 0) = K(0, 0) * double(new_width) / double(width);
        K(0, 2) = K(0, 2) * double(new_width) / double(width);
        K(1, 1) = K(1, 1) * double(new_height) / double(height);
        K(1, 2) = K(1, 2) * double(new_height) / double(height);
        width = new_width;
        height = new_height;
    }
    K_inv = K.inverse();  
}

double PinholeCamera::computeUncertainty(const double& depth, const double var2d) const {
    double f = (K(0, 0) + K(1, 1)) / 2.0;
    double uncertainty = (1.0 * var2d) * depth / f;
    return uncertainty;
}

bool PinholeCamera::checkUndistorted() const {
    for (int i = 0; i < 8; ++i) {
        if (dist_coeffs[i] != 0) 
            return false;
    }
    return true;
}

V3D PinholeCamera::GetPosition() const {
    return -R.transpose() * T;
}

V3D PinholeCamera::GetCameraRay(const V2D& p2d) const {
    return (R.transpose() * K_inv * V3D(p2d(0), p2d(1), 1.0)).normalized();
}

double PinholeCamera::projdepth(const V3D& p3d) const {
    V3D p_homo = K * (R * p3d + T);
    return p_homo(2);
}

V2D PinholeCamera::projection(const V3D& p3d) const {
    V3D p_homo = K * (R * p3d + T);
    V2D p2d;
    p2d(0) = p_homo(0) / p_homo(2);
    p2d(1) = p_homo(1) / p_homo(2);
    return p2d;
}

void PinholeCamera::Write(const std::string& filename) const {
    std::ofstream file;
    file.open(filename.c_str());
    file << std::fixed << std::setprecision(10);
    file << height << " " << width << "\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file << K(i, j) << " ";
        }
        file << "\n";
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file << R(i, j) << " ";
        }
        file << "\n";
    }
    for (int i = 0; i < 3; ++i) {
        file << T(i) << " ";
    }
    file << "\n";
    for (int i = 0; i < 8; ++i) {
        file << dist_coeffs[i] << " ";
    }
    file << "\n";
}

void PinholeCamera::Read(const std::string& filename) {
    std::ifstream file;
    file.open(filename.c_str());
    file >> height >> width;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file >> K(i, j);
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file >> R(i, j);
        }
    }
    for (int i = 0; i < 3; ++i) {
        file >> T(i);
    }
    for (int i = 0; i < 8; ++i) {
        file >> dist_coeffs[i];
    }
}

MinimalPinholeCamera::MinimalPinholeCamera(const PinholeCamera& camera) {
    THROW_CHECK_EQ(camera.checkUndistorted(), true);

    kvec[0] = camera.K(0, 0); kvec[1] = camera.K(1, 1);
    kvec[2] = camera.K(0, 2); kvec[3] = camera.K(1, 2);
    qvec = colmap::RotationMatrixToQuaternion(camera.R);
    tvec = camera.T;
    height = camera.height; width = camera.width;
}

PinholeCamera MinimalPinholeCamera::GetCamera() const {
    M3D K = M3D::Zero();
    K(0, 0) = kvec[0]; K(1, 1) = kvec[1];
    K(0, 2) = kvec[2]; K(1, 2) = kvec[3];
    K(2, 2) = 1.0;
    M3D R = colmap::QuaternionToRotationMatrix(qvec);
    V3D T = tvec;
    PinholeCamera cam = PinholeCamera(K, R, T);
    cam.set_hw(height, width);
    return cam;
}

colmap::Camera cam_ours2colmap(const PinholeCamera& camera) {
    colmap::Camera cam;
    if (camera.checkUndistorted())
        cam.SetModelIdFromName("PINHOLE");
    else if (camera.dist_coeffs[4] == 0.0 && camera.dist_coeffs[5] == 0.0 && camera.dist_coeffs[6] == 0.0 && camera.dist_coeffs[7] == 0.0)
        cam.SetModelIdFromName("OPENCV");
    else
        cam.SetModelIdFromName("FULL_OPENCV");
    cam.SetWidth(camera.width);
    cam.SetHeight(camera.height);
    cam.SetFocalLengthX(camera.K(0, 0));
    cam.SetFocalLengthY(camera.K(1, 1));
    cam.SetPrincipalPointX(camera.K(0, 2));
    cam.SetPrincipalPointY(camera.K(1, 2));
    std::string mname = cam.ModelName();
    if (mname == "PINHOLE") {
        // do nothing
    }
    else if (mname == "OPENCV") {
        cam.Params(4) = camera.dist_coeffs[0];
        cam.Params(5) = camera.dist_coeffs[1];
        cam.Params(6) = camera.dist_coeffs[2];
        cam.Params(7) = camera.dist_coeffs[3];
    }
    else { // mname == "FULL_OPENCV"
        cam.Params(4) = camera.dist_coeffs[0];
        cam.Params(5) = camera.dist_coeffs[1];
        cam.Params(6) = camera.dist_coeffs[2];
        cam.Params(7) = camera.dist_coeffs[3];
        cam.Params(8) = camera.dist_coeffs[4];
        cam.Params(9) = camera.dist_coeffs[5];
        cam.Params(10) = camera.dist_coeffs[6];
        cam.Params(11) = camera.dist_coeffs[7];
    } 
    return cam;
}

PinholeCamera cam_colmap2ours(const colmap::Camera& camera) {
    PinholeCamera cam;
    cam.K = camera.CalibrationMatrix();
    cam.height = camera.Height();
    cam.width = camera.Width();
    std::string mname = camera.ModelName();
    const auto& params = camera.Params();
    if (mname == "SIMPLE_PINHOLE" || mname == "PINHOLE") {
        // do nothing
    }
    else if (mname == "SIMPLE_RADIAL") {
        cam.dist_coeffs[0] = params[3];
    }
    else if (mname == "RADIAL") {
        cam.dist_coeffs[0] = params[3];
        cam.dist_coeffs[1] = params[4];
    }
    else if (mname == "OPENCV") {
        cam.dist_coeffs[0] = params[4];
        cam.dist_coeffs[1] = params[5];
        cam.dist_coeffs[2] = params[6];
        cam.dist_coeffs[3] = params[7];
    }
    else if (mname == "FULL_OPENCV") {
        // only consider the first three radial distortion parameters
        cam.dist_coeffs[0] = params[4];
        cam.dist_coeffs[1] = params[5];
        cam.dist_coeffs[2] = params[6];
        cam.dist_coeffs[3] = params[7];
        cam.dist_coeffs[4] = params[8];
        cam.dist_coeffs[5] = params[9];
        cam.dist_coeffs[6] = params[10];
        cam.dist_coeffs[7] = params[11];
    }
    return cam;
}

MinimalPinholeCamera cam2minimalcam(const PinholeCamera& camera) {
    MinimalPinholeCamera cam = MinimalPinholeCamera(camera);
    return cam;
}

PinholeCamera minimalcam2cam(const MinimalPinholeCamera& camera) {
    return camera.GetCamera();
} 

} // namespace limap

