#include "base/camera.h"

#include <iomanip>
#include <colmap/base/pose.h>

namespace limap {

Camera::Camera(const colmap::Camera& cam) {
    SetCameraId(cam.CameraId());
    SetModelId(cam.ModelId());
    SetParams(cam.Params());
    SetHeight(cam.Height());
    SetWidth(cam.Width());
}

Camera::Camera(int model_id, const std::vector<double>& params, int cam_id, std::pair<int, int> hw) {
    SetModelId(model_id);
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

Camera::Camera(const std::string& model_name, const std::vector<double>& params, int cam_id, std::pair<int, int> hw) {
    SetModelIdFromName(model_name);
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

Camera::Camera(M3D K, int cam_id, std::pair<int, int> hw) {
    THROW_CHECK_EQ(K(0, 1), 0);
    THROW_CHECK_EQ(K(1, 0), 0);
    THROW_CHECK_EQ(K(2, 0), 0);
    THROW_CHECK_EQ(K(2, 1), 0);
    std::vector<double> params;
    if (K(0, 0) == K(1, 1)) {
        SetModelIdFromName("SIMPLE_PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    else {
        SetModelIdFromName("PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(1, 1));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

Camera::Camera(int model_id, M3D K, int cam_id, std::pair<int, int> hw) {
    THROW_CHECK_EQ(K(0, 1), 0);
    THROW_CHECK_EQ(K(1, 0), 0);
    THROW_CHECK_EQ(K(2, 0), 0);
    THROW_CHECK_EQ(K(2, 1), 0);
    std::vector<double> params;
    if (model_id == 0) {
        // SIMPLE_PINHOLE
        THROW_CHECK_EQ(K(0, 0), K(1, 1));
        SetModelIdFromName("SIMPLE_PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    else if (model_id == 1) {
        // PINHOLE
        SetModelIdFromName("PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(1, 1));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    else
        throw std::runtime_error("model initialized with K should be either SIMPLE_PINHOLE or PINHOLE");
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

Camera::Camera(const std::string& model_name, M3D K, int cam_id, std::pair<int, int> hw) {
    THROW_CHECK_EQ(K(0, 1), 0);
    THROW_CHECK_EQ(K(1, 0), 0);
    THROW_CHECK_EQ(K(2, 0), 0);
    THROW_CHECK_EQ(K(2, 1), 0);
    std::vector<double> params;
    if (model_name == "SIMPLE_PINHOLE") {
        // SIMPLE_PINHOLE
        THROW_CHECK_EQ(K(0, 0), K(1, 1));
        SetModelIdFromName("SIMPLE_PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    else if (model_name == "PINHOLE") {
        // PINHOLE
        SetModelIdFromName("PINHOLE");
        params.push_back(K(0, 0));
        params.push_back(K(1, 1));
        params.push_back(K(0, 2));
        params.push_back(K(1, 2));
    }
    else
        throw std::runtime_error("Error! The model initialized with K should be either SIMPLE_PINHOLE (id=0) or PINHOLE (id=1).");
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

void Camera::set_max_image_dim(const int& val) {
    THROW_CHECK_EQ(IsUndistorted(), true);

    double height = Height();
    double width = Width();
    double ratio = double(val) / double(std::max(height, width));
    if (ratio < 1.0) {
        int new_width = int(round(ratio * width));
        int new_height = int(round(ratio * height));
        Rescale(new_width, new_height);
    }
}

double Camera::uncertainty(double depth, double var2d) const {
    double f = -1.0;
    const std::vector<size_t>& idxs = FocalLengthIdxs();
    if (idxs.size() == 1)
        f = FocalLength();
    else if (idxs.size() == 2) {
        double fx = FocalLengthX();
        double fy = FocalLengthY();
        f = (fx + fy) / 2.0;
    }
    else
        throw std::runtime_error("Error! FocalLengthIdxs() should be either 1 or 2");
    double uncertainty = var2d * depth / f;
    return uncertainty;
}

double CameraPose::projdepth(const V3D& p3d) const {
    V3D p_cam = R() * p3d + T();
    return p_cam(2);
}

V2D CameraView::projection(const V3D& p3d) const {
    V3D p_homo = K() * (R() * p3d + T());
    V2D p2d;
    p2d(0) = p_homo(0) / p_homo(2);
    p2d(1) = p_homo(1) / p_homo(2);
    return p2d;
}

V3D CameraView::ray_direction(const V2D& p2d) const {
    return (R().transpose() * K_inv() * V3D(p2d(0), p2d(1), 1.0)).normalized();
}

MinimalPinholeCamera::MinimalPinholeCamera(const CameraView& view) {
    THROW_CHECK_EQ(view.cam.IsUndistorted(), true);

    M3D K = view.K();
    kvec[0] = K(0, 0); kvec[1] = K(1, 1);
    kvec[2] = K(0, 2); kvec[3] = K(1, 2);
    qvec = view.pose.qvec;
    tvec = view.pose.tvec;
    height = view.cam.h(); width = view.cam.w();
}

CameraView MinimalPinholeCamera::GetCameraView() const {
    M3D K = M3D::Zero();
    K(0, 0) = kvec[0]; K(1, 1) = kvec[1];
    K(0, 2) = kvec[2]; K(1, 2) = kvec[3];
    K(2, 2) = 1.0;
    CameraView view = CameraView(Camera(K), CameraPose(qvec, tvec));
    view.cam.SetHeight(height);
    view.cam.SetWidth(width);
    return view;
}

Camera::Camera(py::dict dict) {
    // load data
    int model_id;
    ASSIGN_PYDICT_ITEM(dict, model_id, int);
    std::vector<double> params;
    ASSIGN_PYDICT_ITEM(dict, params, std::vector<double>);
    int cam_id;
    ASSIGN_PYDICT_ITEM(dict, cam_id, int);
    int height, width;
    ASSIGN_PYDICT_ITEM(dict, height, int);
    ASSIGN_PYDICT_ITEM(dict, width, int);

    // set camera
    SetModelId(model_id);
    THROW_CHECK_EQ(params.size(), NumParams());
    SetParams(params);
    SetCameraId(cam_id);
    SetHeight(height);
    SetWidth(width);
}

py::dict Camera::as_dict() const {
    py::dict output;
    output["model_id"] = ModelId();
    output["params"] = params();
    output["cam_id"] = CameraId();
    output["height"] = h();
    output["width"] = w();
    return output;
}

CameraPose::CameraPose(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, qvec, V4D);
    ASSIGN_PYDICT_ITEM(dict, tvec, V3D);
}

py::dict CameraPose::as_dict() const {
    py::dict output;
    output["qvec"] = qvec;
    output["tvec"] = tvec;
    return output;
}

CameraView::CameraView(py::dict dict) {
    cam = Camera(dict);
    pose = CameraPose(dict);

    // load image name
    std::string image_name;
    ASSIGN_PYDICT_ITEM(dict, image_name, std::string);
    SetImageName(image_name);
}

py::dict CameraView::as_dict() const {
    py::dict output;
    output["model_id"] = cam.ModelId();
    output["params"] = cam.params();
    output["cam_id"] = cam.CameraId();
    output["height"] = cam.h();
    output["width"] = cam.w();
    output["qvec"] = pose.qvec;
    output["tvec"] = pose.tvec;
    output["image_name"] = image_name_;
    return output;
}

MinimalPinholeCamera cam2minimalcam(const CameraView& view) {
    MinimalPinholeCamera cam = MinimalPinholeCamera(view);
    return cam;
}

CameraView minimalcam2cam(const MinimalPinholeCamera& camera) {
    return camera.GetCameraView();
} 

} // namespace limap

