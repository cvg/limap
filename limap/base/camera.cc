#include "base/camera.h"

namespace limap {

void Camera::SetModelId(const int model_id) {
    colmap::Camera::SetModelId(model_id);
    initialized.resize(NumParams());
    std::fill(initialized.begin(), initialized.end(), false);
}

void Camera::SetModelIdFromName(const std::string& model_name) {
    colmap::Camera::SetModelIdFromName(model_name);
    initialized.resize(NumParams());
    std::fill(initialized.begin(), initialized.end(), false);
}

void Camera::SetParams(const std::vector<double>& params) {
    THROW_CHECK_EQ(params.size(), NumParams());
    colmap::Camera::SetParams(params);
    std::fill(initialized.begin(), initialized.end(), true);
}

void Camera::InitializeParams(const double focal_length, const int width, const int height) {
    InitializeWithId(ModelId(), focal_length, width, height);
    std::fill(initialized.begin(), initialized.end(), true);
}

bool Camera::IsInitialized() const {
    for (auto it = initialized.begin(); it != initialized.end(); ++it) {
        if (!*it)
            return false;
    }
    return true;
}

Camera::Camera(const colmap::Camera& cam) {
    SetCameraId(cam.CameraId());
    SetModelId(cam.ModelId());
    SetParams(cam.Params());
    SetHeight(cam.Height());
    SetWidth(cam.Width());
}

// empty camera
Camera::Camera(int model_id, int cam_id, std::pair<int, int> hw) {
    SetModelId(model_id);
    SetCameraId(cam_id);
    SetHeight(hw.first);
    SetWidth(hw.second);
}

// empty camera
Camera::Camera(const std::string& model_name, int cam_id, std::pair<int, int> hw) {
    SetModelIdFromName(model_name);
    SetCameraId(cam_id);
    SetHeight(hw.first);
    SetWidth(hw.second);
}

Camera::Camera(int model_id, const std::vector<double>& params, int cam_id, std::pair<int, int> hw) {
    SetModelId(model_id);
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
    SetParams(params);
    if (cam_id != -1)
        SetCameraId(cam_id);
    if (hw.first != -1 && hw.second != -1) {
        SetHeight(hw.first);
        SetWidth(hw.second);
    }
}

Camera::Camera(const Camera& cam) {
    SetCameraId(cam.CameraId());
    SetModelId(cam.ModelId());
    SetParams(cam.Params());
    SetHeight(cam.Height());
    SetWidth(cam.Width());
    initialized = cam.initialized;
}

bool Camera::operator ==(const Camera& cam) {
    if (CameraId() != cam.CameraId())
        return false;
    if (ModelId() != cam.ModelId())
        return false;
    if (h() != cam.h())
        return false;
    if (w() != cam.w())
        return false;
    std::vector<double> params = Params();
    std::vector<double> params_cam = cam.Params();
    for (int i = 0; i < params.size(); ++i) {
        if (params[i] != params_cam[i])
            return false;
    }
    return true;
}

void Camera::set_max_image_dim(const int& val) {
    THROW_CHECK_EQ(IsUndistorted(), true);
    THROW_CHECK_GT(val, 0)

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

    // set initialized
    ASSIGN_PYDICT_ITEM(dict, initialized, std::vector<bool>);
}

py::dict Camera::as_dict() const {
    py::dict output;
    output["model_id"] = ModelId();
    output["params"] = params();
    output["cam_id"] = CameraId();
    output["height"] = h();
    output["width"] = w();
    output["initialized"] = initialized;
    return output;
}

double CameraPose::projdepth(const V3D& p3d) const {
    V3D p_cam = R() * p3d + T();
    return p_cam(2);
}

CameraPose::CameraPose(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, qvec, V4D);
    qvec = qvec.normalized();
    ASSIGN_PYDICT_ITEM(dict, tvec, V3D);
    ASSIGN_PYDICT_ITEM(dict, initialized, bool);
}

py::dict CameraPose::as_dict() const {
    py::dict output;
    output["qvec"] = qvec;
    output["tvec"] = tvec;
    output["initialized"] = initialized;
    return output;
}

} // namespace limap

