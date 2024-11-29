#include "base/camera.h"
#include <colmap/util/logging.h>

namespace limap {

void Camera::SetModelId(int model) {
  model_id = static_cast<colmap::CameraModelId>(model);
  initialized.resize(params.size());
  std::fill(initialized.begin(), initialized.end(), false);
}

void Camera::SetModelIdFromName(const std::string &model_name) {
  model_id = colmap::CameraModelNameToId(model_name);
  initialized.resize(params.size());
  std::fill(initialized.begin(), initialized.end(), false);
}

void Camera::SetParams(const std::vector<double> &params_input) {
  THROW_CHECK_EQ(params.size(), params_input.size());
  params = params_input;
  std::fill(initialized.begin(), initialized.end(), true);
}

void Camera::InitializeParams(double focal_length, int width, int height) {
  width = width;
  height = height;
  params = colmap::CameraModelInitializeParams(model_id, focal_length, width,
                                               height);
  std::fill(initialized.begin(), initialized.end(), true);
}

bool Camera::IsInitialized() const {
  for (auto it = initialized.begin(); it != initialized.end(); ++it) {
    if (!*it)
      return false;
  }
  return true;
}

Camera::Camera(const colmap::Camera &cam) {
  camera_id = cam.camera_id;
  model_id = cam.model_id;
  params = cam.params;
  height = cam.height;
  width = cam.width;
}

// empty camera
Camera::Camera(int model, int cam_id, std::pair<int, int> hw) {
  model_id = static_cast<colmap::CameraModelId>(model);
  camera_id = cam_id;
  height = hw.first;
  width = hw.second;
}

// empty camera
Camera::Camera(const std::string &model_name, int cam_id,
               std::pair<int, int> hw) {
  model_id = colmap::CameraModelNameToId(model_name);
  camera_id = cam_id;
  height = hw.first;
  width = hw.second;
}

Camera::Camera(int model, const std::vector<double> &params_input, int cam_id,
               std::pair<int, int> hw) {
  model_id = static_cast<colmap::CameraModelId>(model);
  params = params_input;
  if (cam_id != -1)
    camera_id = cam_id;
  if (hw.first != -1 && hw.second != -1) {
    height = hw.first;
    width = hw.second;
  }
}

Camera::Camera(const std::string &model_name,
               const std::vector<double> &params_input, int cam_id,
               std::pair<int, int> hw) {
  model_id = colmap::CameraModelNameToId(model_name);
  params = params_input;
  if (cam_id != -1)
    camera_id = cam_id;
  if (hw.first != -1 && hw.second != -1) {
    height = hw.first;
    width = hw.second;
  }
}

Camera::Camera(M3D K, int cam_id, std::pair<int, int> hw) {
  THROW_CHECK_EQ(K(0, 1), 0);
  THROW_CHECK_EQ(K(1, 0), 0);
  THROW_CHECK_EQ(K(2, 0), 0);
  THROW_CHECK_EQ(K(2, 1), 0);
  params.clear();
  if (K(0, 0) == K(1, 1)) {
    model_id = colmap::CameraModelNameToId("SIMPLE_PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  } else {
    model_id = colmap::CameraModelNameToId("PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(1, 1));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  }
  if (cam_id != -1)
    camera_id = cam_id;
  if (hw.first != -1 && hw.second != -1) {
    height = hw.first;
    width = hw.second;
  }
}

Camera::Camera(int model, M3D K, int cam_id, std::pair<int, int> hw) {
  THROW_CHECK_EQ(K(0, 1), 0);
  THROW_CHECK_EQ(K(1, 0), 0);
  THROW_CHECK_EQ(K(2, 0), 0);
  THROW_CHECK_EQ(K(2, 1), 0);
  params.clear();
  if (model == 0) {
    // SIMPLE_PINHOLE
    THROW_CHECK_EQ(K(0, 0), K(1, 1));
    model_id = colmap::CameraModelNameToId("SIMPLE_PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  } else if (model == 1) {
    // PINHOLE
    model_id = colmap::CameraModelNameToId("PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(1, 1));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  } else
    throw std::runtime_error(
        "model initialized with K should be either SIMPLE_PINHOLE or PINHOLE");
  if (cam_id != -1)
    camera_id = cam_id;
  if (hw.first != -1 && hw.second != -1) {
    height = hw.first;
    width = hw.second;
  }
}

Camera::Camera(const std::string &model_name, M3D K, int cam_id,
               std::pair<int, int> hw) {
  THROW_CHECK_EQ(K(0, 1), 0);
  THROW_CHECK_EQ(K(1, 0), 0);
  THROW_CHECK_EQ(K(2, 0), 0);
  THROW_CHECK_EQ(K(2, 1), 0);
  params.clear();
  if (model_name == "SIMPLE_PINHOLE") {
    // SIMPLE_PINHOLE
    THROW_CHECK_EQ(K(0, 0), K(1, 1));
    model_id = colmap::CameraModelNameToId("SIMPLE_PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  } else if (model_name == "PINHOLE") {
    // PINHOLE
    model_id = colmap::CameraModelNameToId("PINHOLE");
    params.push_back(K(0, 0));
    params.push_back(K(1, 1));
    params.push_back(K(0, 2));
    params.push_back(K(1, 2));
  } else
    throw std::runtime_error("Error! The model initialized with K should be "
                             "either SIMPLE_PINHOLE (id=0) or PINHOLE (id=1).");
  if (cam_id != -1)
    camera_id = cam_id;
  if (hw.first != -1 && hw.second != -1) {
    height = hw.first;
    width = hw.second;
  }
}

Camera::Camera(const Camera &cam) {
  camera_id = cam.camera_id;
  model_id = cam.model_id;
  params = cam.params;
  height = cam.height;
  width = cam.width;
  initialized = cam.initialized;
}

Camera &Camera::operator=(const Camera &cam) {
  if (this != &cam) {
    camera_id = cam.camera_id;
    model_id = cam.model_id;
    params = cam.params;
    height = cam.height;
    width = cam.width;
    initialized = cam.initialized;
  }
  return *this;
}

bool Camera::operator==(const Camera &cam) {
  if (camera_id != cam.camera_id)
    return false;
  if (model_id != cam.model_id)
    return false;
  if (h() != cam.h())
    return false;
  if (w() != cam.w())
    return false;
  for (int i = 0; i < params.size(); ++i) {
    if (params[i] != cam.params[i])
      return false;
  }
  return true;
}

void Camera::set_max_image_dim(const int &val) {
  THROW_CHECK_EQ(IsUndistorted(), true);
  THROW_CHECK_GT(val, 0);

  double ratio = double(val) / double(std::max(height, width));
  if (ratio < 1.0) {
    int new_width = int(round(ratio * width));
    int new_height = int(round(ratio * height));
    Rescale(new_width, new_height);
  }
}

double Camera::uncertainty(double depth, double var2d) const {
  double f = -1.0;
  auto idxs = FocalLengthIdxs();
  if (idxs.size() == 1)
    f = FocalLength();
  else if (idxs.size() == 2) {
    double fx = FocalLengthX();
    double fy = FocalLengthY();
    f = (fx + fy) / 2.0;
  } else
    throw std::runtime_error(
        "Error! FocalLengthIdxs() should be either 1 or 2");
  double uncertainty = var2d * depth / f;
  return uncertainty;
}

Camera::Camera(py::dict dict) {
  // model id
  int model_id_loaded;
  ASSIGN_PYDICT_ITEM_TKEY(dict, model_id, model_id_loaded, int);
  model_id = static_cast<colmap::CameraModelId>(model_id_loaded);

  // params
  ASSIGN_PYDICT_ITEM(dict, params, std::vector<double>);
  THROW_CHECK(VerifyParams());

  // other fields
  ASSIGN_PYDICT_ITEM(dict, camera_id, int);
  ASSIGN_PYDICT_ITEM(dict, height, int);
  ASSIGN_PYDICT_ITEM(dict, width, int);
  ASSIGN_PYDICT_ITEM(dict, initialized, std::vector<bool>);
}

py::dict Camera::as_dict() const {
  py::dict output;
  output["model_id"] = int(model_id);
  output["params"] = params;
  output["cam_id"] = int(camera_id);
  output["height"] = h();
  output["width"] = w();
  output["initialized"] = initialized;
  return output;
}

double CameraPose::projdepth(const V3D &p3d) const {
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
