#include "base/camera_view.h"

namespace limap {

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

MinimalPinholeCamera cam2minimalcam(const CameraView& view) {
    MinimalPinholeCamera cam = MinimalPinholeCamera(view);
    return cam;
}

CameraView minimalcam2cam(const MinimalPinholeCamera& camera) {
    return camera.GetCameraView();
} 

} // namespace limap

