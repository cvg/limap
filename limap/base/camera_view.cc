#include "base/camera_view.h"

namespace limap {

CameraImage::CameraImage(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, cam_id, int)
    pose = CameraPose(dict);

    // load image name
    std::string image_name;
    ASSIGN_PYDICT_ITEM(dict, image_name, std::string);
    SetImageName(image_name);
}

py::dict CameraImage::as_dict() const {
    py::dict output;
    output["cam_id"] = cam_id;
    output["qvec"] = pose.qvec;
    output["tvec"] = pose.tvec;
    output["image_name"] = image_name_;
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
    output["image_name"] = image_name();
    return output;
}

py::array_t<uint8_t> CameraView::read_image(const bool set_gray) const {
    py::object cv2 = py::module_::import("cv2");
    py::array_t<uint8_t> img = cv2.attr("imread")(image_name());
    img = cv2.attr("resize")(img, std::make_pair(w(), h()));
    if (set_gray) {
        img = cv2.attr("cvtColor")(img, cv2.attr("COLOR_BGR2GRAY"));
    }
    return img;
}

Eigen::MatrixXd CameraView::matrix() const {
    Eigen::MatrixXd P(3, 4);
    P.block<3, 3>(0, 0) = R();
    P.col(3) = T();
    P = K() * P;
    return P;
}

V2D CameraView::projection(const V3D& p3d) const {
    V3D p_homo = K() * (R() * p3d + T());
    V2D p2d = dehomogeneous(p_homo);
    return p2d;
}

V3D CameraView::ray_direction(const V2D& p2d) const {
    return (R().transpose() * K_inv() * homogeneous(p2d)).normalized();
}

V3D CameraView::get_direction_from_vp(const V3D& vp) const {
    return (R().transpose() * K_inv() * vp).normalized();
}

} // namespace limap

