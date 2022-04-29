#include "base/image_collection.h"

namespace limap {

ImageCollection::ImageCollection(py::dict dict) {
    // load cameras
    std::vector<py::dict> dictvec_cameras;
    if (dict.contains("cameras"))
        dictvec_cameras = dict["cameras"].cast<std::vector<py::dict>>();
    for (auto it = dictvec_cameras.begin(); it != dictvec_cameras.end(); ++it) {
        cameras.push_back(Camera(*it));
    }
    // load images
    std::vector<py::dict> dictvec_images;
    if (dict.contains("images"))
        dictvec_images = dict["images"].cast<std::vector<py::dict>>();
    for (auto it = dictvec_images.begin(); it != dictvec_images.end(); ++it) {
        images.push_back(CameraImage(*it));
    }
}

py::dict ImageCollection::as_dict() const {
    py::dict output;
    std::vector<py::dict> dictvec_cameras;
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        dictvec_cameras.push_back(it->as_dict());
    }
    output["cameras"] = dictvec_cameras;
    std::vector<py::dict> dictvec_images;
    for (auto it = images.begin(); it != images.end(); ++it) {
        dictvec_images.push_back(it->as_dict());
    }
    output["images"] = dictvec_images;
    return output;
}

Camera ImageCollection::cam(const int cam_id) const {
    THROW_CHECK_GE(cam_id, 0);
    THROW_CHECK_LT(cam_id, NumCameras());
    return cameras[cam_id];
}

CameraImage ImageCollection::camimage(const int img_id) const {
    THROW_CHECK_GE(img_id, 0);
    THROW_CHECK_LT(img_id, NumImages());
    return images[img_id];
}

CameraPose ImageCollection::campose(const int img_id) const {
    THROW_CHECK_GE(img_id, 0);
    THROW_CHECK_LT(img_id, NumImages());
    return images[img_id].pose;
}

CameraView ImageCollection::camview(const int img_id) const {
    THROW_CHECK_GE(img_id, 0);
    THROW_CHECK_LT(img_id, NumImages());
    int cam_id = images[img_id].cam_id;
    return CameraView(cameras[cam_id], images[img_id].pose, images[img_id].image_name());
}

std::string ImageCollection::image_name(const int img_id) const {
    THROW_CHECK_GE(img_id, 0);
    THROW_CHECK_LT(img_id, NumImages());
    return images[img_id].image_name();
}

std::vector<std::string> ImageCollection::get_image_list() const {
    std::vector<std::string> image_names;
    for (size_t img_id = 0; img_id < NumImages(); ++img_id) {
        image_names.push_back(image_name(img_id));
    }
    return image_names;
}

py::array_t<uint8_t> ImageCollection::read_image(const int img_id, const bool set_gray) const {
    return camview(img_id).read_image(set_gray);
}

} // namespace limap

