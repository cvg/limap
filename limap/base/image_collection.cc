#include "base/image_collection.h"

namespace limap {

ImageCollection::ImageCollection(const std::vector<Camera>& input_cameras, const std::vector<CameraImage>& input_images) {
    int n_cameras = input_cameras.size();
    for (int cam_id = 0; cam_id < n_cameras; ++cam_id) {
        cameras.insert(std::make_pair(cam_id, input_cameras[cam_id]));
    }
    images = input_images;
}

ImageCollection::ImageCollection(const std::vector<CameraView>& camviews) {
    for (auto it = camviews.begin(); it != camviews.end(); ++it) {
        images.push_back(CameraImage(*it));
        int cam_id = it->cam.CameraId();
        if (cameras.count(cam_id) == 1) {
            CHECK_EQ(cameras.at(cam_id) == it->cam, true);
        }
        else {
            cameras.insert(std::make_pair(it->cam.CameraId(), it->cam));
        }
    }
}

ImageCollection::ImageCollection(py::dict dict) {
    // load cameras
    std::map<int, py::dict> dictvec_cameras;
    if (dict.contains("cameras"))
        dictvec_cameras = dict["cameras"].cast<std::map<int, py::dict>>();
    for (auto it = dictvec_cameras.begin(); it != dictvec_cameras.end(); ++it) {
        int cam_id = it->first;
        Camera cam = Camera(it->second);
        assert (cam_id == cam.CameraId());
        cameras.insert(std::make_pair(cam_id, cam));
    }
    // load images
    std::vector<py::dict> dictvec_images;
    if (dict.contains("images"))
        dictvec_images = dict["images"].cast<std::vector<py::dict>>();
    for (auto it = dictvec_images.begin(); it != dictvec_images.end(); ++it) {
        images.push_back(CameraImage(*it));
    }
}

std::vector<Camera> ImageCollection::get_cameras() const {
    std::vector<Camera> output_cameras;
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        output_cameras.push_back(it->second);
    }
    return output_cameras;
}

std::vector<int> ImageCollection::get_cam_ids() const {
    std::vector<int> output_ids;
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        output_ids.push_back(it->first);
    }
    return output_ids;
}

std::vector<CameraView> ImageCollection::get_camviews() const {
    std::vector<CameraView> camviews;
    for (int img_id = 0; img_id < NumImages(); ++img_id) {
        camviews.push_back(camview(img_id));
    }
    return camviews;
}

py::dict ImageCollection::as_dict() const {
    py::dict output;
    std::map<int, py::dict> dictvec_cameras;
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        dictvec_cameras.insert(std::make_pair(it->first, it->second.as_dict()));
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
    THROW_CHECK_EQ(cameras.count(cam_id), 1);
    return cameras.at(cam_id);
}

bool ImageCollection::exist_cam(const int cam_id) const {
    return cameras.count(cam_id) == 1;
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
    return CameraView(cameras.at(cam_id), images[img_id].pose, images[img_id].image_name());
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

void ImageCollection::set_max_image_dim(const int& val) {
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        it->second.set_max_image_dim(val);
    }
}

bool ImageCollection::IsUndistorted() const {
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        if (!it->second.IsUndistorted())
            return false;
    }
    return true;
}

} // namespace limap

