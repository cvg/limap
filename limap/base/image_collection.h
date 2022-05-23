#ifndef LIMAP_BASE_CAMERA_VIEW_H
#define LIMAP_BASE_CAMERA_VIEW_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cmath>
#include <fstream>

namespace py = pybind11;

#include "util/types.h"
#include "_limap/helpers.h"

#include "base/camera.h"
#include "base/camera_view.h"

namespace limap {

class ImageCollection {
public:
    ImageCollection() {}
    ImageCollection(const std::map<int, Camera>& input_cameras, const std::vector<CameraImage>& input_images): cameras(input_cameras), images(input_images) {}
    ImageCollection(const std::vector<Camera>& input_cameras, const std::vector<CameraImage>& input_images);
    ImageCollection(const std::vector<CameraView>& camviews);
    ImageCollection(py::dict dict);
    py::dict as_dict() const;

    size_t NumCameras() const { return cameras.size(); }
    size_t NumImages() const { return images.size(); }
    std::vector<Camera> get_cameras() const;
    std::vector<int> get_cam_ids() const;
    std::vector<CameraImage> get_images() const { return images; }
    std::vector<CameraView> get_camviews() const;
    bool IsUndistorted() const;

    Camera cam(const int cam_id) const;
    bool exist_cam(const int cam_id) const;
    CameraImage camimage(const int img_id) const;
    CameraPose campose(const int img_id) const;
    CameraView camview(const int img_id) const;
    std::string image_name(const int img_id) const;
    std::vector<std::string> get_image_list() const;

    py::array_t<uint8_t> read_image(const int img_id, const bool set_gray) const;
    void set_max_image_dim(const int& val);
    
private:
    std::map<int, Camera> cameras;
    std::vector<CameraImage> images;
};

} // namespace limap

#endif

