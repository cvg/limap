#ifndef LIMAP_BASE_IMAGE_COLLECTION_H_
#define LIMAP_BASE_IMAGE_COLLECTION_H_

#include <cmath>
#include <fstream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <unordered_set>

namespace py = pybind11;

#include "_limap/helpers.h"
#include "util/types.h"

#include "base/camera.h"
#include "base/camera_view.h"
#include "base/transforms.h"

namespace limap {

class ImageCollection {
public:
  ImageCollection() {}
  ImageCollection(const std::map<int, Camera> &input_cameras,
                  const std::map<int, CameraImage> &input_images)
      : cameras(input_cameras), images(input_images) {}
  ImageCollection(const std::map<int, Camera> &input_cameras,
                  const std::vector<CameraImage> &input_images);
  ImageCollection(const std::vector<Camera> &input_cameras,
                  const std::map<int, CameraImage> &input_images);
  ImageCollection(const std::vector<Camera> &input_cameras,
                  const std::vector<CameraImage> &input_images);
  ImageCollection(const std::vector<CameraView> &camviews);
  ImageCollection(py::dict dict);
  ImageCollection(const ImageCollection &imagecols);
  py::dict as_dict() const;
  ImageCollection
  subset_by_camera_ids_set(const std::set<int> &valid_camera_ids) const;
  ImageCollection subset_by_camera_ids_set(
      const std::unordered_set<int> &valid_camera_ids) const;
  ImageCollection
  subset_by_camera_ids(const std::vector<int> &valid_camera_ids) const;
  ImageCollection
  subset_by_image_ids_set(const std::set<int> &valid_image_ids) const;
  ImageCollection
  subset_by_image_ids_set(const std::unordered_set<int> &valid_image_ids) const;
  ImageCollection
  subset_by_image_ids(const std::vector<int> &valid_image_ids) const;
  ImageCollection subset_initialized() const;
  std::map<int, std::vector<int>>
  update_neighbors(const std::map<int, std::vector<int>> &neighbors) const;

  size_t NumCameras() const { return cameras.size(); }
  size_t NumImages() const { return images.size(); }
  std::vector<Camera> get_cameras() const;
  std::vector<int> get_cam_ids() const;
  std::vector<CameraImage> get_images() const;
  std::vector<int> get_img_ids() const;
  std::vector<CameraView> get_camviews() const;
  std::map<int, CameraView> get_map_camviews() const;
  std::vector<V3D> get_locations() const;
  std::map<int, V3D> get_map_locations() const;
  bool IsUndistorted() const;
  bool IsUndistortedCameraModel() const;

  Camera cam(const int cam_id) const;
  bool exist_cam(const int cam_id) const;
  CameraImage camimage(const int img_id) const;
  bool exist_image(const int img_id) const;
  CameraPose campose(const int img_id) const;
  CameraView camview(const int img_id) const;
  std::string image_name(const int img_id) const;
  std::vector<std::string> get_image_name_list() const;
  std::map<int, std::string> get_image_name_dict() const;

  py::array_t<uint8_t> read_image(const int img_id, const bool set_gray) const;
  void set_max_image_dim(const int &val);
  void set_camera_params(const int cam_id, const std::vector<double> &params);
  void change_camera(const int cam_id, const Camera cam);

  void set_camera_pose(const int img_id,
                       const CameraPose pose); // load new poses into imagecols
  CameraPose
  get_camera_pose(const int img_id) const; // get poses from imagecols

  void change_image(const int img_id, const CameraImage camimage);
  void change_image_name(const int img_id, const std::string new_name);

  double *params_data(const int img_id);
  double *qvec_data(const int img_id);
  double *tvec_data(const int img_id);

  ImageCollection
  apply_similarity_transform(const SimilarityTransform3 &transform) const;

  // inverse indexing
  int get_first_image_id_by_camera_id(const int cam_id) const;

  // init uninitialized cameras
  void init_uninitialized_cameras();

  // unintialization
  void uninitialize_poses();
  void uninitialize_intrinsics();

private:
  std::map<int, Camera> cameras;
  std::map<int, CameraImage> images;

  void init_cameras_from_vector(const std::vector<Camera> &input_cameras);
  void init_images_from_vector(const std::vector<CameraImage> &input_images);
};

} // namespace limap

#endif
