#ifndef LIMAP_POINTSFM_SFM_MODEL_H_
#define LIMAP_POINTSFM_SFM_MODEL_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "util/types.h"
#include <colmap/mvs/image.h>
#include <colmap/mvs/model.h>

namespace py = pybind11;

namespace limap {

namespace pointsfm {

colmap::mvs::Image CreateSfmImage(const std::string &filename, const int width,
                                  const int height,
                                  const std::vector<double> &K,
                                  const std::vector<double> &R,
                                  const std::vector<double> &T);

class SfmModel : public colmap::mvs::Model {
public:
  SfmModel() : colmap::mvs::Model() {}

  void addPoint(double x, double y, double z,
                const std::vector<int> &image_ids);

  void addImage(const colmap::mvs::Image &image, const int img_id = -1);

  void ReadFromCOLMAP(const std::string &path,
                      const std::string &sparse_path = "sparse",
                      const std::string &images_path = "images");

  std::vector<std::string> GetImageNames() const;

  std::vector<int> ComputeNumPoints() const;

  std::map<int, std::vector<int>>
  GetMaxOverlapImages(const size_t num_images,
                      const double min_triangulationo_angle) const;

  std::map<int, std::vector<int>>
  GetMaxIoUImages(const size_t num_images,
                  const double min_triangulationo_angle) const;

  std::map<int, std::vector<int>>
  GetMaxDiceCoeffImages(const size_t num_images,
                        const double min_triangulationo_angle) const;

  std::pair<V3D, V3D>
  ComputeRanges(const std::pair<double, double> &range_robust,
                const double &kstretch) const;

private:
  std::pair<float, float>
  get_robust_range(std::vector<float> &data,
                   const std::pair<double, double> &range_robust,
                   const double &kstretch) const;

  std::vector<int> reg_image_ids;
  std::map<int, std::vector<int>>
  neighbors_vec_to_map(const std::vector<std::vector<int>> &neighbors) const;
};

} // namespace pointsfm

} // namespace limap

#endif
