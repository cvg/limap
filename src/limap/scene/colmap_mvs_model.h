#pragma once

#include <filesystem>

#include "limap/util/eigen_types.h"
#include <colmap/mvs/image.h>
#include <colmap/mvs/model.h>

namespace limap {

namespace scene {

colmap::mvs::Image CreateCOLMAPMVSImage(const std::string &filename,
                                        const int width, const int height,
                                        const std::vector<double> &K,
                                        const std::vector<double> &R,
                                        const std::vector<double> &T);

class COLMAPMVSModel : public colmap::mvs::Model {
public:
  COLMAPMVSModel() : colmap::mvs::Model() {}

  void AddPoint(double x, double y, double z,
                const std::vector<int> &image_ids);

  void AddImage(const colmap::mvs::Image &image, const int img_id = -1);

  void ReadFromCOLMAP(const std::filesystem::path &path,
                      const std::filesystem::path &sparse_path = "sparse",
                      const std::filesystem::path &image_path = "images");

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
  std::vector<int> reg_image_ids_;
};

} // namespace scene

} // namespace limap
