#ifndef LIMAP_POINTSFM_SFM_MODEL_H_
#define LIMAP_POINTSFM_SFM_MODEL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include <colmap/mvs/model.h>
#include <colmap/mvs/image.h>
#include "util/types.h"

namespace py = pybind11;

namespace limap {

namespace pointsfm {

colmap::mvs::Image CreateSfmImage(const std::string& filename, 
                                  const int width, const int height, 
                                  const std::vector<double>& K, 
                                  const std::vector<double>& R, 
                                  const std::vector<double>& T);

class SfmModel: public colmap::mvs::Model {
public:
    SfmModel(): colmap::mvs::Model() {}

    void addPoint(double x, double y, double z, const std::vector<int>& image_ids);

    void addImage(const colmap::mvs::Image& image);

    std::vector<std::string> GetImageNames() const;

    std::vector<int> ComputeNumPoints() const;

    std::vector<std::vector<int>> GetMaxIoUImages(
            const size_t num_images, const double min_triangulationo_angle) const;

    std::vector<std::vector<int>> GetMaxDiceCoeffImages(
            const size_t num_images, const double min_triangulationo_angle) const;

    std::pair<V3D, V3D> ComputeRanges(const std::pair<double, double>& range_robust, const double& kstretch) const;

private:
    std::pair<float, float> get_robust_range(std::vector<float>& data, const std::pair<double, double>& range_robust, const double& kstretch) const;
};

} // namespace pointsfm

} // namespace limap

#endif

