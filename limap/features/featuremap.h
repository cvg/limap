#ifndef LIMAP_FEATURES_FEATUREMAP_H_
#define LIMAP_FEATURES_FEATUREMAP_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ceresbase/interpolation.h"
#include "ceresbase/point_projection.h"
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include "util/types.h"
#include <colmap/util/logging.h>

namespace py = pybind11;

namespace limap {

namespace features {

template <typename DTYPE> class FeatureMap {
public:
  FeatureMap() {}
  FeatureMap(const Eigen::MatrixXd &array); // copy here
  FeatureMap(const py::array_t<DTYPE, py::array::c_style> &pyarray,
             bool do_copy = false);

  inline int Width() const { return width; }
  inline int Height() const { return height; }
  inline int Channels() const { return channels; }
  inline int Size() const { return width * height * channels; }

  inline int NumBytes() const { return Size() * sizeof(DTYPE); }
  inline double MemMB() const { return NumBytes() / 1024.0 / 1024.0; }
  inline double MemGB() const { return MemMB() / 1024.0; }

  inline const DTYPE *Data() const { return data_ptr_; }

protected:
  int height, width, channels;
  std::vector<DTYPE>
      data_;        // If we do not own any data, leave this uninitialized
  DTYPE *data_ptr_; // Ptr to data. Used for interface

private:
  Eigen::MatrixXd array_; // temp data
};

// Does not support dynamic channels
template <typename DTYPE, int CHANNELS>
class FeatureInterpolator
    : public PixelInterpolator<ceres::Grid2D<DTYPE, CHANNELS>> {

  using Grid2D = ceres::Grid2D<DTYPE, CHANNELS>;
  using PixelInterpolator<Grid2D>::config_;

public:
  FeatureInterpolator(const InterpolationConfig &config,
                      const FeatureMap<DTYPE> &fmap);
  FeatureMap<DTYPE> GetMap() const { return fmap_; }

  template <typename T> bool Evaluate(const T *xy, T *f);

  template <typename T> inline bool CheckBounds(const T *xy) const;

private:
  Grid2D grid2D_;
  FeatureMap<DTYPE> fmap_;
};

template <typename DTYPE, int CHANNELS>
FeatureInterpolator<DTYPE, CHANNELS>::FeatureInterpolator(
    const InterpolationConfig &config, const FeatureMap<DTYPE> &fmap)
    : grid2D_(fmap.Data(), 0, fmap.Height(), 0, fmap.Width()),
      PixelInterpolator<Grid2D>(config, grid2D_), fmap_(fmap) {}

template <typename DTYPE, int CHANNELS>
template <typename T>
bool FeatureInterpolator<DTYPE, CHANNELS>::Evaluate(const T *xy, T *f) {
  PixelInterpolator<Grid2D>::EvaluateNodes(xy[1], xy[0], f);
  if (config_.check_bounds)
    return CheckBounds(xy);
  else
    return true;
}

template <typename DTYPE, int CHANNELS>
template <typename T>
bool FeatureInterpolator<DTYPE, CHANNELS>::CheckBounds(const T *xy) const {
  return (IsInsideZeroL(xy[0], static_cast<double>(fmap_.Width())) &&
          IsInsideZeroL(xy[1], static_cast<double>(fmap_.Height())));
}

} // namespace features

} // namespace limap

#endif
