#ifndef LIMAP_FEATURES_FEATUREPATCH_H_
#define LIMAP_FEATURES_FEATUREPATCH_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "features/featuremap.h"
#include "util/types.h"

#include <colmap/util/logging.h>

namespace py = pybind11;

namespace limap {

namespace features {

template <typename DTYPE> struct PatchInfo {
  py::array_t<DTYPE, py::array::c_style> array;
  M2D R;
  V2D tvec;
  std::pair<int, int> img_hw;

  PatchInfo() {}
  PatchInfo(const py::array_t<DTYPE, py::array::c_style> &array_, const M2D &R_,
            const V2D &tvec_, const std::pair<int, int> &img_hw_)
      : array(array_), R(R_), tvec(tvec_), img_hw(img_hw_) {}
};

// patches as oriented bounding box
template <typename DTYPE> class FeaturePatch : public FeatureMap<DTYPE> {
public:
  FeaturePatch(const py::array_t<DTYPE, py::array::c_style> &pyarray,
               const M2D &R_, const V2D &tvec_,
               const std::pair<int, int> &img_hw_, bool do_copy = false)
      : FeatureMap<DTYPE>(pyarray, do_copy), R(R_), tvec(tvec_),
        img_hw(img_hw_) {}
  FeaturePatch(const PatchInfo<DTYPE> &pinfo, bool do_copy = false)
      : FeaturePatch(pinfo.array, pinfo.R, pinfo.tvec, pinfo.img_hw, do_copy) {}

  // local_xy = R * (xy - tvec)
  M2D R;
  V2D tvec;
  std::pair<int, int> img_hw; // the height and width of the original image
                              // (used for checking bounds)
  PatchInfo<DTYPE> GetPatchInfo() const;

  template <typename T> void GlobalToLocal(const T *xy, T *local_xy) const;

  template <typename T> void LocalToGlobal(const T *local_xy, T *xy) const;
};

template <typename DTYPE>
template <typename T>
void FeaturePatch<DTYPE>::GlobalToLocal(const T *xy, T *local_xy) const {
  T R11, R12, R21, R22;
  R11 = T(R(0, 0));
  R12 = T(R(0, 1));
  R21 = T(R(1, 0));
  R22 = T(R(1, 1));
  T t1, t2;
  t1 = T(tvec(0));
  t2 = T(tvec(1));

  local_xy[0] = R11 * (xy[0] - t1) + R12 * (xy[1] - t2);
  local_xy[1] = R21 * (xy[0] - t1) + R22 * (xy[1] - t2);
}

template <typename DTYPE>
template <typename T>
void FeaturePatch<DTYPE>::LocalToGlobal(const T *local_xy, T *xy) const {
  T R11, R12, R21, R22;
  R11 = T(R(0, 0));
  R12 = T(R(0, 1));
  R21 = T(R(1, 0));
  R22 = T(R(1, 1));
  T t1, t2;
  t1 = T(tvec(0));
  t2 = T(tvec(1));

  xy[0] = R11 * local_xy[0] + R21 * local_xy[1] + t1;
  xy[1] = R12 * local_xy[0] + R22 * local_xy[1] + t2;
}

// Does not support dynamic channels
template <typename DTYPE, int CHANNELS>
class PatchInterpolator
    : public PixelInterpolator<ceres::Grid2D<DTYPE, CHANNELS>> {

  using Grid2D = ceres::Grid2D<DTYPE, CHANNELS>;
  using PixelInterpolator<Grid2D>::config_;

public:
  PatchInterpolator(const InterpolationConfig &config,
                    const FeaturePatch<DTYPE> &fpatch);
  FeaturePatch<DTYPE> GetPatch() const { return fpatch_; }

  template <typename T> bool Evaluate(const T *xy, T *f);

  template <typename T> inline bool CheckBounds(const T *xy) const;

private:
  Grid2D grid2D_;
  FeaturePatch<DTYPE> fpatch_;
};

template <typename DTYPE, int CHANNELS>
PatchInterpolator<DTYPE, CHANNELS>::PatchInterpolator(
    const InterpolationConfig &config, const FeaturePatch<DTYPE> &fpatch)
    : grid2D_(fpatch.Data(), 0, fpatch.Height(), 0, fpatch.Width()),
      PixelInterpolator<Grid2D>(config, grid2D_), fpatch_(fpatch) {}

template <typename DTYPE, int CHANNELS>
template <typename T>
bool PatchInterpolator<DTYPE, CHANNELS>::Evaluate(const T *xy, T *f) {
  T local_xy[2];
  fpatch_.GlobalToLocal(xy, local_xy);
  PixelInterpolator<Grid2D>::EvaluateNodes(local_xy[1], local_xy[0], f);
  if (config_.check_bounds)
    return CheckBounds(xy);
  else
    return true;
}

template <typename DTYPE, int CHANNELS>
template <typename T>
bool PatchInterpolator<DTYPE, CHANNELS>::CheckBounds(const T *xy) const {
  bool res_global =
      (IsInsideZeroL(xy[0], static_cast<double>(fpatch_.img_hw.second)) &&
       IsInsideZeroL(xy[1], static_cast<double>(fpatch_.img_hw.first)));
  T local_xy[2];
  fpatch_.GlobalToLocal(xy, local_xy);
  bool res_local =
      (IsInsideZeroL(local_xy[0], static_cast<double>(fpatch_.Width())) &&
       IsInsideZeroL(local_xy[1], static_cast<double>(fpatch_.Height())));
  return res_global && res_local;
}

} // namespace features

} // namespace limap

#endif
