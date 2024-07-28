#include "features/line_patch_extractor.h"
#include "ceresbase/interpolation.h"
#include "features/featuremap.h"

#include <algorithm>
#include <iostream>

namespace limap {

namespace features {

template <typename DTYPE, int CHANNELS>
PatchInfo<DTYPE> LinePatchExtractor<DTYPE, CHANNELS>::ExtractLinePatch(
    const Line2d &line2d,
    const py::array_t<DTYPE, py::array::c_style> &feature) {
  // initialize interpolator
  InterpolationConfig interp_cfg;
  interp_cfg.interpolation = InterpolatorType::BILINEAR;
  interp_cfg.check_bounds = true;
  auto interpolator = std::unique_ptr<FeatureInterpolator<DTYPE, CHANNELS>>(
      new FeatureInterpolator<DTYPE, CHANNELS>(interp_cfg,
                                               FeatureMap<DTYPE>(feature)));

  // get directions
  V2D direc = line2d.direction();
  V2D perp_direc = line2d.perp_direction();

  // stretch the range
  V2D midpoint = line2d.midpoint();
  double finallength = std::max(line2d.length() * options_.k_stretch,
                                line2d.length() + options_.t_stretch);
  finallength = std::ceil(finallength);
  Line2d line2d_final;
  line2d_final.start = midpoint - direc * finallength / 2.0;
  line2d_final.end = midpoint + direc * finallength / 2.0;

  // compute transformation
  V2D corner = line2d_final.start - perp_direc * options_.range_perp / 2.0;
  M2D R2d;
  R2d.row(0) = perp_direc;
  R2d.row(1) = direc;

  // sample on the feature to get patches (size: (finallength + 1,
  // options_.range_perp + 1, channels))
  int patch_h = int(std::round(line2d_final.length())) + 1;
  int patch_w = options_.range_perp + 1;
  py::array_t<DTYPE, py::array::c_style> pyarray =
      py::array_t<DTYPE, py::array::c_style>(
          std::vector<size_t>{static_cast<unsigned long>(patch_h),
                              static_cast<unsigned long>(patch_w), CHANNELS});
  DTYPE *data_ptr = static_cast<DTYPE *>(pyarray.request().ptr);

#pragma omp parallel for
  for (int x = 0; x < patch_w; ++x) {
    for (int y = 0; y < patch_h; ++y) {
      V2D local_xy(x, y);
      V2D xy = R2d.transpose() * local_xy + corner;
      DTYPE input[2];
      input[0] = DTYPE(xy(0));
      input[1] = DTYPE(xy(1));
      DTYPE *ptr = data_ptr + pyarray.index_at(y, x, 0);
      interpolator->Evaluate(input, ptr);
    }
  }

  PatchInfo<DTYPE> patchinfo;
  patchinfo.array = pyarray;
  patchinfo.R = R2d;
  patchinfo.tvec = corner;
  patchinfo.img_hw.first = interpolator->GetMap().Height();
  patchinfo.img_hw.second = interpolator->GetMap().Width();
  return patchinfo;
}

template <typename DTYPE, int CHANNELS>
std::vector<PatchInfo<DTYPE>>
LinePatchExtractor<DTYPE, CHANNELS>::ExtractLinePatches(
    const std::vector<Line2d> &line2ds,
    const py::array_t<DTYPE, py::array::c_style> &feature) {
  // initialize interpolator
  InterpolationConfig interp_cfg;
  interp_cfg.interpolation = InterpolatorType::BILINEAR;
  interp_cfg.check_bounds = true;

  std::vector<PatchInfo<DTYPE>> patchinfos(line2ds.size());
  for (size_t line_id = 0; line_id < line2ds.size(); ++line_id) {
    auto interpolator = std::unique_ptr<FeatureInterpolator<DTYPE, CHANNELS>>(
        new FeatureInterpolator<DTYPE, CHANNELS>(interp_cfg,
                                                 FeatureMap<DTYPE>(feature)));
    const Line2d &line2d = line2ds[line_id];

    // get directions
    V2D direc = line2d.direction();
    V2D perp_direc = line2d.perp_direction();

    // stretch the range
    V2D midpoint = line2d.midpoint();
    double finallength = std::max(line2d.length() * options_.k_stretch,
                                  line2d.length() + options_.t_stretch);
    finallength = std::ceil(finallength);
    Line2d line2d_final;
    line2d_final.start = midpoint - direc * finallength / 2.0;
    line2d_final.end = midpoint + direc * finallength / 2.0;

    // compute transformation
    V2D corner = line2d_final.start - perp_direc * options_.range_perp / 2.0;
    M2D R2d;
    R2d.row(0) = perp_direc;
    R2d.row(1) = direc;

    // sample on the feature to get patches (size: (finallength + 1,
    // options_.range_perp + 1, channels))
    int patch_h = int(std::round(line2d_final.length())) + 1;
    int patch_w = options_.range_perp + 1;
    py::array_t<DTYPE, py::array::c_style> pyarray =
        py::array_t<DTYPE, py::array::c_style>(
            std::vector<size_t>{static_cast<unsigned long>(patch_h),
                                static_cast<unsigned long>(patch_w), CHANNELS});
    DTYPE *data_ptr = static_cast<DTYPE *>(pyarray.request().ptr);

    for (int x = 0; x < patch_w; ++x) {
      for (int y = 0; y < patch_h; ++y) {
        V2D local_xy(x, y);
        V2D xy = R2d.transpose() * local_xy + corner;
        DTYPE input[2];
        input[0] = DTYPE(xy(0));
        input[1] = DTYPE(xy(1));
        DTYPE *ptr = data_ptr + pyarray.index_at(y, x, 0);
        interpolator->Evaluate(input, ptr);
      }
    }

    patchinfos[line_id].array = pyarray;
    patchinfos[line_id].R = R2d;
    patchinfos[line_id].tvec = corner;
    patchinfos[line_id].img_hw.first = interpolator->GetMap().Height();
    patchinfos[line_id].img_hw.second = interpolator->GetMap().Width();
  }
  return patchinfos;
}

template <typename DTYPE, int CHANNELS>
Line2d LinePatchExtractor<DTYPE, CHANNELS>::GetLine2DRange(
    const LineTrack &track, const int image_id, const CameraView &view) {
  // collect all supporting 2d segments
  std::vector<Line2d> line2d_supports;
  for (int line_id; line_id < track.count_lines(); ++line_id) {
    if (track.image_id_list[line_id] != image_id)
      continue;
    line2d_supports.push_back(track.line2d_list[line_id]);
  }

  // get global 2d projection and project all endpoints on it to get 2d range
  // along the line
  Line2d line2d_global = track.line.projection(view);
  V2D direc = line2d_global.direction();
  std::vector<double> projections;
  for (const auto &line2d : line2d_supports) {
    projections.push_back((line2d.start - line2d_global.start).dot(direc));
    projections.push_back((line2d.end - line2d_global.start).dot(direc));
  }
  std::sort(projections.begin(), projections.end());
  Line2d line2d_range;
  line2d_range.start =
      line2d_global.start + direc * std::max(0.0, projections[0]);
  line2d_range.end =
      line2d_global.start +
      direc * std::min(line2d_global.length(), projections.back());
  return line2d_range;
}

template <typename DTYPE, int CHANNELS>
PatchInfo<DTYPE> LinePatchExtractor<DTYPE, CHANNELS>::ExtractOneImage(
    const LineTrack &track, const int image_id, const CameraView &view,
    const py::array_t<DTYPE, py::array::c_style> &feature) {
  // get a global line2d
  Line2d line2d_range = GetLine2DRange(track, image_id, view);

  // extract line patch
  return ExtractLinePatch(line2d_range, feature);
}

template <typename DTYPE, int CHANNELS>
void LinePatchExtractor<DTYPE, CHANNELS>::Extract(
    const LineTrack &track, const std::vector<CameraView> &p_views,
    const std::vector<py::array_t<DTYPE, py::array::c_style>> &p_features,
    std::vector<PatchInfo<DTYPE>> &patchinfos) {
  patchinfos.clear();
  int n_images = track.count_images();
  THROW_CHECK_EQ(p_views.size(), n_images);
  THROW_CHECK_EQ(p_features.size(), n_images);
  std::vector<int> sorted_ids = track.GetSortedImageIds();
  THROW_CHECK_EQ(sorted_ids.size(), n_images);
  for (int i = 0; i < n_images; ++i) {
    PatchInfo<DTYPE> patchinfo =
        ExtractOneImage(track, sorted_ids[i], p_views[i], p_features[i]);
    patchinfos.push_back(patchinfo);
  }
  return;
}

// The interpolator only supports double and JetT at inference
// So we interpolate on double and cast in python
#define REGISTER_CHANNEL(CHANNELS)                                             \
  template class LinePatchExtractor<double, CHANNELS>;

REGISTER_CHANNEL(1);
REGISTER_CHANNEL(2);
REGISTER_CHANNEL(3);
REGISTER_CHANNEL(16);
REGISTER_CHANNEL(32);
REGISTER_CHANNEL(64);
REGISTER_CHANNEL(128);

#undef REGISTER_CHANNEL

} // namespace features

} // namespace limap
