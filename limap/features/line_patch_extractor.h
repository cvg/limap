#ifndef LIMAP_FEATURES_LINE_PATCH_EXTRACTOR_H_
#define LIMAP_FEATURES_LINE_PATCH_EXTRACTOR_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/camera_view.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "features/featurepatch.h"
#include "util/types.h"

namespace py = pybind11;

namespace limap {

namespace features {

class LinePatchExtractorOptions {
public:
  LinePatchExtractorOptions() {}
  LinePatchExtractorOptions(py::dict dict) : LinePatchExtractorOptions() {
    ASSIGN_PYDICT_ITEM(dict, k_stretch, double);
    ASSIGN_PYDICT_ITEM(dict, t_stretch, int);
    ASSIGN_PYDICT_ITEM(dict, range_perp, int);
  }
  // finallength = std::max(length * k_stretch, length + t_stretch)
  double k_stretch = 1.0; // by default we do not stretch lines
  int t_stretch = 10;     // in pixels
  int range_perp = 20;    // in pixels
};

template <typename DTYPE, int CHANNELS> class LinePatchExtractor {
public:
  LinePatchExtractor() {}
  LinePatchExtractor(const LinePatchExtractorOptions &options)
      : options_(options) {}

  PatchInfo<DTYPE>
  ExtractLinePatch(const Line2d &line2d,
                   const py::array_t<DTYPE, py::array::c_style> &feature);
  std::vector<PatchInfo<DTYPE>>
  ExtractLinePatches(const std::vector<Line2d> &line2ds,
                     const py::array_t<DTYPE, py::array::c_style> &feature);

  Line2d GetLine2DRange(const LineTrack &track, const int image_id,
                        const CameraView &view);

  PatchInfo<DTYPE>
  ExtractOneImage(const LineTrack &track, const int image_id,
                  const CameraView &view,
                  const py::array_t<DTYPE, py::array::c_style> &feature);

  void
  Extract(const LineTrack &track, const std::vector<CameraView> &p_views,
          const std::vector<py::array_t<DTYPE, py::array::c_style>> &p_features,
          std::vector<PatchInfo<DTYPE>> &patchinfos);

private:
  LinePatchExtractorOptions options_;
};

} // namespace features

} // namespace limap

#endif
