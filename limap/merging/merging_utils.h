#ifndef LIMAP_MERGING_MERGING_UTILS_H_
#define LIMAP_MERGING_MERGING_UTILS_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "base/camera_view.h"
#include "base/graph.h"
#include "base/image_collection.h"
#include "base/linebase.h"
#include "base/linetrack.h"

namespace limap {

namespace merging {

std::vector<Line3d> SetUncertaintySegs3d(const std::vector<Line3d> &lines,
                                         const CameraView &view,
                                         const double var2d = 5.0);

void CheckReprojection(std::vector<bool> &results, const LineTrack &linetrack,
                       const ImageCollection &imagecols,
                       const double &th_angular2d, const double &th_perp2d);

void FilterSupportingLines(std::vector<LineTrack> &new_linetracks,
                           const std::vector<LineTrack> &linetracks,
                           const ImageCollection &imagecols,
                           const double &th_angular2d, const double &th_perp2d,
                           const int num_outliers = 2);

void CheckSensitivity(std::vector<bool> &results, const LineTrack &linetrack,
                      const ImageCollection &imagecols,
                      const double &th_angular3d);

void FilterTracksBySensitivity(std::vector<LineTrack> &new_linetracks,
                               const std::vector<LineTrack> &linetracks,
                               const ImageCollection &imagecols,
                               const double &th_angular3d,
                               const int &min_support_ns);

void FilterTracksByOverlap(std::vector<LineTrack> &new_linetracks,
                           const std::vector<LineTrack> &linetracks,
                           const ImageCollection &imagecols,
                           const double &th_overlap, const int &min_support_ns);

} // namespace merging

} // namespace limap

#endif
