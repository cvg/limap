#ifndef LIMAP_BASE_LINETRACK_H_
#define LIMAP_BASE_LINETRACK_H_

#include <cmath>
#include <map>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>

namespace py = pybind11;

#include "base/camera_view.h"
#include "base/infinite_line.h"
#include "base/linebase.h"
#include "util/types.h"

namespace limap {

class LineTrack {
public:
  LineTrack() {}
  LineTrack(const LineTrack &track);
  LineTrack(const Line3d &line_, const std::vector<int> &image_id_list_,
            const std::vector<int> &line_id_list_,
            const std::vector<Line2d> &line2d_list_)
      : line(line_), image_id_list(image_id_list_), line_id_list(line_id_list_),
        line2d_list(line2d_list_) {}
  py::dict as_dict() const;
  LineTrack(py::dict dict);

  // properties
  Line3d line;
  std::vector<int> image_id_list;
  std::vector<int> line_id_list;
  std::vector<Line2d> line2d_list;

  // auxiliary information (may not be initialized)
  std::vector<int> node_id_list;
  std::vector<Line3d> line3d_list;
  std::vector<double> score_list;

  // active status for recursive merging
  bool active = true;

  size_t count_lines() const { return line2d_list.size(); }
  std::vector<int> GetSortedImageIds() const;
  std::map<int, int> GetIndexMapforSorted() const;
  std::vector<int> GetIndexesforSorted() const;
  size_t count_images() const { return GetSortedImageIds().size(); }
  std::vector<Line2d> projection(const std::vector<CameraView> &views) const;
  std::map<int, std::vector<int>> GetIdMap() const; // (img_id, {index})

  void Resize(const size_t &n_lines);
  bool HasImage(const int &image_id) const;
  void Read(const std::string &filename);
  void Write(const std::string &filename) const;
};

////////////////////////////////////////////////////////////
// sampling for optimization
////////////////////////////////////////////////////////////
void ComputeLineWeights(
    const LineTrack &track,
    std::vector<double> &weights); // weights.size() == track.count_lines()

void ComputeLineWeightsNormalized(
    const LineTrack &track,
    std::vector<double> &weights); // weights.size() == track.count_lines()

void ComputeHeatmapSamples(const LineTrack &track,
                           std::vector<std::vector<InfiniteLine2d>>
                               &heatmap_samples, // samples for each line
                           const std::pair<double, double> sample_range,
                           const int n_samples);

void ComputeFConsistencySamples(
    const LineTrack &track,
    const std::map<int, CameraView> &views, // {img_id, view}
    std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>>
        &fconsis_samples, // [ref_image_id, sample, {tgt_image_id(s)}]
    const std::pair<double, double> sample_range, const int n_samples);

} // namespace limap

#endif
