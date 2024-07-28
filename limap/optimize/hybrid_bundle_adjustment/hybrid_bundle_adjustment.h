#ifndef LIMAP_OPTIMIZE_HYBRIDBA_HYBRIDBA_H_
#define LIMAP_OPTIMIZE_HYBRIDBA_HYBRIDBA_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/image_collection.h"
#include "base/infinite_line.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"
#include "util/types.h"
#include "vplib/vpbase.h"

#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment_config.h"
#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace hybrid_bundle_adjustment {

class HybridBAEngine {
protected:
  HybridBAConfig config_;

  // minimal data
  ImageCollection imagecols_;
  std::map<int, V3D> points_;
  std::map<int, MinimalInfiniteLine3d> lines_;

  // tracks
  std::map<int, PointTrack> point_tracks_;
  std::map<int, LineTrack> line_tracks_;

  // set up ceres problem
  void ParameterizeCameras();
  void ParameterizePoints();
  void ParameterizeLines();
  void AddPointGeometricResiduals(const int track_id);
  void AddLineGeometricResiduals(const int track_id);

public:
  HybridBAEngine() {}
  HybridBAEngine(const HybridBAConfig &cfg) : config_(cfg) {}

  void InitImagecols(const ImageCollection &imagecols) {
    imagecols_ = imagecols;
  }
  void InitPointTracks(const std::vector<PointTrack> &point_tracks);
  void InitPointTracks(const std::map<int, PointTrack> &point_tracks);
  void InitLineTracks(const std::vector<LineTrack> &line_tracks);
  void InitLineTracks(const std::map<int, LineTrack> &line_tracks);
  void SetUp();
  bool Solve();

  // output
  ImageCollection GetOutputImagecols() const { return imagecols_; }
  std::map<int, V3D> GetOutputPoints() const;
  std::map<int, PointTrack> GetOutputPointTracks() const;
  std::map<int, Line3d> GetOutputLines(const int num_outliers) const;
  std::map<int, LineTrack> GetOutputLineTracks(const int num_outliers) const;

  // ceres
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
};

} // namespace hybrid_bundle_adjustment

} // namespace optimize

} // namespace limap

#endif
