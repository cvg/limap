#ifndef LIMAP_BASE_LINE_RECONSTRUCTION_H_
#define LIMAP_BASE_LINE_RECONSTRUCTION_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/camera_view.h"
#include "base/image_collection.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/infinite_line.h"

namespace py = pybind11;

namespace limap {

// line reconstruction with minimal line paramaterization
class LineReconstruction {
public:
    LineReconstruction() {}
    LineReconstruction(const std::vector<LineTrack>& linetracks, const ImageCollection& imagecols);

    // data to be optimized
    std::vector<MinimalInfiniteLine3d> lines_; // minimal line for each track
    ImageCollection imagecols_;

    // some interfaces for the initial data
    LineTrack GetInitTrack(const int track_id) const {return init_tracks_[track_id]; }
    std::vector<LineTrack> GetInitTracks() const { return init_tracks_; }
    ImageCollection GetInitImagecols() const { return init_imagecols_; }
    std::map<int, CameraView> GetInitCameraMap() const;
    size_t NumTracks() const {return lines_.size(); }
    size_t NumImages() const {return imagecols_.NumImages(); }
    size_t NumSupportingLines(const int track_id) const {return init_tracks_[track_id].count_lines(); }
    size_t NumSupportingImages(const int track_id) const {return init_tracks_[track_id].count_images(); }
    std::vector<int> GetImageIds(const int track_id) const {return init_tracks_[track_id].image_id_list; }
    std::vector<Line2d> GetLine2ds(const int track_id) const {return init_tracks_[track_id].line2d_list; }
    std::vector<Line3d> GetLine3ds(const int track_id) const {return init_tracks_[track_id].line3d_list; }

    // output
    std::vector<MinimalInfiniteLine3d> GetStates() const { return lines_; }
    ImageCollection GetImagecols() const { return imagecols_; };
    std::vector<Line3d> GetLines(const int num_outliers = 2) const;
    std::vector<LineTrack> GetTracks(const int num_outliers = 2) const;

private:
    std::vector<LineTrack> init_tracks_;
    ImageCollection init_imagecols_;
};

} // namespace limap

#endif

