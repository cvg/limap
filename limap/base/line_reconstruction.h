#ifndef LIMAP_BASE_LINE_RECONSTRUCTION_H_
#define LIMAP_BASE_LINE_RECONSTRUCTION_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/camera.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/infinite_line.h"

#include <map>

namespace py = pybind11;

namespace limap {

// minimal representation of the line reconstruction
class LineReconstruction {
public:
    LineReconstruction() {}
    LineReconstruction(const std::vector<LineTrack>& linetracks, const std::vector<CameraView>& cameras);

    // minimal data
    std::vector<MinimalInfiniteLine3d> lines_; // minimal line for each track
    std::vector<MinimalPinholeCamera> cameras_; // minimal camera for each image

    // interface
    LineTrack GetInitTrack(const int track_id) const {return init_tracks_[track_id]; }
    std::vector<LineTrack> GetInitTracks() const {return init_tracks_; }
    std::map<int, CameraView> GetCameraMap() const {return init_cameras_; }
    size_t NumTracks() const {return lines_.size(); }
    size_t NumCameras() const {return cameras_.size(); }

    size_t NumSupportingLines(const int track_id) const {return init_tracks_[track_id].count_lines(); }
    size_t NumSupportingImages(const int track_id) const {return init_tracks_[track_id].count_images(); }
    std::vector<int> GetImageIds(const int track_id) const {return init_tracks_[track_id].image_id_list; }
    std::vector<Line2d> GetLine2ds(const int track_id) const {return init_tracks_[track_id].line2d_list; }
    std::vector<Line3d> GetLine3ds(const int track_id) const {return init_tracks_[track_id].line3d_list; }

    std::vector<MinimalInfiniteLine3d> GetStates() const {return lines_;}
    std::vector<CameraView> GetCameras() const;
    std::vector<Line3d> GetLines(const int num_outliers = 2) const;
    std::vector<LineTrack> GetTracks(const int num_outliers = 2) const;

private:
    // original data
    std::vector<LineTrack> init_tracks_;
    std::map<int, CameraView> init_cameras_; // cameras in map format {img_id, camera}
};

} // namespace limap

#endif

