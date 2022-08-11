#include "base/line_reconstruction.h"

namespace limap {

LineReconstruction::LineReconstruction(const std::vector<LineTrack>& linetracks, const ImageCollection& imagecols) 
{
    // initialize minimal infinite lines
    init_tracks_ = linetracks;
    for (auto it = linetracks.begin(); it != linetracks.end(); ++it) {
        lines_.push_back(MinimalInfiniteLine3d(it->line));
    }

    // initialize minimal cameras
    init_imagecols_ = imagecols;
    imagecols_ = ImageCollection(imagecols);
}

std::vector<Line3d> LineReconstruction::GetLines(const int num_outliers) const {
    std::map<int, CameraView> cameras = GetCameras();
    std::vector<Line3d> lines;
    int n_tracks = lines_.size();
    for (int track_id = 0; track_id < n_tracks; ++track_id) {
        std::vector<int> image_ids = GetImageIds(track_id);
        std::vector<CameraView> p_cameras;
        for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
            p_cameras.push_back(cameras[*it]);
        }
        Line3d line = GetLineSegmentFromInfiniteLine3d(lines_[track_id].GetInfiniteLine(), GetLine3ds(track_id), num_outliers);
        lines.push_back(line);
    }
    return lines;
}

std::vector<LineTrack> LineReconstruction::GetTracks(const int num_outliers) const {
    std::vector<LineTrack> tracks;
    std::vector<Line3d> lines = GetLines(num_outliers);
    int n_tracks = lines_.size();
    for (int track_id = 0; track_id < n_tracks; ++track_id) {
        LineTrack newtrack;
        newtrack = init_tracks_[track_id];
        newtrack.line = lines[track_id];
        tracks.push_back(newtrack);
    }
    return tracks;
}

} // namespace limap

