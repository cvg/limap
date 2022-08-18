#ifndef LIMAP_OPTIMIZE_LINEBA_LINEBA_H_
#define LIMAP_OPTIMIZE_LINEBA_LINEBA_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/featuremap.h"
#include "base/featurepatch.h"
#include "util/types.h"
#include "vplib/vpdet.h"

#include <ceres/ceres.h>
#include "base/line_reconstruction.h"
#include "optimize/line_bundle_adjustment/lineba_config.h"

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace line_bundle_adjustment {

template <typename DTYPE, int CHANNELS>
class LineBAEngine {
private:
    LineBAConfig config_;
    LineReconstruction reconstruction_;

    // VPs
    bool enable_vp = false;
    std::map<int, vplib::VPResult> vpresults_;
    
    // heatmaps (for each image)
    bool enable_heatmap = false; // set to true when calling InitializeHeatmaps()
    std::vector<FeatureMap<DTYPE>> p_heatmaps_f_;
    std::vector<std::unique_ptr<FeatureInterpolator<DTYPE, 1>>> p_heatmaps_itp_;

    // patches (for each supporting image in each track)
    bool enable_feature = false; // InitializePatches()
    std::vector<std::vector<PatchInfo<DTYPE>>> p_patches_; //
    std::vector<std::vector<FeaturePatch<DTYPE>>> p_patches_f_;
    std::vector<std::vector<std::unique_ptr<PatchInterpolator<DTYPE, CHANNELS>>>> p_patches_itp_;

    // set up ceres problem
    void ParameterizeCameras();
    void ParameterizeLines();
    void AddGeometricResiduals(const int track_id);
    void AddVPResiduals(const int track_id);
    void AddHeatmapResiduals(const int track_id);
    void AddFeatureConsistencyResiduals(const int track_id);
    void AddResiduals(const int track_id);

public:
    LineBAEngine() {}
    LineBAEngine(const LineBAConfig& cfg): config_(cfg) {}

    void Initialize(const std::vector<LineTrack>& tracks, const ImageCollection& imagecols) {
        reconstruction_ = LineReconstruction(tracks, imagecols);
    }
    void InitializeReconstruction(const LineReconstruction& reconstruction) {
        reconstruction_ = reconstruction;
    }
    void InitializeVPs(const std::map<int, vplib::VPResult>& vpresults);
    void InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& heatmaps);
    void InitializePatches(const std::vector<std::vector<PatchInfo<DTYPE>>>& patchinfos);
    void SetUp();
    bool Solve();

    // output
    ImageCollection GetOutputCameras() const {return reconstruction_.GetImagecols(); }
    std::vector<Line3d> GetOutputLines() const {return reconstruction_.GetLines(); }
    std::vector<LineTrack> GetOutputTracks() const {return reconstruction_.GetTracks(); }
    LineReconstruction GetOutputReconstruction() const {return reconstruction_; }

    // for visualization
    std::vector<std::vector<V2D>> GetHeatmapIntersections(const LineReconstruction& reconstruction) const; // get all heatmap samples on each image

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
};

} // namespace line_bundle_adjustment 

} // namespace optimize 

} // namespace limap

#endif

