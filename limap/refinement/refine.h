#ifndef LIMAP_REFINEMENT_REFINE_H_
#define LIMAP_REFINEMENT_REFINE_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/camera_view.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/infinite_line.h"
#include "base/featuremap.h"
#include "base/featurepatch.h"
#include "util/types.h"
#include "vpdetection/vpdet.h"

#include <ceres/ceres.h>
#include <tuple>
#include "refinement/refinement_config.h"

namespace py = pybind11;

namespace limap {

namespace refinement {

class RefinementCallback: public ceres::IterationCallback {
public:
    RefinementCallback(): ceres::IterationCallback() {}
    RefinementCallback(ceres::Problem* problem_): ceres::IterationCallback(), problem(problem_) { states.clear(); }
    ceres::CallbackReturnType operator() (const ceres::IterationSummary& summary) {
        std::vector<double*> parameter_blocks;
        problem->GetParameterBlocks(&parameter_blocks);
        if (parameter_blocks.size() != 2) // not the case with camera fixed
            return ceres::SOLVER_CONTINUE;
        std::vector<double> state;
        auto& uvec = parameter_blocks[0];
        auto& wvec = parameter_blocks[1];
        state.push_back(uvec[0]);
        state.push_back(uvec[1]);
        state.push_back(uvec[2]);
        state.push_back(uvec[3]);
        state.push_back(wvec[0]);
        state.push_back(wvec[1]);
        states.push_back(state);
        return ceres::SOLVER_CONTINUE;
    }
    ceres::Problem* problem;
    std::vector<std::vector<double>> states;
};

template <typename DTYPE, int CHANNELS>
class RefinementEngine {
private:
    // p_camviews_ and p_heatmaps have same size with track.count_images(), and the order is the same as track.GetSortedImageIds()
    RefinementConfig config_;
    LineTrack track_;
    // cameras are with the same order as track_.GetSortedImageIds()
    std::vector<CameraView> p_camviews_matrixform_; // original input, with matrices as attributes
    std::vector<MinimalPinholeCamera> p_camviews_; 

    // optimized line
    MinimalInfiniteLine3d inf_line_;

    // VPs
    bool enable_vp = false;
    std::vector<vpdetection::VPResult> p_vpresults_;

    // heatmaps
    bool enable_heatmap = false; // set to true when calling InitializeHeatmaps()
    std::vector<FeatureMap<DTYPE>> p_heatmaps_f_;
    std::vector<std::unique_ptr<FeatureInterpolator<DTYPE, 1>>> p_heatmaps_itp_;

    // features
    bool enable_feature = false; // set to true when calling InitializeFeatures() or InitializeFeaturesAsPatches()
    bool use_patches = false; // set to true when calling InitializeFeaturesAsPatches()
    std::vector<py::array_t<DTYPE, py::array::c_style>> p_features_; // size: number of supporting images
    std::vector<FeatureMap<DTYPE>> p_features_f_;
    std::vector<std::unique_ptr<FeatureInterpolator<DTYPE, CHANNELS>>> p_features_itp_;

    std::vector<PatchInfo<DTYPE>> p_patches_; // size: number of supporting images 
    std::vector<FeaturePatch<DTYPE>> p_patches_f_;
    std::vector<std::unique_ptr<PatchInterpolator<DTYPE, CHANNELS>>> p_patches_itp_;

    // set up ceres problem
    void ParameterizeMinimalLine();
    void AddGeometricResiduals();
    void AddVPResiduals();
    void AddHeatmapResiduals();
    void AddFeatureConsistencyResiduals();

public:
    RefinementEngine() {}
    RefinementEngine(const RefinementConfig& cfg): config_(cfg) {}

    void Initialize(const LineTrack& track,
                    const std::vector<CameraView>& p_views); // the order of p_camviews conform that of track.GetSortedImageIds()
    void InitializeVPs(const std::vector<vpdetection::VPResult>& p_vpresults);
    void InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& p_heatmaps);
    void InitializeFeatures(const std::vector<py::array_t<DTYPE, py::array::c_style>>& p_featuremaps);
    void InitializeFeaturesAsPatches(const std::vector<PatchInfo<DTYPE>>& patchinfos);
    void SetUp();

    bool Solve();
    Line3d GetLine3d() const;

    // for visualization, ids are corresponding to track_.GetSortedImageIds()
    std::vector<Line3d> GetAllStates() const;
    std::vector<std::vector<V2D>> GetHeatmapIntersections(const Line3d& line) const; // collection of 2d samples for each image
    std::vector<std::vector<std::pair<int, V2D>>> GetFConsistencyIntersections(const Line3d& line) const; // collection of (image_id, projection) for each 3d point track
    
    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    RefinementCallback state_collector_;
};

} // namespace refinement

} // namespace limap

#endif

