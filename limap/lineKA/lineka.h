#ifndef LIMAP_LINEKA_LINEKA_H_
#define LIMAP_LINEKA_LINEKA_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/linebase.h"
#include "base/line2d_parameterization.h"
#include "base/camera.h"
#include "base/graph.h"
#include "base/featurepatch.h"
#include "util/types.h"

#include <ceres/ceres.h>
#include "lineKA/lineka_config.h"

namespace py = pybind11;

namespace limap {

namespace lineKA {

template <typename DTYPE, int CHANNELS>
class LineKAEngine {
private:
    LineKAConfig config_;

    // 2DOF lines to optimize
    std::vector<std::vector<std::shared_ptr<Line2d_2DOF>>> all_lines_2dof_; 

    // cameras
    std::vector<CameraView> cameras_matrixform_; // original input, with matrices as attributes
    std::vector<MinimalPinholeCamera> cameras_; 

    // matching information
    DirectedGraph graph_;
        
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
    void ParameterizeLines();
    void AddHeatmapResiduals(const int img_id);

    // return t_array_local that fits the given match
    std::vector<double> check_samples_epipolar(const std::shared_ptr<Line2d_2DOF>& ref_line, const MinimalPinholeCamera& cam_ref, const std::shared_ptr<Line2d_2DOF>& tgt_line, const MinimalPinholeCamera& cam_tgt, const std::vector<double> t_array) const;
    void AddFeatureConsistencyResiduals();

public:
    LineKAEngine() {}
    LineKAEngine(const LineKAConfig& cfg): config_(cfg) {}

    void Initialize(const std::vector<std::vector<Line2d>>& all_lines_2d, 
                    const std::vector<CameraView>& cameras);
    void InitializeMatches(const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                           const std::vector<std::vector<int>>& neighbors);
    void InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& heatmaps);
    void InitializePatches(const std::vector<std::vector<PatchInfo<DTYPE>>>& patchinfos);
    void SetUp();
    bool Solve();

    // output
    std::vector<std::vector<Line2d>> GetOutputLines() const;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
};

} // namespace lineKA 

} // namespace limap

#endif

