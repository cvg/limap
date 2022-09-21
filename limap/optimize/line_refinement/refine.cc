#include "optimize/line_refinement/refine.h"
#include "optimize/line_refinement/cost_functions.h"
#include "base/camera_models.h"
#include "ceresbase/line_projection.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

namespace limap {

namespace optimize {

namespace line_refinement {

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::Initialize(const LineTrack& track,
                                  const std::vector<CameraView>& p_camviews) 
{
    // validity check
    track_ = track;
    p_camviews_ = p_camviews;

    // initialize optimized line
    inf_line_ = MinimalInfiniteLine3d(track_.line);
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeVPs(const std::vector<vplib::VPResult>& p_vpresults) {
    enable_vp = true;
    p_vpresults_ = p_vpresults;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::ParameterizeMinimalLine() {
    double* uvec_data = inf_line_.uvec.data();
    double* wvec_data = inf_line_.wvec.data();
    ceres::LocalParameterization* quaternion_parameterization = 
        new ceres::QuaternionParameterization;
    problem_->SetParameterization(uvec_data, quaternion_parameterization);
    ceres::LocalParameterization* homo2d_parameterization = 
        new ceres::HomogeneousVectorParameterization(2);
    problem_->SetParameterization(wvec_data, homo2d_parameterization);
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddGeometricResiduals() {
    // compute line weights
    auto idmap = track_.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track_, weights);

    // add to problem for each supporting image
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();
    for (int i = 0; i < n_images; ++i) {
        ceres::LossFunction* loss_function = config_.geometric_loss_function.get();

        auto& view = p_camviews_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            const Line2d& line = track_.line2d_list[*it];
            double weight = weights[*it];
            ceres::CostFunction* cost_function = nullptr;

            switch (view.cam.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = GeometricRefinementFunctor<CameraModel>::Create(line, view.cam.Params().data(), view.pose.qvec.data(), view.pose.tvec.data(), config_.geometric_alpha); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }
    }
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddVPResiduals() {
    // compute line weights
    auto idmap = track_.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track_, weights);

    // add to problem for each supporting image
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();
    ceres::LossFunction* loss_function = config_.vp_loss_function.get();
    for (int i = 0; i < n_images; ++i) {
        auto& view = p_camviews_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            int line_id = track_.line_id_list[*it];
            if (p_vpresults_[i].HasVP(line_id)) {
                const V3D& vp = p_vpresults_[i].GetVP(line_id);
                double weight = weights[*it] * config_.vp_multiplier;
                ceres::CostFunction* cost_function = nullptr;

                switch (view.cam.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = VPConstraintsFunctor<CameraModel>::Create(vp, view.cam.Params().data(), view.pose.qvec.data()); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
            }
        }
    }
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::SetUp() {
    // setup problem
    problem_.reset(new ceres::Problem(config_.problem_options));

    // add residual
    if (config_.use_geometric)
        AddGeometricResiduals();
    if (enable_vp)
        AddVPResiduals();
#ifdef INTERPOLATION_ENABLED
    if (enable_heatmap)
        AddHeatmapResiduals();
    if (enable_feature)
        AddFeatureConsistencyResiduals();
#endif // INTERPOLATION_ENABLED
}

template <typename DTYPE, int CHANNELS>
bool RefinementEngine<DTYPE, CHANNELS>::Solve() {
    if (problem_->NumParameterBlocks() == 0)
        return false;
    if (problem_->NumResiduals() == 0)
        return false;
    // parameterization
    ParameterizeMinimalLine();

    ceres::Solver::Options solver_options = config_.solver_options;
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;

    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_threads);
    #if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
    #endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    state_collector_ = RefinementCallback(problem_.get());
    solver_options.callbacks.push_back(&state_collector_);
    solver_options.update_state_every_iteration = true;
    ceres::Solve(solver_options, problem_.get(), &summary_);

    if (config_.print_summary) {
        colmap::PrintHeading2("Optimization report");
        colmap::PrintSolverSummary(summary_); // We need to replace this with our own Printer!!!
    }
    return true;
}

template <typename DTYPE, int CHANNELS>
Line3d RefinementEngine<DTYPE, CHANNELS>::GetLine3d() const {
    // get cameras for each line
    std::vector<int> p_image_ids = track_.GetIndexesforSorted();

    std::vector<CameraView> cameras;
    int n_lines = track_.count_lines();
    for (int i = 0; i < n_lines; ++i) {
        int index = p_image_ids[i];
        cameras.push_back(p_camviews_[index]);
    }

    // get line segment
    // Line3d line = GetLineSegmentFromInfiniteLine3d(inf_line_.GetInfiniteLine(), cameras, track_.line2d_list, config_.num_outliers_aggregate);
    Line3d line = GetLineSegmentFromInfiniteLine3d(inf_line_.GetInfiniteLine(), track_.line3d_list, config_.num_outliers_aggregate);
    return line;
}

template <typename DTYPE, int CHANNELS>
std::vector<Line3d> RefinementEngine<DTYPE, CHANNELS>::GetAllStates() const {
    // get cameras for each line
    std::vector<int> p_image_ids = track_.GetIndexesforSorted();

    std::vector<CameraView> cameras;
    int n_lines = track_.count_lines();
    for (int i = 0; i < n_lines; ++i) {
        int index = p_image_ids[i];
        cameras.push_back(p_camviews_[index]);
    }
    
    // get line segment for each state
    std::vector<Line3d> line3ds;
    const auto& states = state_collector_.states;
    for (auto it = states.begin(); it!= states.end(); ++it) {
        MinimalInfiniteLine3d minimal_inf_line = MinimalInfiniteLine3d(*it);
        Line3d line3d = GetLineSegmentFromInfiniteLine3d(minimal_inf_line.GetInfiniteLine(), track_.line3d_list, config_.num_outliers_aggregate);
        line3ds.push_back(line3d);
    }
    return line3ds;
}

// Register
template class RefinementEngine<float16, 128>;

} // namespace line_refinement

} // namespace optimize

} // namespace limap

