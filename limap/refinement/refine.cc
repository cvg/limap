#include "refinement/refine.h"
#include "refinement/cost_functions.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

namespace limap {

namespace refinement {

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::Initialize(const LineTrack& track,
                                  const std::vector<PinholeCamera>& p_cameras) 
{
    // validity check
    track_ = track;
    p_cameras_.clear();
    for (auto it = p_cameras.begin(); it != p_cameras.end(); ++it) {
        p_cameras_matrixform_.push_back(*it);
        p_cameras_.push_back(cam2minimalcam(*it));
    }

    // initialize optimized line
    inf_line_ = MinimalInfiniteLine3d(track_.line);
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeVPs(const std::vector<vpdetection::VPResult>& p_vpresults) {
    enable_vp = true;
    p_vpresults_ = p_vpresults;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& p_heatmaps) {
    enable_heatmap = true;
    int n_images = track_.count_images();
    THROW_CHECK_EQ(p_heatmaps.size(), n_images);
    for (int i = 0; i < n_images; ++i) {
        THROW_CHECK_EQ(p_cameras_[i].height, p_heatmaps[i].rows());
        THROW_CHECK_EQ(p_cameras_[i].width, p_heatmaps[i].cols());
    }
    
    auto& interp_cfg = config_.heatmap_interpolation_config;
    p_heatmaps_f_.clear();
    p_heatmaps_itp_.clear();
    double memGB = 0;
    for (auto it = p_heatmaps.begin(); it != p_heatmaps.end(); ++it) {
        p_heatmaps_f_.push_back(FeatureMap<DTYPE>(*it));
        p_heatmaps_itp_.push_back(
                std::unique_ptr<FeatureInterpolator<DTYPE, 1>>
                    (new FeatureInterpolator<DTYPE, 1>(interp_cfg, p_heatmaps_f_.back()))
        );
        memGB += p_heatmaps_f_.back().MemGB();
    }
    if (config_.print_summary)
        std::cout<<"[INFO] Initialize heatmaps (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeFeatures(const std::vector<py::array_t<DTYPE, py::array::c_style>>& p_features) {
    enable_feature = true;
    int n_images = track_.count_images();
    THROW_CHECK_EQ(p_features.size(), n_images);
    
    auto& interp_cfg = config_.feature_interpolation_config;
    p_features_.clear();
    p_features_f_.clear();
    p_features_itp_.clear();
    double memGB = 0;
    for (auto it = p_features.begin(); it != p_features.end(); ++it) {
        p_features_.push_back(*it);
        p_features_f_.push_back(FeatureMap<DTYPE>(*it));
        p_features_itp_.push_back(
                std::unique_ptr<FeatureInterpolator<DTYPE, CHANNELS>>
                    (new FeatureInterpolator<DTYPE, CHANNELS>(interp_cfg, p_features_f_.back()))
        );
        memGB += p_features_f_.back().MemGB();
    }
    if (config_.print_summary)
        std::cout<<"[INFO] Initialize features (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeFeaturesAsPatches(const std::vector<PatchInfo<DTYPE>>& patchinfos) {
    enable_feature = true;
    use_patches = true;
    int n_images = track_.count_images();
    THROW_CHECK_EQ(patchinfos.size(), n_images);
    
    auto& interp_cfg = config_.feature_interpolation_config;
    p_patches_.clear();
    p_patches_f_.clear();
    p_patches_itp_.clear();
    double memGB = 0;
    for (auto it = patchinfos.begin(); it != patchinfos.end(); ++it) {
        p_patches_.push_back(*it);
        p_patches_f_.push_back(FeaturePatch<DTYPE>(*it));
        p_patches_itp_.push_back(
                std::unique_ptr<PatchInterpolator<DTYPE, CHANNELS>>
                    (new PatchInterpolator<DTYPE, CHANNELS>(interp_cfg, p_patches_f_.back()))
        );
        memGB += p_patches_f_.back().MemGB();
    }
    if (config_.print_summary)
        std::cout<<"[INFO] Initialize features as patches (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
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

        auto& camera = p_cameras_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            const Line2d& line = track_.line2d_list[*it];
            double weight = weights[*it];
            ceres::CostFunction* cost_function = GeometricRefinementFunctor::Create(line, camera.kvec.data(), camera.qvec.data(), camera.tvec.data(), config_.geometric_alpha);
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
        auto& camera = p_cameras_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            int line_id = track_.line_id_list[*it];
            if (p_vpresults_[i].HasVP(line_id)) {
                const V3D& vp = p_vpresults_[i].GetVP(line_id);
                double weight = weights[*it] * config_.vp_multiplier;
                ceres::CostFunction* cost_function = VPConstraintsFunctor::Create(vp, camera.kvec.data(), camera.qvec.data());
                ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
            }
        }
    }
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddHeatmapResiduals() {
    // compute heatmap samples
    std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
    ComputeHeatmapSamples(track_, 
                          heatmap_samples,
                          std::make_pair(config_.sample_range_min, config_.sample_range_max),
                          config_.n_samples_heatmap);

    // compute line weights
    auto idmap = track_.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track_, weights);

    // add to problem for each supporting image 
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();
    for (int i = 0; i < n_images; ++i) {
        ceres::LossFunction* loss_function = config_.heatmap_loss_function.get();
        auto& camera = p_cameras_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            const auto& samples = heatmap_samples.at(*it);
            double weight = weights[*it] * config_.heatmap_multiplier / double(config_.n_samples_heatmap / 10.0);
            ceres::CostFunction* cost_function = MaxHeatmapFunctor<DTYPE>::Create(p_heatmaps_itp_[i], samples, camera.kvec.data(), camera.qvec.data(), camera.tvec.data());
            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }
    }
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddFeatureConsistencyResiduals() {
    // compute fconsis samples
    std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>> fconsis_samples; // [referenece_image_id, infiniteline2d, {target_image_ids}]
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();
    std::map<int, PinholeCamera> cameramap;
    for (int i = 0; i < n_images; ++i) {
        cameramap.insert(std::make_pair(image_ids[i], p_cameras_matrixform_[i]));
    }

    ComputeFConsistencySamples(track_,
                               cameramap,
                               fconsis_samples,
                               std::make_pair(config_.sample_range_min, config_.sample_range_max),
                               config_.n_samples_feature);

    // add to problem for each sample
    std::map<int, int> idmap = track_.GetIndexMapforSorted();
    int n_samples = fconsis_samples.size();
    for (int i = 0; i < n_samples; ++i) {
        ceres::LossFunction* loss_function = config_.fconsis_loss_function.get();

        const auto& sample_tp = fconsis_samples[i];
        int ref_image_id = std::get<0>(sample_tp);
        int ref_index = idmap.at(ref_image_id);
        const auto& sample = std::get<1>(sample_tp);
        std::vector<int> tgt_image_ids = std::get<2>(sample_tp);
        auto& cam_ref = p_cameras_[ref_index];

        // get reference descriptor and compute ref residuals
        double weight = config_.fconsis_multiplier / (double(n_samples / 100.0) * double(tgt_image_ids.size() / 5.0));
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        double* ref_descriptor = NULL;
        if (config_.use_ref_descriptor) {
            ref_descriptor = new double[CHANNELS];
            V2D ref_intersection;
            GetIntersection2d<double>(inf_line_.uvec.data(), inf_line_.wvec.data(),
                                      cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                      sample.p.data(), sample.direc.data(),
                                      ref_intersection.data());
            if (use_patches)
                p_patches_itp_[ref_index]->Evaluate(ref_intersection.data(), ref_descriptor);
            else
                p_features_itp_[ref_index]->Evaluate(ref_intersection.data(), ref_descriptor);

            // source residuals
            ceres::CostFunction* cost_function = nullptr;
            if (use_patches) {
                cost_function = FeatureConsisSrcFunctor<PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_index], sample, ref_descriptor, 
                                                                                                              cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data());
            }
            else {
                cost_function = FeatureConsisSrcFunctor<FeatureInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_features_itp_[ref_index], sample, ref_descriptor, 
                                                                                                                cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data());
            }
            ceres::LossFunction* src_scaled_loss_function = new ceres::ScaledLoss(scaled_loss_function, config_.ref_multiplier, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, src_scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }

        // compute tgt residuals
        for (const int& tgt_image_id: tgt_image_ids) {
            int tgt_index = idmap.at(tgt_image_id);
            auto& cam_tgt = p_cameras_[tgt_index];
            // tgt residuals
            ceres::CostFunction* cost_function = nullptr;
            if (use_patches) {
                cost_function = FeatureConsisTgtFunctor<PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_index], p_patches_itp_[tgt_index], sample, ref_descriptor, 
                                                                                                              cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                                                                                              cam_tgt.kvec.data(), cam_tgt.qvec.data(), cam_tgt.tvec.data());
            }
            else {
                cost_function = FeatureConsisTgtFunctor<FeatureInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_features_itp_[ref_index], p_features_itp_[tgt_index], sample, ref_descriptor,
                                                                                                                cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                                                                                                cam_tgt.kvec.data(), cam_tgt.qvec.data(), cam_tgt.tvec.data());
            }
            ceres::ResidualBlockId block_id_tgt = problem_->AddResidualBlock(cost_function, scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }
        if (config_.use_ref_descriptor)
            delete ref_descriptor;
    }
    return;
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
    if (enable_heatmap)
        AddHeatmapResiduals();
    if (enable_feature)
        AddFeatureConsistencyResiduals();
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

    std::vector<PinholeCamera> cameras;
    int n_lines = track_.count_lines();
    for (int i = 0; i < n_lines; ++i) {
        int index = p_image_ids[i];
        cameras.push_back(p_cameras_matrixform_[index]);
    }

    // get line segment
    Line3d line = GetLineSegmentFromInfiniteLine3d(inf_line_.GetInfiniteLine(), track_.line3d_list, config_.num_outliers_aggregate);
    return line;
}

template <typename DTYPE, int CHANNELS>
std::vector<Line3d> RefinementEngine<DTYPE, CHANNELS>::GetAllStates() const {
    // get cameras for each line
    std::vector<int> p_image_ids = track_.GetIndexesforSorted();

    std::vector<PinholeCamera> cameras;
    int n_lines = track_.count_lines();
    for (int i = 0; i < n_lines; ++i) {
        int index = p_image_ids[i];
        cameras.push_back(p_cameras_matrixform_[index]);
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

template <typename DTYPE, int CHANNELS>
std::vector<std::vector<V2D>> RefinementEngine<DTYPE, CHANNELS>::GetHeatmapIntersections(const Line3d& line) const {
    MinimalInfiniteLine3d inf_line = MinimalInfiniteLine3d(line);
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();

    // compute heatmap samples
    std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
    ComputeHeatmapSamples(track_, 
                          heatmap_samples,
                          std::make_pair(config_.sample_range_min, config_.sample_range_max),
                          config_.n_samples_heatmap);
    auto idmap = track_.GetIdMap();

    // compute intersections for each supporting image
    std::vector<std::vector<V2D>> out_samples;
    for (int i = 0; i < n_images; ++i) {
        const auto& camera = p_cameras_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);

        std::vector<V2D> out_samples_idx;
        for (auto it1 = ids.begin(); it1 != ids.end(); ++it1) {
            const auto& samples = heatmap_samples.at(*it1);
            for (auto it2 = samples.begin(); it2 != samples.end(); ++it2) {
                V2D intersection;
                GetIntersection2d<double>(inf_line.uvec.data(), inf_line.wvec.data(), 
                                          camera.kvec.data(), camera.qvec.data(), camera.tvec.data(),
                                          it2->p.data(), it2->direc.data(),
                                          intersection.data());
                out_samples_idx.push_back(intersection);
            }
        }
        out_samples.push_back(out_samples_idx);
    }
    return out_samples;
}

template <typename DTYPE, int CHANNELS>
std::vector<std::vector<std::pair<int, V2D>>> RefinementEngine<DTYPE, CHANNELS>::GetFConsistencyIntersections(const Line3d& line) const {
    MinimalInfiniteLine3d inf_line = MinimalInfiniteLine3d(line);

    // compute fconsis samples
    std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>> fconsis_samples; // [referenece_image_id, infiniteline2d, {target_image_ids}]
    int n_images = track_.count_images();
    std::vector<int> image_ids = track_.GetSortedImageIds();
    std::map<int, PinholeCamera> cameramap;
    for (int i = 0; i < n_images; ++i) {
        cameramap.insert(std::make_pair(image_ids[i], p_cameras_matrixform_[i]));
    }

    ComputeFConsistencySamples(track_,
                               cameramap,
                               fconsis_samples,
                               std::make_pair(config_.sample_range_min, config_.sample_range_max),
                               config_.n_samples_feature);

    // compute intersections for each sample
    std::vector<std::vector<std::pair<int, V2D>>> out_samples;
    std::map<int, int> idmap = track_.GetIndexMapforSorted();
    for (auto& sample_tp: fconsis_samples) {
        int ref_image_id = std::get<0>(sample_tp);
        int ref_index = idmap.at(ref_image_id);
        const InfiniteLine2d& sample = std::get<1>(sample_tp);
        std::vector<int> tgt_image_ids = std::get<2>(sample_tp);
        const auto& cam_ref = p_cameras_[ref_index];

        // get ref_intersection
        std::vector<std::pair<int, V2D>> out_samples_idx;
        V2D ref_intersection;
        GetIntersection2d<double>(inf_line.uvec.data(), inf_line.wvec.data(),
                                  cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                  sample.p.data(), sample.direc.data(),
                                  ref_intersection.data());
        out_samples_idx.push_back(std::make_pair(ref_index, ref_intersection));

        // get tgt_intersection for each target image
        for (const int& tgt_image_id: tgt_image_ids) {
            int tgt_index = idmap.at(tgt_image_id);
            const auto& cam_tgt = p_cameras_[tgt_index];
            V3D epiline_coord;
            V2D tgt_intersection;
            GetEpipolarLineCoordinate<double>(cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                              cam_tgt.kvec.data(), cam_tgt.qvec.data(), cam_tgt.tvec.data(),
                                              ref_intersection.data(), epiline_coord.data());
            GetIntersection2d_line_coordinate<double>(inf_line.uvec.data(), inf_line.wvec.data(),
                                                      cam_tgt.kvec.data(), cam_tgt.qvec.data(), cam_tgt.tvec.data(),
                                                      epiline_coord.data(), tgt_intersection.data());
            out_samples_idx.push_back(std::make_pair(tgt_index, tgt_intersection));
        }
        out_samples.push_back(out_samples_idx);
    }
    return out_samples;
}

#define REGISTER_CHANNEL(CHANNELS) \
    template class RefinementEngine<float16, CHANNELS>; \
    template class RefinementEngine<float, CHANNELS>; \
    template class RefinementEngine<double, CHANNELS>;

REGISTER_CHANNEL(1);
REGISTER_CHANNEL(3);
REGISTER_CHANNEL(128);

#undef REGISTER_CHANNEL

} // namespace refinement

} // namespace limap

