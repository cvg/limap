#include "optimize/line_bundle_adjustment/lineba.h"
#include "optimize/line_refinement/cost_functions.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

namespace limap {

namespace optimize {

namespace line_bundle_adjustment {

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::InitializeVPs(const std::map<int, vplib::VPResult>& vpresults) {
    THROW_CHECK_EQ(reconstruction_.NumImages(), vpresults.size());
    enable_vp = true;
    vpresults_ = vpresults;
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& heatmaps) {
    enable_heatmap = true;
    THROW_CHECK_EQ(reconstruction_.NumImages(), heatmaps.size());
    auto& interp_cfg = config_.heatmap_interpolation_config;

    int n_images = heatmaps.size();
    p_heatmaps_f_.clear();
    p_heatmaps_itp_.clear();
    double memGB = 0;
    for (auto it = heatmaps.begin(); it != heatmaps.end(); ++it) {
        p_heatmaps_f_.push_back(features::FeatureMap<DTYPE>(*it));
        p_heatmaps_itp_.push_back(
                std::unique_ptr<features::FeatureInterpolator<DTYPE, 1>>
                    (new features::FeatureInterpolator<DTYPE, 1>(interp_cfg, p_heatmaps_f_.back()))
        );
        memGB += p_heatmaps_f_.back().MemGB();
    }
    std::cout<<"[INFO] Initialize heatmaps (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::InitializePatches(const std::vector<std::vector<features::PatchInfo<DTYPE>>>& patchinfos) {
    enable_feature = true;
    THROW_CHECK_EQ(reconstruction_.NumTracks(), patchinfos.size());
    auto& interp_cfg = config_.feature_interpolation_config;

    int n_tracks = patchinfos.size();
    p_patches_.clear(); p_patches_.resize(n_tracks);
    p_patches_f_.clear(); p_patches_f_.resize(n_tracks);
    p_patches_itp_.clear(); p_patches_itp_.resize(n_tracks);

    double memGB = 0;
    int n_patches = 0;
    for (int track_id = 0; track_id < n_tracks; ++track_id) {
        const auto& patches = patchinfos[track_id];
        for (auto it = patches.begin(); it != patches.end(); ++it) {
            p_patches_[track_id].push_back(*it);
            p_patches_f_[track_id].push_back(features::FeaturePatch<DTYPE>(*it));
            p_patches_itp_[track_id].push_back(
                    std::unique_ptr<features::PatchInterpolator<DTYPE, CHANNELS>>
                        (new features::PatchInterpolator<DTYPE, CHANNELS>(interp_cfg, p_patches_f_[track_id].back()))
            );
            memGB += p_patches_f_[track_id].back().MemGB();
            n_patches += 1;
        }
    }
    std::cout<<"[INFO] Initialize features as patches (n_patches = "<<n_patches<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::ParameterizeCameras() {
    for (const int& img_id: reconstruction_.imagecols_.get_img_ids()) {
        double* params_data = reconstruction_.imagecols_.params_data(img_id);
        double* qvec_data = reconstruction_.imagecols_.qvec_data(img_id);
        double* tvec_data = reconstruction_.imagecols_.tvec_data(img_id);

        if (config_.constant_intrinsics) {
            if (!problem_->HasParameterBlock(params_data))
                continue;
            problem_->SetParameterBlockConstant(params_data);
        }

        if (config_.constant_pose) {
            if (!problem_->HasParameterBlock(qvec_data))
                continue;
            problem_->SetParameterBlockConstant(qvec_data);
            if (!problem_->HasParameterBlock(tvec_data))
                continue;
            problem_->SetParameterBlockConstant(tvec_data);
        }
        else {
            ceres::LocalParameterization* quaternion_parameterization = 
                new ceres::QuaternionParameterization;
            problem_->SetParameterization(qvec_data, quaternion_parameterization);
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::ParameterizeLines() {
    size_t n_tracks = reconstruction_.NumTracks();
    for (size_t track_id = 0; track_id < n_tracks; ++track_id) {
        size_t n_images = reconstruction_.GetInitTrack(track_id).count_images();
        double* uvec_data = reconstruction_.lines_[track_id].uvec.data();
        double* wvec_data = reconstruction_.lines_[track_id].wvec.data();

        // check if the track is in the problem
        if (!problem_->HasParameterBlock(uvec_data))
            continue;
        if (!problem_->HasParameterBlock(wvec_data))
            continue;

        if (config_.constant_line || n_images < config_.min_num_images) {
            problem_->SetParameterBlockConstant(uvec_data);
            problem_->SetParameterBlockConstant(wvec_data);
        }
        else {
            ceres::LocalParameterization* quaternion_parameterization = 
                new ceres::QuaternionParameterization;
            problem_->SetParameterization(uvec_data, quaternion_parameterization);
            ceres::LocalParameterization* homo2d_parameterization = 
                new ceres::HomogeneousVectorParameterization(2);
            problem_->SetParameterization(wvec_data, homo2d_parameterization);
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::AddGeometricResiduals(const int track_id) {
    const LineTrack& track = reconstruction_.GetInitTrack(track_id);

    // compute line weights
    auto idmap = track.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track, weights);

    // add to problem for each supporting image (for each supporting line) 
    auto& inf_line = reconstruction_.lines_[track_id];
    std::vector<int> image_ids = track.GetSortedImageIds();
    ceres::LossFunction* loss_function = config_.geometric_loss_function.get();
    for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
        int img_id = *it1;
        int model_id = reconstruction_.imagecols_.camview(img_id).cam.ModelId();
        const auto& ids = idmap.at(img_id);
        for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
            const Line2d& line = track.line2d_list[*it2];
            double weight = weights[*it2];
            ceres::CostFunction* cost_function = nullptr;

            switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::GeometricRefinementFunctor<CameraModel>::Create(line, NULL, NULL, NULL, config_.geometric_alpha); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                         inf_line.uvec.data(), inf_line.wvec.data(),
                                                                         reconstruction_.imagecols_.params_data(img_id), reconstruction_.imagecols_.qvec_data(img_id), reconstruction_.imagecols_.tvec_data(img_id));
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::AddVPResiduals(const int track_id) {
    const LineTrack& track = reconstruction_.GetInitTrack(track_id);

    // compute line weights
    auto idmap = track.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track, weights);

    // add to problem for each supporting image (for each supporting line)
    auto& inf_line = reconstruction_.lines_[track_id];
    std::vector<int> image_ids = track.GetSortedImageIds();
    ceres::LossFunction* loss_function = config_.vp_loss_function.get();
    for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
        int img_id = *it1;
        int model_id = reconstruction_.imagecols_.camview(img_id).cam.ModelId();
        const auto& ids = idmap.at(img_id);
        for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
            int line_id = track.line_id_list[*it2];
            if (vpresults_[img_id].HasVP(line_id)) {
                const V3D& vp = vpresults_[img_id].GetVP(line_id);
                double weight = weights[*it2] * config_.vp_multiplier;
                ceres::CostFunction* cost_function = nullptr;
                
                switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::VPConstraintsFunctor<CameraModel>::Create(vp); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                             inf_line.uvec.data(), inf_line.wvec.data(),
                                                                             reconstruction_.imagecols_.params_data(img_id), reconstruction_.imagecols_.qvec_data(img_id));
            }
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::AddHeatmapResiduals(const int track_id) {
    const LineTrack& track = reconstruction_.GetInitTrack(track_id);

    // compute heatmap samples
    std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
    ComputeHeatmapSamples(track, 
                          heatmap_samples,
                          std::make_pair(config_.sample_range_min, config_.sample_range_max),
                          config_.n_samples_heatmap);

    // compute line weights
    auto idmap = track.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track, weights);

    // add to problem for each supporting image (for each supporting line)
    auto& inf_line = reconstruction_.lines_[track_id];
    std::vector<int> image_ids = track.GetSortedImageIds();
    ceres::LossFunction* loss_function = config_.heatmap_loss_function.get();
    for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
        int img_id = *it1;
        int model_id = reconstruction_.imagecols_.camview(img_id).cam.ModelId();
        const auto& ids = idmap.at(img_id);
        for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
            const auto& samples = heatmap_samples.at(*it2);
            double weight = weights[*it2] * config_.heatmap_multiplier / double(config_.n_samples_heatmap / 10.0);
            ceres::CostFunction* cost_function = nullptr;

            switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::MaxHeatmapFunctor<CameraModel, DTYPE>::Create(p_heatmaps_itp_[img_id], samples); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, loss_function, 
                                                                         inf_line.uvec.data(), inf_line.wvec.data(),
                                                                         reconstruction_.imagecols_.params_data(img_id), reconstruction_.imagecols_.qvec_data(img_id), reconstruction_.imagecols_.tvec_data(img_id));
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::AddFeatureConsistencyResiduals(const int track_id) {
    // compute fconsis samples
    std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>> fconsis_samples; // [referenece_image_id, infiniteline2d, {target_image_ids}]
    ComputeFConsistencySamples(reconstruction_.GetInitTrack(track_id),
                               reconstruction_.GetInitCameraMap(),
                               fconsis_samples,
                               std::make_pair(config_.sample_range_min, config_.sample_range_max),
                               config_.n_samples_feature);

    // add to problem for each sample
    auto& inf_line = reconstruction_.lines_[track_id];
    std::map<int, int> idmap = reconstruction_.GetInitTrack(track_id).GetIndexMapforSorted();
    int n_samples = fconsis_samples.size();
    for (int i = 0; i < n_samples; ++i) {
        ceres::LossFunction* loss_function = config_.fconsis_loss_function.get();

        const auto& sample_tp = fconsis_samples[i];
        int ref_image_id = std::get<0>(sample_tp);
        int ref_index = idmap.at(ref_image_id);
        CameraView view_ref = reconstruction_.imagecols_.camview(ref_image_id);
        const auto& sample = std::get<1>(sample_tp);
        std::vector<int> tgt_image_ids = std::get<2>(sample_tp);

        // get reference descriptor and compute ref residuals
        double weight = config_.fconsis_multiplier / (double(n_samples / 100.0) * double(tgt_image_ids.size() / 5.0));
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        double* ref_descriptor = NULL;
        if (config_.use_ref_descriptor) {
            ref_descriptor = new double[CHANNELS];
            V2D ref_intersection;
            GetIntersection2d<double>(inf_line.uvec.data(), inf_line.wvec.data(),
                                      view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
                                      sample.p.data(), sample.direc.data(),
                                      ref_intersection.data());
            p_patches_itp_[track_id][ref_index]->Evaluate(ref_intersection.data(), ref_descriptor);

            // source residuals
            ceres::CostFunction* cost_function = nullptr;

            switch (view_ref.cam.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::FeatureConsisSrcFunctor<CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[track_id][ref_index], sample, ref_descriptor);
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            ceres::LossFunction* ref_scaled_loss_function = new ceres::ScaledLoss(scaled_loss_function, config_.ref_multiplier, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, ref_scaled_loss_function, 
                    inf_line.uvec.data(), inf_line.wvec.data(), 
                    reconstruction_.imagecols_.params_data(ref_image_id), reconstruction_.imagecols_.qvec_data(ref_image_id), reconstruction_.imagecols_.tvec_data(ref_image_id));
        }

        // compute tgt residuals
        for (const int& tgt_image_id: tgt_image_ids) {
            int tgt_index = idmap.at(tgt_image_id);
            CameraView view_tgt = reconstruction_.imagecols_.camview(tgt_image_id);

            // switch 2x2 = 4 camera model configurations
            ceres::CostFunction* cost_function = nullptr;
            switch (view_ref.cam.ModelId()) {
                // reference camera model == colmap::SimplePinholeCameraModel
                case colmap::SimplePinholeCameraModel::kModelId:
                    switch (view_tgt.cam.ModelId()) {

#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::FeatureConsisTgtFunctor<colmap::SimplePinholeCameraModel, CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[track_id][ref_index], p_patches_itp_[track_id][tgt_index], sample, ref_descriptor);  \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE

                    }
                // reference camera model == colmap::PinholeCameraModel
                case colmap::PinholeCameraModel::kModelId:
                    switch (view_tgt.cam.ModelId()) {

#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::FeatureConsisTgtFunctor<colmap::PinholeCameraModel, CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[track_id][ref_index], p_patches_itp_[track_id][tgt_index], sample, ref_descriptor);  \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE

                    }
                default:
                    LIMAP_CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
            }

            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                         inf_line.uvec.data(), inf_line.wvec.data(), 
                                                                         reconstruction_.imagecols_.params_data(ref_image_id), reconstruction_.imagecols_.qvec_data(ref_image_id), reconstruction_.imagecols_.tvec_data(ref_image_id),
                                                                         reconstruction_.imagecols_.params_data(tgt_image_id), reconstruction_.imagecols_.qvec_data(tgt_image_id), reconstruction_.imagecols_.tvec_data(tgt_image_id));
        }
        if (config_.use_ref_descriptor)
            delete ref_descriptor;
    }
    return;
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::AddResiduals(const int track_id) {
    if (config_.use_geometric)
        AddGeometricResiduals(track_id);
    if (enable_vp)
        AddVPResiduals(track_id);
    if (enable_heatmap)
        AddHeatmapResiduals(track_id);
    if (enable_feature)
        AddFeatureConsistencyResiduals(track_id);
}

template <typename DTYPE, int CHANNELS>
void LineBAEngine<DTYPE, CHANNELS>::SetUp() {
    // setup problem
    problem_.reset(new ceres::Problem(config_.problem_options));

    // add residual for each track
    int n_tracks = reconstruction_.NumTracks();
    for (int track_id = 0; track_id < n_tracks; ++track_id)
        AddResiduals(track_id);
    
    // parameterization
    ParameterizeCameras();
    ParameterizeLines();
}

template <typename DTYPE, int CHANNELS>
bool LineBAEngine<DTYPE, CHANNELS>::Solve() {
    if (problem_->NumResiduals() == 0)
        return false;
    ceres::Solver::Options solver_options = config_.solver_options;
    
    // Empirical choice.
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 900;
    const size_t num_images = reconstruction_.NumImages();
    if (num_images <= kMaxNumImagesDirectDenseSolver) {
        solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_threads);
    #if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
    #endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem_.get(), &summary_);
    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (config_.print_summary) {
        colmap::PrintHeading2("Optimization report");
        colmap::PrintSolverSummary(summary_); // We need to replace this with our own Printer!!!
    }
    return true;
}

template <typename DTYPE, int CHANNELS>
std::vector<std::vector<V2D>> LineBAEngine<DTYPE, CHANNELS>::GetHeatmapIntersections(const LineReconstruction& reconstruction) const
{
    size_t n_tracks = reconstruction.NumTracks();
    size_t n_images = reconstruction.NumImages();
    
    std::vector<std::vector<V2D>> out_samples;
    out_samples.resize(n_images);
    for (size_t track_id = 0; track_id < n_tracks; ++track_id) {
        const auto& inf_line = reconstruction.lines_[track_id];
        // compute heatmap samples
        std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
        ComputeHeatmapSamples(reconstruction.GetInitTrack(track_id), 
                              heatmap_samples,
                              std::make_pair(config_.sample_range_min, config_.sample_range_max),
                              config_.n_samples_heatmap);
        auto idmap = reconstruction.GetInitTrack(track_id).GetIdMap();

        std::vector<int> image_ids = reconstruction.GetInitTrack(track_id).GetSortedImageIds();
        for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
            int img_id = *it;
            const auto& view = reconstruction.imagecols_.camview(img_id);
            const auto& ids = idmap.at(img_id);
            std::vector<V2D> out_samples_idx;
            for (auto it1 = ids.begin(); it1 != ids.end(); ++it1) {
                const auto& samples = heatmap_samples.at(*it1);
                for (auto it2 = samples.begin(); it2 != samples.end(); ++it2) {
                    V2D intersection;
                    GetIntersection2d<double>(inf_line.uvec.data(), inf_line.wvec.data(), 
                                              view.cam.Params().data(), view.pose.qvec.data(), view.pose.tvec.data(),
                                              it2->p.data(), it2->direc.data(),
                                              intersection.data());
                    out_samples_idx.push_back(intersection);
                }
            }
            out_samples[img_id].insert(out_samples[img_id].end(), out_samples_idx.begin(), out_samples_idx.end());
        }
    }
    return out_samples;
}

// Register
// Only float16 can be supported (due to limited memory) for the full bundle adjustment
template class LineBAEngine<float16, 128>;

} // namespace line_bundle_adjustment 

} // namespace optimize 

} // namespace limap

