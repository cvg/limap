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

#ifdef INTERPOLATION_ENABLED

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& p_heatmaps) {
    enable_heatmap = true;
    int n_images = track_.count_images();
    THROW_CHECK_EQ(p_heatmaps.size(), n_images);
    for (int i = 0; i < n_images; ++i) {
        THROW_CHECK_EQ(p_camviews_[i].h(), p_heatmaps[i].rows());
        THROW_CHECK_EQ(p_camviews_[i].w(), p_heatmaps[i].cols());
    }
    
    auto& interp_cfg = config_.heatmap_interpolation_config;
    p_heatmaps_f_.clear();
    p_heatmaps_itp_.clear();
    double memGB = 0;
    for (auto it = p_heatmaps.begin(); it != p_heatmaps.end(); ++it) {
        p_heatmaps_f_.push_back(features::FeatureMap<DTYPE>(*it));
        p_heatmaps_itp_.push_back(
                std::unique_ptr<features::FeatureInterpolator<DTYPE, 1>>
                    (new features::FeatureInterpolator<DTYPE, 1>(interp_cfg, p_heatmaps_f_.back()))
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
        p_features_f_.push_back(features::FeatureMap<DTYPE>(*it));
        p_features_itp_.push_back(
                std::unique_ptr<features::FeatureInterpolator<DTYPE, CHANNELS>>
                    (new features::FeatureInterpolator<DTYPE, CHANNELS>(interp_cfg, p_features_f_.back()))
        );
        memGB += p_features_f_.back().MemGB();
    }
    if (config_.print_summary)
        std::cout<<"[INFO] Initialize features (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeFeaturesAsPatches(const std::vector<features::PatchInfo<DTYPE>>& patchinfos) {
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
        p_patches_f_.push_back(features::FeaturePatch<DTYPE>(*it));
        p_patches_itp_.push_back(
                std::unique_ptr<features::PatchInterpolator<DTYPE, CHANNELS>>
                    (new features::PatchInterpolator<DTYPE, CHANNELS>(interp_cfg, p_patches_f_.back()))
        );
        memGB += p_patches_f_.back().MemGB();
    }
    if (config_.print_summary)
        std::cout<<"[INFO] Initialize features as patches (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
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
        auto& view = p_camviews_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);
        for (auto it = ids.begin(); it != ids.end(); ++it) {
            const auto& samples = heatmap_samples.at(*it);
            double weight = weights[*it] * config_.heatmap_multiplier / double(config_.n_samples_heatmap / 10.0);
            ceres::CostFunction* cost_function = nullptr;

            switch (view.cam.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = MaxHeatmapFunctor<CameraModel, DTYPE>::Create(p_heatmaps_itp_[i], samples, view.cam.Params().data(), view.pose.qvec.data(), view.pose.tvec.data()); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

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
    std::map<int, CameraView> cameramap;
    for (int i = 0; i < n_images; ++i) {
        cameramap.insert(std::make_pair(image_ids[i], p_camviews_[i]));
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
        auto& view_ref = p_camviews_[ref_index];

        // get reference descriptor and compute ref residuals
        double weight = config_.fconsis_multiplier / (double(n_samples / 100.0) * double(tgt_image_ids.size() / 5.0));
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        double* ref_descriptor = NULL;
        if (config_.use_ref_descriptor) {
            ref_descriptor = new double[CHANNELS];
            V2D ref_intersection;
            double kvec_ref[4];
            ParamsToKvec<double>(view_ref.cam.ModelId(), view_ref.cam.Params().data(), kvec_ref);
            Ceres_GetIntersection2dFromInfiniteLine3d<double>(inf_line_.uvec.data(), inf_line_.wvec.data(), 
                                                              kvec_ref, view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
                                                              sample.coords.data(), ref_intersection.data());

            if (use_patches)
                p_patches_itp_[ref_index]->Evaluate(ref_intersection.data(), ref_descriptor);
            else
                p_features_itp_[ref_index]->Evaluate(ref_intersection.data(), ref_descriptor);

            // source residuals
            ceres::CostFunction* cost_function = nullptr;

            switch (view_ref.cam.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        if (use_patches) {             \
            cost_function = FeatureConsisSrcFunctor<CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_index], sample, ref_descriptor, \
                                                                                                              view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data()); \
        } \
        else { \
            cost_function = FeatureConsisSrcFunctor<CameraModel, features::FeatureInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_features_itp_[ref_index], sample, ref_descriptor, \
                                                                                                              view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data()); \
        } \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            ceres::LossFunction* src_scaled_loss_function = new ceres::ScaledLoss(scaled_loss_function, config_.ref_multiplier, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, src_scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }

        // compute tgt residuals
        for (const int& tgt_image_id: tgt_image_ids) {
            int tgt_index = idmap.at(tgt_image_id);
            auto& view_tgt = p_camviews_[tgt_index];
            // tgt residuals
            ceres::CostFunction* cost_function = nullptr;

            // switch 2x2 = 4 camera model configurations
            switch (view_ref.cam.ModelId()) {
                // reference camera model == colmap::SimplePinholeCameraModel
                case colmap::SimplePinholeCameraModel::kModelId:
                    switch (view_tgt.cam.ModelId()) {

#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        if (use_patches) {             \
           cost_function = FeatureConsisTgtFunctor<colmap::SimplePinholeCameraModel, CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_index], p_patches_itp_[tgt_index], sample, ref_descriptor,  \
                                                                                                         view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),  \
                                                                                                         view_tgt.cam.Params().data(), view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data()); \
        } \
        else { \
           cost_function = FeatureConsisTgtFunctor<colmap::SimplePinholeCameraModel, CameraModel, features::FeatureInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_features_itp_[ref_index], p_features_itp_[tgt_index], sample, ref_descriptor,  \
                                                                                                         view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data(), \
                                                                                                         view_tgt.cam.Params().data(), view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data()); \
        } \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE

                    }
                // reference camera model == colmap::PinholeCameraModel
                case colmap::PinholeCameraModel::kModelId:
                    switch (view_tgt.cam.ModelId()) {

#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        if (use_patches) {             \
           cost_function = FeatureConsisTgtFunctor<colmap::PinholeCameraModel, CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_index], p_patches_itp_[tgt_index], sample, ref_descriptor,  \
                                                                                                         view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),  \
                                                                                                         view_tgt.cam.Params().data(), view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data()); \
        } \
        else { \
           cost_function = FeatureConsisTgtFunctor<colmap::PinholeCameraModel, CameraModel, features::FeatureInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_features_itp_[ref_index], p_features_itp_[tgt_index], sample, ref_descriptor,  \
                                                                                                         view_ref.cam.Params().data(), view_ref.pose.qvec.data(), view_ref.pose.tvec.data(), \
                                                                                                         view_tgt.cam.Params().data(), view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data()); \
        } \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE

                    }
                default:
                    LIMAP_CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
            }

            ceres::ResidualBlockId block_id_tgt = problem_->AddResidualBlock(cost_function, scaled_loss_function, inf_line_.uvec.data(), inf_line_.wvec.data());
        }
        if (config_.use_ref_descriptor)
            delete ref_descriptor;
    }
    return;
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
        const auto& view = p_camviews_[i];
        int img_id = image_ids[i];
        const auto& ids = idmap.at(img_id);

        std::vector<V2D> out_samples_idx;
        for (auto it1 = ids.begin(); it1 != ids.end(); ++it1) {
            const auto& samples = heatmap_samples.at(*it1);
            for (auto it2 = samples.begin(); it2 != samples.end(); ++it2) {
                V2D intersection;
                double kvec[4];
                ParamsToKvec<double>(view.cam.ModelId(), view.cam.Params().data(), kvec);
                Ceres_GetIntersection2dFromInfiniteLine3d<double>(inf_line.uvec.data(), inf_line.wvec.data(), 
                                                                  kvec, view.pose.qvec.data(), view.pose.tvec.data(),
                                                                  it2->coords.data(), intersection.data());
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
    std::map<int, CameraView> cameramap;
    for (int i = 0; i < n_images; ++i) {
        cameramap.insert(std::make_pair(image_ids[i], p_camviews_[i]));
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
        const auto& view_ref = p_camviews_[ref_index];

        // get ref_intersection
        std::vector<std::pair<int, V2D>> out_samples_idx;
        V2D ref_intersection;
        double kvec_ref[4];
        ParamsToKvec<double>(view_ref.cam.ModelId(), view_ref.cam.Params().data(), kvec_ref);
        Ceres_GetIntersection2dFromInfiniteLine3d<double>(inf_line.uvec.data(), inf_line.wvec.data(),
                                                          kvec_ref, view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
                                                          sample.coords.data(), ref_intersection.data());
        out_samples_idx.push_back(std::make_pair(ref_index, ref_intersection));

        // get tgt_intersection for each target image
        for (const int& tgt_image_id: tgt_image_ids) {
            int tgt_index = idmap.at(tgt_image_id);
            const auto& view_tgt = p_camviews_[tgt_index];
            V3D epiline_coord;
            V2D tgt_intersection;
            double kvec_ref[4], kvec_tgt[4];
            ParamsToKvec<double>(view_ref.cam.ModelId(), view_ref.cam.Params().data(), kvec_ref);
            ParamsToKvec<double>(view_tgt.cam.ModelId(), view_tgt.cam.Params().data(), kvec_tgt);
            GetEpipolarLineCoordinate<double>(kvec_ref, view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
                                              kvec_tgt, view_ref.pose.qvec.data(), view_tgt.pose.tvec.data(),
                                              ref_intersection.data(), epiline_coord.data());
            Ceres_GetIntersection2dFromInfiniteLine3d<double>(inf_line.uvec.data(), inf_line.wvec.data(),
                                                              kvec_tgt, view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data(),
                                                              epiline_coord.data(), tgt_intersection.data());
            out_samples_idx.push_back(std::make_pair(tgt_index, tgt_intersection));
        }
        out_samples.push_back(out_samples_idx);
    }
    return out_samples;
}

#endif // INTERPOLATION_ENABLED

} // namespace line_refinement

} // namespace optimize

} // namespace limap

