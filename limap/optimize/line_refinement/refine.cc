#include "optimize/line_refinement/refine.h"
#include "base/camera_models.h"
#include "ceresbase/line_projection.h"
#include "ceresbase/parameterization.h"
#include "optimize/line_refinement/cost_functions.h"

#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>

namespace limap {

namespace optimize {

namespace line_refinement {

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::Initialize(
    const LineTrack &track, const std::vector<CameraView> &p_camviews) {
  // validity check
  track_ = track;
  p_camviews_ = p_camviews;

  // initialize optimized line
  inf_line_ = MinimalInfiniteLine3d(track_.line);
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeVPs(
    const std::vector<vplib::VPResult> &p_vpresults) {
  enable_vp = true;
  p_vpresults_ = p_vpresults;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::ParameterizeMinimalLine() {
  double *uvec_data = inf_line_.uvec.data();
  double *wvec_data = inf_line_.wvec.data();
  SetQuaternionManifold(problem_.get(), uvec_data);
  SetSphereManifold<2>(problem_.get(), wvec_data);
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
    ceres::LossFunction *loss_function =
        config_.line_geometric_loss_function.get();

    auto &view = p_camviews_[i];
    int img_id = image_ids[i];
    const auto &ids = idmap.at(img_id);
    for (auto it = ids.begin(); it != ids.end(); ++it) {
      const Line2d &line = track_.line2d_list[*it];
      double weight = weights[*it];
      ceres::CostFunction *cost_function = nullptr;

      switch (view.cam.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function = GeometricRefinementFunctor<CameraModel>::Create(           \
        line, view.cam.params.data(), view.pose.qvec.data(),                   \
        view.pose.tvec.data(), config_.geometric_alpha);                       \
    break;
        LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
      }

      ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, inf_line_.uvec.data(),
          inf_line_.wvec.data());
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
  ceres::LossFunction *loss_function = config_.vp_loss_function.get();
  for (int i = 0; i < n_images; ++i) {
    auto &view = p_camviews_[i];
    int img_id = image_ids[i];
    const auto &ids = idmap.at(img_id);
    for (auto it = ids.begin(); it != ids.end(); ++it) {
      int line_id = track_.line_id_list[*it];
      if (p_vpresults_[i].HasVP(line_id)) {
        const V3D &vp = p_vpresults_[i].GetVP(line_id);
        double weight = weights[*it] * config_.vp_multiplier;
        ceres::CostFunction *cost_function = nullptr;

        switch (view.cam.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function = VPConstraintsFunctor<CameraModel>::Create(                 \
        vp, view.cam.params.data(), view.pose.qvec.data());                    \
    break;
          LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }

        ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
            loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
            cost_function, scaled_loss_function, inf_line_.uvec.data(),
            inf_line_.wvec.data());
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
#endif // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  state_collector_ = RefinementCallback(problem_.get());
  solver_options.callbacks.push_back(&state_collector_);
  solver_options.update_state_every_iteration = true;
  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (config_.print_summary) {
    colmap::PrintSolverSummary(summary_, "Optimization report");
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
  // Line3d line = GetLineSegmentFromInfiniteLine3d(inf_line_.GetInfiniteLine(),
  // cameras, track_.line2d_list, config_.num_outliers_aggregate);
  Line3d line = GetLineSegmentFromInfiniteLine3d(
      inf_line_.GetInfiniteLine(), track_.line3d_list,
      config_.num_outliers_aggregate);
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
  const auto &states = state_collector_.states;
  for (auto it = states.begin(); it != states.end(); ++it) {
    MinimalInfiniteLine3d minimal_inf_line = MinimalInfiniteLine3d(*it);
    Line3d line3d = GetLineSegmentFromInfiniteLine3d(
        minimal_inf_line.GetInfiniteLine(), track_.line3d_list,
        config_.num_outliers_aggregate);
    line3ds.push_back(line3d);
  }
  return line3ds;
}

///////////////////////////////////////////////////////////////////
// Pixel-wise optimization with ceres interpolation
// - Heatmap optimization
// - Featuremetric consistency optimization
///////////////////////////////////////////////////////////////////

#ifdef INTERPOLATION_ENABLED

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeHeatmaps(
    const std::vector<Eigen::MatrixXd> &p_heatmaps) {
  enable_heatmap = true;
  int n_images = track_.count_images();
  THROW_CHECK_EQ(p_heatmaps.size(), n_images);
  for (int i = 0; i < n_images; ++i) {
    THROW_CHECK_EQ(p_camviews_[i].h(), p_heatmaps[i].rows());
    THROW_CHECK_EQ(p_camviews_[i].w(), p_heatmaps[i].cols());
  }

  auto &interp_cfg = config_.heatmap_interpolation_config;
  p_heatmaps_f_.clear();
  p_heatmaps_itp_.clear();
  double memGB = 0;
  for (auto it = p_heatmaps.begin(); it != p_heatmaps.end(); ++it) {
    p_heatmaps_f_.push_back(features::FeatureMap<DTYPE>(*it));
    p_heatmaps_itp_.push_back(
        std::unique_ptr<features::FeatureInterpolator<DTYPE, 1>>(
            new features::FeatureInterpolator<DTYPE, 1>(interp_cfg,
                                                        p_heatmaps_f_.back())));
    memGB += p_heatmaps_f_.back().MemGB();
  }
  if (config_.print_summary)
    std::cout << "[INFO] Initialize heatmaps (n_images = " << n_images
              << "): " << memGB << " GB" << std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeFeatures(
    const std::vector<py::array_t<DTYPE, py::array::c_style>> &p_features) {
  enable_feature = true;
  int n_images = track_.count_images();
  THROW_CHECK_EQ(p_features.size(), n_images);

  auto &interp_cfg = config_.feature_interpolation_config;
  p_features_.clear();
  p_features_f_.clear();
  p_features_itp_.clear();
  double memGB = 0;
  for (auto it = p_features.begin(); it != p_features.end(); ++it) {
    p_features_.push_back(*it);
    p_features_f_.push_back(features::FeatureMap<DTYPE>(*it));
    p_features_itp_.push_back(
        std::unique_ptr<features::FeatureInterpolator<DTYPE, CHANNELS>>(
            new features::FeatureInterpolator<DTYPE, CHANNELS>(
                interp_cfg, p_features_f_.back())));
    memGB += p_features_f_.back().MemGB();
  }
  if (config_.print_summary)
    std::cout << "[INFO] Initialize features (n_images = " << n_images
              << "): " << memGB << " GB" << std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::InitializeFeaturesAsPatches(
    const std::vector<features::PatchInfo<DTYPE>> &patchinfos) {
  enable_feature = true;
  use_patches = true;
  int n_images = track_.count_images();
  THROW_CHECK_EQ(patchinfos.size(), n_images);

  auto &interp_cfg = config_.feature_interpolation_config;
  p_patches_.clear();
  p_patches_f_.clear();
  p_patches_itp_.clear();
  double memGB = 0;
  for (auto it = patchinfos.begin(); it != patchinfos.end(); ++it) {
    p_patches_.push_back(*it);
    p_patches_f_.push_back(features::FeaturePatch<DTYPE>(*it));
    p_patches_itp_.push_back(
        std::unique_ptr<features::PatchInterpolator<DTYPE, CHANNELS>>(
            new features::PatchInterpolator<DTYPE, CHANNELS>(
                interp_cfg, p_patches_f_.back())));
    memGB += p_patches_f_.back().MemGB();
  }
  if (config_.print_summary)
    std::cout << "[INFO] Initialize features as patches (n_images = "
              << n_images << "): " << memGB << " GB" << std::endl;
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddHeatmapResiduals() {
  // compute heatmap samples
  std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
  ComputeHeatmapSamples(
      track_, heatmap_samples,
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
    ceres::LossFunction *loss_function = config_.heatmap_loss_function.get();
    auto &view = p_camviews_[i];
    int img_id = image_ids[i];
    const auto &ids = idmap.at(img_id);
    for (auto it = ids.begin(); it != ids.end(); ++it) {
      const auto &samples = heatmap_samples.at(*it);
      double weight = weights[*it] * config_.heatmap_multiplier /
                      double(config_.n_samples_heatmap / 10.0);
      ceres::CostFunction *cost_function = nullptr;

      switch (view.cam.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function = MaxHeatmapFunctor<CameraModel, DTYPE>::Create(             \
        p_heatmaps_itp_[i], samples, view.cam.params.data(),                   \
        view.pose.qvec.data(), view.pose.tvec.data());                         \
    break;
        LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
      }

      ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, loss_function, inf_line_.uvec.data(),
          inf_line_.wvec.data());
    }
  }
}

template <typename DTYPE, int CHANNELS>
void RefinementEngine<DTYPE, CHANNELS>::AddFeatureConsistencyResiduals() {
  // compute fconsis samples
  std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>>
      fconsis_samples; // [referenece_image_id, infiniteline2d,
                       // {target_image_ids}]
  int n_images = track_.count_images();
  std::vector<int> image_ids = track_.GetSortedImageIds();
  std::map<int, CameraView> cameramap;
  for (int i = 0; i < n_images; ++i) {
    cameramap.insert(std::make_pair(image_ids[i], p_camviews_[i]));
  }

  ComputeFConsistencySamples(
      track_, cameramap, fconsis_samples,
      std::make_pair(config_.sample_range_min, config_.sample_range_max),
      config_.n_samples_feature);

  // add to problem for each sample
  std::map<int, int> idmap = track_.GetIndexMapforSorted();
  int n_samples = fconsis_samples.size();
  for (int i = 0; i < n_samples; ++i) {
    ceres::LossFunction *loss_function = config_.fconsis_loss_function.get();

    const auto &sample_tp = fconsis_samples[i];
    int ref_image_id = std::get<0>(sample_tp);
    int ref_index = idmap.at(ref_image_id);
    const auto &sample = std::get<1>(sample_tp);
    std::vector<int> tgt_image_ids = std::get<2>(sample_tp);
    auto &view_ref = p_camviews_[ref_index];

    // get reference descriptor and compute ref residuals
    double weight =
        config_.fconsis_multiplier /
        (double(n_samples / 100.0) * double(tgt_image_ids.size() / 5.0));
    ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
    double *ref_descriptor = NULL;
    if (config_.use_ref_descriptor) {
      ref_descriptor = new double[CHANNELS];
      V2D ref_intersection;
      double kvec_ref[4];
      ParamsToKvec<double>(view_ref.cam.model_id, view_ref.cam.params.data(),
                           kvec_ref);
      Ceres_GetIntersection2dFromInfiniteLine3d<double>(
          inf_line_.uvec.data(), inf_line_.wvec.data(), kvec_ref,
          view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
          sample.coords.data(), ref_intersection.data());

      if (use_patches)
        p_patches_itp_[ref_index]->Evaluate(ref_intersection.data(),
                                            ref_descriptor);
      else
        p_features_itp_[ref_index]->Evaluate(ref_intersection.data(),
                                             ref_descriptor);

      // source residuals
      ceres::CostFunction *cost_function = nullptr;

      switch (view_ref.cam.model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    if (use_patches) {                                                         \
      cost_function = FeatureConsisSrcFunctor<                                 \
          CameraModel, features::PatchInterpolator<DTYPE, CHANNELS>,           \
          CHANNELS>::Create(p_patches_itp_[ref_index], sample, ref_descriptor, \
                            view_ref.cam.params.data(),                        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data());                        \
    } else {                                                                   \
      cost_function = FeatureConsisSrcFunctor<                                 \
          CameraModel, features::FeatureInterpolator<DTYPE, CHANNELS>,         \
          CHANNELS>::Create(p_features_itp_[ref_index], sample,                \
                            ref_descriptor, view_ref.cam.params.data(),        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data());                        \
    }                                                                          \
    break;
        LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
      }

      ceres::LossFunction *src_scaled_loss_function =
          new ceres::ScaledLoss(scaled_loss_function, config_.ref_multiplier,
                                ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, src_scaled_loss_function, inf_line_.uvec.data(),
          inf_line_.wvec.data());
    }

    // compute tgt residuals
    for (const int &tgt_image_id : tgt_image_ids) {
      int tgt_index = idmap.at(tgt_image_id);
      auto &view_tgt = p_camviews_[tgt_index];
      // tgt residuals
      ceres::CostFunction *cost_function = nullptr;

      // switch 2x2 = 4 camera model configurations
      switch (view_ref.cam.model_id) {
      // reference camera model == colmap::SimplePinholeCameraModel
      case colmap::SimplePinholeCameraModel::model_id:
        switch (view_tgt.cam.model_id) {

#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    if (use_patches) {                                                         \
      cost_function = FeatureConsisTgtFunctor<                                 \
          colmap::SimplePinholeCameraModel, CameraModel,                       \
          features::PatchInterpolator<DTYPE, CHANNELS>,                        \
          CHANNELS>::Create(p_patches_itp_[ref_index],                         \
                            p_patches_itp_[tgt_index], sample, ref_descriptor, \
                            view_ref.cam.params.data(),                        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data(),                         \
                            view_tgt.cam.params.data(),                        \
                            view_tgt.pose.qvec.data(),                         \
                            view_tgt.pose.tvec.data());                        \
    } else {                                                                   \
      cost_function = FeatureConsisTgtFunctor<                                 \
          colmap::SimplePinholeCameraModel, CameraModel,                       \
          features::FeatureInterpolator<DTYPE, CHANNELS>,                      \
          CHANNELS>::Create(p_features_itp_[ref_index],                        \
                            p_features_itp_[tgt_index], sample,                \
                            ref_descriptor, view_ref.cam.params.data(),        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data(),                         \
                            view_tgt.cam.params.data(),                        \
                            view_tgt.pose.qvec.data(),                         \
                            view_tgt.pose.tvec.data());                        \
    }                                                                          \
    break;
          LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }
      // reference camera model == colmap::PinholeCameraModel
      case colmap::PinholeCameraModel::model_id:
        switch (view_tgt.cam.model_id) {

#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    if (use_patches) {                                                         \
      cost_function = FeatureConsisTgtFunctor<                                 \
          colmap::PinholeCameraModel, CameraModel,                             \
          features::PatchInterpolator<DTYPE, CHANNELS>,                        \
          CHANNELS>::Create(p_patches_itp_[ref_index],                         \
                            p_patches_itp_[tgt_index], sample, ref_descriptor, \
                            view_ref.cam.params.data(),                        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data(),                         \
                            view_tgt.cam.params.data(),                        \
                            view_tgt.pose.qvec.data(),                         \
                            view_tgt.pose.tvec.data());                        \
    } else {                                                                   \
      cost_function = FeatureConsisTgtFunctor<                                 \
          colmap::PinholeCameraModel, CameraModel,                             \
          features::FeatureInterpolator<DTYPE, CHANNELS>,                      \
          CHANNELS>::Create(p_features_itp_[ref_index],                        \
                            p_features_itp_[tgt_index], sample,                \
                            ref_descriptor, view_ref.cam.params.data(),        \
                            view_ref.pose.qvec.data(),                         \
                            view_ref.pose.tvec.data(),                         \
                            view_tgt.cam.params.data(),                        \
                            view_tgt.pose.qvec.data(),                         \
                            view_tgt.pose.tvec.data());                        \
    }                                                                          \
    break;
          LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }
      default:
        LIMAP_CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
      }

      ceres::ResidualBlockId block_id_tgt = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, inf_line_.uvec.data(),
          inf_line_.wvec.data());
    }
    if (config_.use_ref_descriptor)
      delete ref_descriptor;
  }
  return;
}

template <typename DTYPE, int CHANNELS>
std::vector<std::vector<V2D>>
RefinementEngine<DTYPE, CHANNELS>::GetHeatmapIntersections(
    const Line3d &line) const {
  MinimalInfiniteLine3d inf_line = MinimalInfiniteLine3d(line);
  int n_images = track_.count_images();
  std::vector<int> image_ids = track_.GetSortedImageIds();

  // compute heatmap samples
  std::vector<std::vector<InfiniteLine2d>> heatmap_samples;
  ComputeHeatmapSamples(
      track_, heatmap_samples,
      std::make_pair(config_.sample_range_min, config_.sample_range_max),
      config_.n_samples_heatmap);
  auto idmap = track_.GetIdMap();

  // compute intersections for each supporting image
  std::vector<std::vector<V2D>> out_samples;
  for (int i = 0; i < n_images; ++i) {
    const auto &view = p_camviews_[i];
    int img_id = image_ids[i];
    const auto &ids = idmap.at(img_id);

    std::vector<V2D> out_samples_idx;
    for (auto it1 = ids.begin(); it1 != ids.end(); ++it1) {
      const auto &samples = heatmap_samples.at(*it1);
      for (auto it2 = samples.begin(); it2 != samples.end(); ++it2) {
        V2D intersection;
        double kvec[4];
        ParamsToKvec<double>(view.cam.model_id, view.cam.params.data(), kvec);
        Ceres_GetIntersection2dFromInfiniteLine3d<double>(
            inf_line.uvec.data(), inf_line.wvec.data(), kvec,
            view.pose.qvec.data(), view.pose.tvec.data(), it2->coords.data(),
            intersection.data());
        out_samples_idx.push_back(intersection);
      }
    }
    out_samples.push_back(out_samples_idx);
  }
  return out_samples;
}

template <typename DTYPE, int CHANNELS>
std::vector<std::vector<std::pair<int, V2D>>>
RefinementEngine<DTYPE, CHANNELS>::GetFConsistencyIntersections(
    const Line3d &line) const {
  MinimalInfiniteLine3d inf_line = MinimalInfiniteLine3d(line);

  // compute fconsis samples
  std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>>
      fconsis_samples; // [referenece_image_id, infiniteline2d,
                       // {target_image_ids}]
  int n_images = track_.count_images();
  std::vector<int> image_ids = track_.GetSortedImageIds();
  std::map<int, CameraView> cameramap;
  for (int i = 0; i < n_images; ++i) {
    cameramap.insert(std::make_pair(image_ids[i], p_camviews_[i]));
  }

  ComputeFConsistencySamples(
      track_, cameramap, fconsis_samples,
      std::make_pair(config_.sample_range_min, config_.sample_range_max),
      config_.n_samples_feature);

  // compute intersections for each sample
  std::vector<std::vector<std::pair<int, V2D>>> out_samples;
  std::map<int, int> idmap = track_.GetIndexMapforSorted();
  for (auto &sample_tp : fconsis_samples) {
    int ref_image_id = std::get<0>(sample_tp);
    int ref_index = idmap.at(ref_image_id);
    const InfiniteLine2d &sample = std::get<1>(sample_tp);
    std::vector<int> tgt_image_ids = std::get<2>(sample_tp);
    const auto &view_ref = p_camviews_[ref_index];

    // get ref_intersection
    std::vector<std::pair<int, V2D>> out_samples_idx;
    V2D ref_intersection;
    double kvec_ref[4];
    ParamsToKvec<double>(view_ref.cam.model_id, view_ref.cam.params.data(),
                         kvec_ref);
    Ceres_GetIntersection2dFromInfiniteLine3d<double>(
        inf_line.uvec.data(), inf_line.wvec.data(), kvec_ref,
        view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
        sample.coords.data(), ref_intersection.data());
    out_samples_idx.push_back(std::make_pair(ref_index, ref_intersection));

    // get tgt_intersection for each target image
    for (const int &tgt_image_id : tgt_image_ids) {
      int tgt_index = idmap.at(tgt_image_id);
      const auto &view_tgt = p_camviews_[tgt_index];
      V3D epiline_coord;
      V2D tgt_intersection;
      double kvec_ref[4], kvec_tgt[4];
      ParamsToKvec<double>(view_ref.cam.model_id, view_ref.cam.params.data(),
                           kvec_ref);
      ParamsToKvec<double>(view_tgt.cam.model_id, view_tgt.cam.params.data(),
                           kvec_tgt);
      GetEpipolarLineCoordinate<double>(
          kvec_ref, view_ref.pose.qvec.data(), view_ref.pose.tvec.data(),
          kvec_tgt, view_ref.pose.qvec.data(), view_tgt.pose.tvec.data(),
          ref_intersection.data(), epiline_coord.data());
      Ceres_GetIntersection2dFromInfiniteLine3d<double>(
          inf_line.uvec.data(), inf_line.wvec.data(), kvec_tgt,
          view_tgt.pose.qvec.data(), view_tgt.pose.tvec.data(),
          epiline_coord.data(), tgt_intersection.data());
      out_samples_idx.push_back(std::make_pair(tgt_index, tgt_intersection));
    }
    out_samples.push_back(out_samples_idx);
  }
  return out_samples;
}

#endif // INTERPOLATION_ENABLED

template class RefinementEngine<float16, 128>;

} // namespace line_refinement

} // namespace optimize

} // namespace limap
