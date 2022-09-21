#include "optimize/line_bundle_adjustment/lineba.h"
#include "optimize/line_refinement/cost_functions.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

#ifdef INTERPOLATION_ENABLED
#include "optimize/line_bundle_adjustment/pixel_lineba.cc"
#endif // INTERPOLATION_ENABLED

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
void LineBAEngine<DTYPE, CHANNELS>::AddResiduals(const int track_id) {
    if (config_.use_geometric)
        AddGeometricResiduals(track_id);
    if (enable_vp)
        AddVPResiduals(track_id);
#ifdef INTERPOLATION_ENABLED
    if (enable_heatmap)
        AddHeatmapResiduals(track_id);
    if (enable_feature)
        AddFeatureConsistencyResiduals(track_id);
#endif // INTERPOLATION_ENABLED
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

// Register
// Only float16 can be supported (due to limited memory) for the full bundle adjustment
template class LineBAEngine<float16, 128>;

} // namespace line_bundle_adjustment 

} // namespace optimize 

} // namespace limap

