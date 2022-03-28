#include "lineKA/lineka.h"
#include "lineKA/cost_functions.h"
#include "triangulation/build_initial_graph.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

namespace limap {

namespace lineKA {

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::Initialize(const std::vector<std::vector<Line2d>>& all_lines_2d,
                                               const std::vector<CameraView>& cameras) 
{
    // initialize cameras
    for (auto it = cameras.begin(); it != cameras.end(); ++it) {
        cameras_matrixform_.push_back(*it);
        cameras_.push_back(MinimalPinholeCamera(*it));
    }

    // initialize lines
    size_t n_images = all_lines_2d.size();
    all_lines_2dof_.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        const auto& lines = all_lines_2d[img_id];
        for (auto it = lines.begin(); it != lines.end(); ++it) {
            if (config_.line2d_type == "fixedprojection")
                all_lines_2dof_[img_id].push_back(std::make_shared<Line2d_2DOF>(Line2d_2DOF(*it, "fixedprojection")));
            else if (config_.line2d_type == "fixedlength")
                all_lines_2dof_[img_id].push_back(std::make_shared<Line2d_2DOF>(Line2d_2DOF(*it, "fixedlength")));
            else
                throw std::runtime_error("line2d parameterization type not supported");
        }
    }
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::InitializeMatches(const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                                                      const std::vector<std::vector<int>>& neighbors) {
    triangulation::BuildInitialGraph(graph_, all_matches, neighbors);
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::InitializeHeatmaps(const std::vector<Eigen::MatrixXd>& heatmaps) {
    enable_heatmap = true;
    THROW_CHECK_EQ(cameras_.size(), heatmaps.size());
    auto& interp_cfg = config_.heatmap_interpolation_config;

    int n_images = heatmaps.size();
    p_heatmaps_f_.clear();
    p_heatmaps_itp_.clear();
    double memGB = 0;
    for (auto it = heatmaps.begin(); it != heatmaps.end(); ++it) {
        p_heatmaps_f_.push_back(FeatureMap<DTYPE>(*it));
        p_heatmaps_itp_.push_back(
                std::unique_ptr<FeatureInterpolator<DTYPE, 1>>
                    (new FeatureInterpolator<DTYPE, 1>(interp_cfg, p_heatmaps_f_.back()))
        );
        memGB += p_heatmaps_f_.back().MemGB();
    }
    std::cout<<"[INFO] Initialize heatmaps (n_images = "<<n_images<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::InitializePatches(const std::vector<std::vector<PatchInfo<DTYPE>>>& patchinfos) {
    enable_feature = true;
    THROW_CHECK_EQ(cameras_.size(), patchinfos.size());
    auto& interp_cfg = config_.feature_interpolation_config;

    int n_images = patchinfos.size();
    p_patches_.clear(); p_patches_.resize(n_images);
    p_patches_f_.clear(); p_patches_f_.resize(n_images);
    p_patches_itp_.clear(); p_patches_itp_.resize(n_images);

    double memGB = 0;
    int n_patches = 0;
    for (int img_id = 0; img_id < n_images; ++img_id) {
        const auto& patches = patchinfos[img_id];
        for (auto it = patches.begin(); it != patches.end(); ++it) {
            p_patches_[img_id].push_back(*it);
            p_patches_f_[img_id].push_back(FeaturePatch<DTYPE>(*it));
            p_patches_itp_[img_id].push_back(
                    std::unique_ptr<PatchInterpolator<DTYPE, CHANNELS>>
                        (new PatchInterpolator<DTYPE, CHANNELS>(interp_cfg, p_patches_f_[img_id].back()))
            );
            memGB += p_patches_f_[img_id].back().MemGB();
            n_patches += 1;
        }
    }
    std::cout<<"[INFO] Initialize features as patches (n_patches = "<<n_patches<<"): "<<memGB<<" GB"<<std::endl;
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::ParameterizeLines() {
    return;
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::AddHeatmapResiduals(const int img_id) {
    // compute t_array
    int n_samples = config_.n_samples_heatmap;
    double interval = (config_.sample_range_min - config_.sample_range_max) / (n_samples - 1);
    std::vector<double> t_array(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double val = config_.sample_range_min + interval * i;
        t_array[i] = val;
    }

    // add to problem for each line
    size_t n_lines = all_lines_2dof_[img_id].size();
    for (size_t line_id = 0; line_id < n_lines; ++line_id) {
        ceres::CostFunction* cost_function = nullptr;
        ceres::LossFunction* loss_function = config_.heatmap_loss_function.get();

        auto& line = all_lines_2dof_[img_id][line_id];
        cost_function = MaxHeatmap2DFunctor<DTYPE>::Create(p_heatmaps_itp_[img_id], line, t_array);
        ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, loss_function, line->GetVars().data());
    }
}

template <typename DTYPE, int CHANNELS>
std::vector<double> LineKAEngine<DTYPE, CHANNELS>::check_samples_epipolar(const std::shared_ptr<Line2d_2DOF>& ref_line, const MinimalPinholeCamera& cam_ref,
                                                                          const std::shared_ptr<Line2d_2DOF>& tgt_line, const MinimalPinholeCamera& cam_tgt,
                                                                          const std::vector<double> t_array) const
{
    std::vector<double> t_array_local;
    for (auto it = t_array.begin(); it != t_array.end(); ++it) {
        double val = *it;
        V2D ref_point = ref_line->GetPoint(val);

        // get epipolar line
        V3D epiline_coord;
        GetEpipolarLineCoordinate(cam_ref.kvec.data(), cam_ref.qvec.data(), cam_ref.tvec.data(),
                                  cam_tgt.kvec.data(), cam_tgt.qvec.data(), cam_tgt.tvec.data(),
                                  ref_point.data(), epiline_coord.data());
        Line2d tgt_line2d = tgt_line->GetOriginalLine();
        V3D coord = InfiniteLine2d(tgt_line2d).GetLineCoordinate();
        V2D tgt_point;
        CeresIntersect_LineCoordinates(epiline_coord.data(), coord.data(), tgt_point.data());

        if ((tgt_point - tgt_line2d.start).dot(tgt_line2d.direction()) < 0)
            continue;
        double tgt_val = (tgt_point - tgt_line2d.start).norm() / tgt_line2d.length();
        if (tgt_val < config_.sample_range_min || tgt_val > config_.sample_range_max)
            continue;
        t_array_local.push_back(val);
    }
    return t_array_local;
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::AddFeatureConsistencyResiduals() {
    // Add all feature consistency constraints in graph_
    // compute t_array
    int n_samples = config_.n_samples_feature_2d;
    double interval = (config_.sample_range_min - config_.sample_range_max) / (n_samples - 1);
    std::vector<double> t_array(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        double val = config_.sample_range_min + interval * i;
        t_array[i] = val;
    }

    size_t n_edges = graph_.directed_edges.size();
    for (size_t edge_id = 0; edge_id < n_edges; ++edge_id) {
        Edge* e = graph_.directed_edges[edge_id];
        int ref_img_id = graph_.nodes[e->node_idx1]->image_idx;
        int ref_line_id = graph_.nodes[e->node_idx1]->line_idx;
        int tgt_img_id = graph_.nodes[e->node_idx2]->image_idx;
        int tgt_line_id = graph_.nodes[e->node_idx2]->line_idx;

        auto& ref_line = all_lines_2dof_[ref_img_id][ref_line_id];
        auto& cam_ref = cameras_[ref_img_id];
        auto& tgt_line = all_lines_2dof_[tgt_img_id][tgt_line_id];
        auto& cam_tgt = cameras_[tgt_img_id];
        double weight = 1.0; // TODO: use weight from matcher

        // check intersections in advance
        std::vector<double> t_array_local = check_samples_epipolar(ref_line, cam_ref, tgt_line, cam_tgt, t_array);
        if (t_array_local.size() == 0)
            continue;

        ceres::CostFunction* cost_function = nullptr;
        ceres::LossFunction* loss_function = config_.fconsis_loss_function.get();
        cost_function = FeatureConsis2DFunctor<PatchInterpolator<DTYPE, CHANNELS>, CHANNELS>::Create(p_patches_itp_[ref_img_id][ref_line_id], ref_line, cam_ref, 
                                                                                                     p_patches_itp_[tgt_img_id][tgt_line_id], tgt_line, cam_tgt, 
                                                                                                     t_array_local, weight);
        ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, loss_function, ref_line->GetVars().data(), tgt_line->GetVars().data());
    }
}

template <typename DTYPE, int CHANNELS>
void LineKAEngine<DTYPE, CHANNELS>::SetUp() {
    // setup problem
    problem_.reset(new ceres::Problem(config_.problem_options));

    // add residuals for heatmap
    if (enable_heatmap) {
        int n_images = cameras_.size();
        for (int img_id = 0; img_id < n_images; ++img_id)
            AddHeatmapResiduals(img_id);
    }
    if (enable_feature)
        AddFeatureConsistencyResiduals();
    
    // parameterization
    ParameterizeLines();
}

template <typename DTYPE, int CHANNELS>
bool LineKAEngine<DTYPE, CHANNELS>::Solve() {
    if (problem_->NumResiduals() == 0)
        return false;
    ceres::Solver::Options solver_options = config_.solver_options;
    
    // Empirical choice.
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 900;
    const size_t num_images = cameras_.size();
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
std::vector<std::vector<Line2d>> LineKAEngine<DTYPE, CHANNELS>::GetOutputLines() const {
    std::vector<std::vector<Line2d>> all_lines_2d;
    size_t n_images = all_lines_2dof_.size();
    all_lines_2d.resize(n_images);
    for (size_t img_id = 0; img_id < n_images; ++img_id) {
        auto& lines = all_lines_2dof_[img_id];
        for (auto it = lines.begin(); it != lines.end(); ++it) {
            all_lines_2d[img_id].push_back((*it)->GetLine());
        }
    }
    return all_lines_2d;
}

// Only float16 can be supported (due to limited memory) for the keyline adjustment
#define REGISTER_CHANNEL(CHANNELS) \
    template class LineKAEngine<float16, CHANNELS>; \

REGISTER_CHANNEL(1);
REGISTER_CHANNEL(3);
REGISTER_CHANNEL(128);

#undef REGISTER_CHANNEL

} // namespace lineKA 

} // namespace limap

