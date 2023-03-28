#include "optimize/line_localization/lineloc.h"
#include "optimize/line_localization/cost_functions.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>

namespace limap {

namespace optimize {

namespace line_localization {

void LineLocEngine::ParameterizeCamera() {
    double* kvec_data = cam.Params().data();
    double* qvec_data = campose.qvec.data();
    double* tvec_data = campose.tvec.data();

    // We do not optimize for intrinsics
    problem_->SetParameterBlockConstant(kvec_data);

#ifdef CERES_PARAMETERIZATION_ENABLED
    ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
    problem_->SetParameterization(qvec_data, quaternion_parameterization);
#else
    ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold;
    problem_->SetManifold(qvec_data, quaternion_manifold);
#endif
}

void LineLocEngine::AddResiduals() {
    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.1);
    ceres::LossFunction* loss_function = config_.loss_function.get();

    // add to problem for each pair of lines
    for (int i = 0; i < l3ds.size(); i++) {
        const Line3d &l3d = l3ds[i];

        int n_lines = l2ds[i].size();
        double sum_lengths = 0;
        for (auto &l2d : l2ds[i]) 
            sum_lengths += l2d.length();

        for (auto &l2d : l2ds[i]) {
            double length = l2d.length();
            double weight = length / sum_lengths;

            ceres::CostFunction* cost_function = ReprojectionLineFunctor::Create(config_.cost_function, config_.cost_function_weight, l3d, l2d);
            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                        cam.Params().data(), campose.qvec.data(), campose.tvec.data());
        }
    }
}

void LineLocEngine::SetUp() {
    // setup problem
    problem_.reset(new ceres::Problem(config_.problem_options));

    AddResiduals();
    
    ParameterizeCamera();
}

bool LineLocEngine::Solve() {
    if (problem_->NumResiduals() == 0)
        return false;
    ceres::Solver::Options solver_options = config_.solver_options;
    
    solver_options.linear_solver_type = ceres::DENSE_QR;

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

void JointLocEngine::AddResiduals() {
    CHECK_EQ(this->cam.ModelId(), colmap::PinholeCameraModel::model_id);

    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.1);
    ceres::LossFunction* loss_function = this->config_.loss_function.get();

    int num_lines = this->l3ds.size();
    int num_points = this->p3ds.size();
    double weight_lines = double(num_points) / (num_lines + num_points);
    double weight_points = 1 - weight_lines;
    if (!this->config_.normalize_weight)
        weight_lines = weight_points = 1.0;
    weight_lines *= this->config_.weight_line;
    weight_points *= this->config_.weight_point;
    
    // add to problem for each pair of lines
    for (int i = 0; i < this->l3ds.size(); i++) {
        const Line3d &l3d = this->l3ds[i];

        int n_lines = this->l2ds[i].size();
        double sum_lengths = 0;
        for (auto &l2d : this->l2ds[i]) 
            sum_lengths += l2d.length();

        for (auto &l2d : this->l2ds[i]) {
            double length = l2d.length();
            double weight = weight_lines * length / sum_lengths;

            ceres::CostFunction* cost_function = ReprojectionLineFunctor::Create(this->config_.cost_function, this->config_.cost_function_weight, l3d, l2d);
            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            ceres::ResidualBlockId block_id = this->problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                        this->cam.Params().data(), this->campose.qvec.data(), this->campose.tvec.data());
        }
    }

    for (int i = 0; i < this->p3ds.size(); i++) {
        ceres::LossFunction* weighted_loss_function = new ceres::ScaledLoss(loss_function, weight_points, ceres::DO_NOT_TAKE_OWNERSHIP);
        ceres::CostFunction* cost_function = ReprojectionPointFunctor::Create(this->p3ds[i], this->p2ds[i], this->config_.points_3d_dist);
        ceres::ResidualBlockId block_id = this->problem_->AddResidualBlock(cost_function, weighted_loss_function, 
                                                                    this->cam.Params().data(), this->campose.qvec.data(), this->campose.tvec.data());
    }
}

} // namespace line_localization 

} // namespace optimize

} // namespace limap
