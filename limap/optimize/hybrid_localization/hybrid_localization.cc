#include "optimize/hybrid_localization/hybrid_localization.h"
#include "ceresbase/parameterization.h"
#include "optimize/hybrid_localization/cost_functions.h"

#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>

namespace limap {

namespace optimize {

namespace hybrid_localization {

void LineLocEngine::ParameterizeCamera() {
  double *kvec_data = cam.params.data();
  double *qvec_data = campose.qvec.data();
  double *tvec_data = campose.tvec.data();

  // We do not optimize for intrinsics
  problem_->SetParameterBlockConstant(kvec_data);
  SetQuaternionManifold(problem_.get(), qvec_data);
}

void LineLocEngine::AddResiduals() {
  // ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.1);
  ceres::LossFunction *loss_function = config_.loss_function.get();

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

      ceres::CostFunction *cost_function = ReprojectionLineFunctor::Create(
          config_.cost_function, config_.cost_function_weight, l3d, l2d);
      ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, cam.params.data(),
          campose.qvec.data(), campose.tvec.data());
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
#endif // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (config_.print_summary) {
    colmap::PrintSolverSummary(summary_, "Optimization report");
  }
  return true;
}

void JointLocEngine::AddResiduals() {
  CHECK_EQ(static_cast<int>(this->cam.model_id),
           static_cast<int>(colmap::PinholeCameraModel::model_id));

  // ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.1);
  ceres::LossFunction *loss_function = this->config_.loss_function.get();

  double weight_lines = this->config_.weight_line;
  double weight_points = this->config_.weight_point;

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

      ceres::CostFunction *cost_function = ReprojectionLineFunctor::Create(
          this->config_.cost_function, this->config_.cost_function_weight, l3d,
          l2d);
      ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = this->problem_->AddResidualBlock(
          cost_function, scaled_loss_function, this->cam.params.data(),
          this->campose.qvec.data(), this->campose.tvec.data());
    }
  }

  for (int i = 0; i < this->p3ds.size(); i++) {
    ceres::LossFunction *weighted_loss_function = new ceres::ScaledLoss(
        loss_function, weight_points, ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::CostFunction *cost_function = ReprojectionPointFunctor::Create(
        this->p3ds[i], this->p2ds[i], this->config_.points_3d_dist);
    ceres::ResidualBlockId block_id = this->problem_->AddResidualBlock(
        cost_function, weighted_loss_function, this->cam.params.data(),
        this->campose.qvec.data(), this->campose.tvec.data());
  }
}

} // namespace hybrid_localization

} // namespace optimize

} // namespace limap
