// Copyright (c) 2019, Torsten Sattler
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// author: Torsten Sattler, torsten.sattler.de@googlemail.com

#ifndef LIMAP_ESTIMATORS_POSE_PL_ABSOLUTE_POSE_RANSAC_H_
#define LIMAP_ESTIMATORS_POSE_PL_ABSOLUTE_POSE_RANSAC_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <RansacLib/sampling.h>
#include <RansacLib/utils.h>

namespace limap {

namespace estimators {

namespace absolute_pose {

using namespace ransac_lib;

enum MinimalSolverType { P3P, P2P1LL, P1P2LL, P3LL };

template <class Solver> class PointLineSolverFirstSampler {
public:
  PointLineSolverFirstSampler(const unsigned int random_seed,
                              const Solver &solver)
      : num_data_points(solver.num_data_points()),
        num_data_lines(solver.num_data_lines()),
        sample_size_(solver.min_sample_size()) {
    rng_.seed(random_seed);
    uniform_dstr_solver.param(std::uniform_int_distribution<int>::param_type(
        0, 4 - 1)); // Number of solver: 4
    uniform_dstr_point.param(
        std::uniform_int_distribution<int>::param_type(0, num_data_points - 1));
    uniform_dstr_line.param(std::uniform_int_distribution<int>::param_type(
        num_data_points, num_data_lines + num_data_points - 1));
  }

  // Draws minimal sample.
  void Sample(std::vector<int> *random_sample) {
    // Draw a solver first
    MinimalSolverType minimal_solver =
        static_cast<MinimalSolverType>(uniform_dstr_solver(rng_));

    switch (minimal_solver) {
    case P3P:
      DrawSample(random_sample, 3, 0);
      break;
    case P2P1LL:
      DrawSample(random_sample, 2, 1);
      break;
    case P1P2LL:
      DrawSample(random_sample, 1, 2);
      break;
    case P3LL:
      DrawSample(random_sample, 0, 3);
      break;
    }
  }

  void DrawSample(std::vector<int> *random_sample, int point_num,
                  int line_num) {
    std::vector<int> &sample = *random_sample;
    sample.resize(sample_size_);

    for (int i = 0; i < point_num; ++i) {
      bool found = true;
      while (found) {
        found = false;
        sample[i] = uniform_dstr_point(rng_);
        for (int j = 0; j < i; ++j)
          if (sample[j] == sample[i]) {
            found = true;
            break;
          }
      }
    }

    for (int i = point_num; i < sample_size_; ++i) {
      bool found = true;
      while (found) {
        found = false;
        sample[i] = uniform_dstr_line(rng_);
        for (int j = point_num; j < i; ++j)
          if (sample[j] == sample[i]) {
            found = true;
            break;
          }
      }
    }
  }

protected:
  // The random number generator used by RANSAC.
  std::mt19937 rng_;
  std::uniform_int_distribution<int> uniform_dstr_solver;
  std::uniform_int_distribution<int> uniform_dstr_point;
  std::uniform_int_distribution<int> uniform_dstr_line;
  int num_data_points, num_data_lines;
  int sample_size_;
};

// Our customized RANSAC based on LocallyOptimizedMSAC from RansacLib
// [LINK]
// https://github.com/tsattler/RansacLib/blob/master/RansacLib/ransac.h#L121
template <class Model, class ModelVector, class Solver,
          class Sampler = PointLineSolverFirstSampler<Solver>>
class PointLineAbsolutePoseRansac : public RansacBase {
public:
  // Estimates a model using a given solver. Notice that the solver contains
  // all data and is responsible to implement a non-minimal solver and
  // least-squares refinement. The latter two are optional, i.e., a dummy
  // implementation returning false is sufficient.
  // Returns the number of inliers.
  int EstimateModel(const LORansacOptions &options, const Solver &solver,
                    Model *best_model, RansacStatistics *statistics) const {
    ResetStatistics(statistics);
    RansacStatistics &stats = *statistics;

    // Sanity check: No need to run RANSAC if there are not enough data
    // points.
    const int kMinSampleSize = solver.min_sample_size();
    const int kNumData = solver.num_data();
    if (kMinSampleSize > kNumData || kMinSampleSize <= 0) {
      return 0;
    }

    // Initializes variables, etc.
    Sampler sampler(options.random_seed_, solver);
    std::mt19937 rng;
    rng.seed(options.random_seed_);

    uint32_t max_num_iterations =
        std::max(options.max_num_iterations_, options.min_num_iterations_);

    const double kSqrInlierThresh = options.squared_inlier_threshold_;

    Model best_minimal_model;
    double best_min_model_score = std::numeric_limits<double>::max();

    std::vector<int> minimal_sample(kMinSampleSize);
    ModelVector estimated_models;

    // Runs random sampling.
    for (stats.num_iterations = 0u; stats.num_iterations < max_num_iterations;
         ++stats.num_iterations) {
      // As proposed by Lebeda et al., Local Optimization is not executed in
      // the first lo_starting_iterations_ iterations. We thus run LO on the
      // best model found so far once we reach this iteration.
      if (stats.num_iterations == options.lo_starting_iterations_ &&
          best_min_model_score < std::numeric_limits<double>::max()) {
        ++stats.number_lo_iterations;
        LocalOptimization(options, solver, &rng, best_model,
                          &(stats.best_model_score));

        // Updates the number of RANSAC iterations.
        stats.best_num_inliers = GetInliers(
            solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
        max_num_iterations = utils::NumRequiredIterations(
            stats.inlier_ratio, 1.0 - options.success_probability_,
            kMinSampleSize, options.min_num_iterations_,
            options.max_num_iterations_);
      }

      sampler.Sample(&minimal_sample);

      // MinimalSolver returns the number of estimated models.
      const int kNumEstimatedModels =
          solver.MinimalSolver(minimal_sample, &estimated_models);
      if (kNumEstimatedModels <= 0)
        continue;

      // Finds the best model among all estimated models.
      double best_local_score = std::numeric_limits<double>::max();
      int best_local_model_id = 0;
      GetBestEstimatedModelId(solver, estimated_models, kNumEstimatedModels,
                              kSqrInlierThresh, &best_local_score,
                              &best_local_model_id);

      // Updates the best model found so far.
      if (best_local_score < best_min_model_score ||
          stats.num_iterations == options.lo_starting_iterations_) {
        const bool kBestMinModel = best_local_score < best_min_model_score;

        if (kBestMinModel) {
          // New best model (estimated from inliers found. Stores this model
          // and runs local optimization.
          best_min_model_score = best_local_score;
          best_minimal_model = estimated_models[best_local_model_id];

          // Updates the best model.
          UpdateBestModel(best_min_model_score, best_minimal_model,
                          &(stats.best_model_score), best_model);
        }

        const bool kRunLO =
            (stats.num_iterations >= options.lo_starting_iterations_ &&
             best_min_model_score < std::numeric_limits<double>::max());

        if ((!kBestMinModel) && (!kRunLO))
          continue;

        // Performs local optimization. By construction, the local optimization
        // method returns the best model between all models found by local
        // optimization and the input model, i.e., score_refined_model <=
        // best_min_model_score holds.
        if (kRunLO) {
          ++stats.number_lo_iterations;
          double score = best_min_model_score;
          LocalOptimization(options, solver, &rng, &best_minimal_model, &score);

          // Updates the best model.
          UpdateBestModel(score, best_minimal_model, &(stats.best_model_score),
                          best_model);
        }

        // Updates the number of RANSAC iterations.
        stats.best_num_inliers = GetInliers(
            solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
        max_num_iterations = utils::NumRequiredIterations(
            stats.inlier_ratio, 1.0 - options.success_probability_,
            kMinSampleSize, options.min_num_iterations_,
            options.max_num_iterations_);
      }
    }

    // As proposed by Lebeda et al., Local Optimization is not executed in
    // the first lo_starting_iterations_ iterations. If LO-MSAC needs less than
    // lo_starting_iterations_ iterations, we run LO now.
    if (stats.num_iterations <= options.lo_starting_iterations_ &&
        stats.best_model_score < std::numeric_limits<double>::max()) {
      ++stats.number_lo_iterations;
      LocalOptimization(options, solver, &rng, best_model,
                        &(stats.best_model_score));

      stats.best_num_inliers = GetInliers(solver, *best_model, kSqrInlierThresh,
                                          &(stats.inlier_indices));
      stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                           static_cast<double>(kNumData);
    }

    if (options.final_least_squares_) {
      Model refined_model = *best_model;
      solver.LeastSquares(stats.inlier_indices, &refined_model);

      double score = std::numeric_limits<double>::max();
      ScoreModel(solver, refined_model, kSqrInlierThresh, &score);
      if (score < stats.best_model_score) {
        stats.best_model_score = score;
        *best_model = refined_model;

        stats.best_num_inliers = GetInliers(
            solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
      }
    }

    return stats.best_num_inliers;
  }

protected:
  void GetBestEstimatedModelId(const Solver &solver, const ModelVector &models,
                               const int num_models,
                               const double squared_inlier_threshold,
                               double *best_score, int *best_model_id) const {
    *best_score = std::numeric_limits<double>::max();
    *best_model_id = 0;
    for (int m = 0; m < num_models; ++m) {
      double score = std::numeric_limits<double>::max();
      ScoreModel(solver, models[m], squared_inlier_threshold, &score);

      if (score < *best_score) {
        *best_score = score;
        *best_model_id = m;
      }
    }
  }

  void ScoreModel(const Solver &solver, const Model &model,
                  const double squared_inlier_threshold, double *score) const {
    const int kNumData = solver.num_data();
    *score = 0.0;
    for (int i = 0; i < kNumData; ++i) {
      double squared_error = solver.EvaluateModelOnPoint(model, i);
      *score += ComputeScore(squared_error, squared_inlier_threshold);
    }
  }

  // MSAC (top-hat) scoring function.
  inline double ComputeScore(const double squared_error,
                             const double squared_error_threshold) const {
    return std::min(squared_error, squared_error_threshold);
  }

  int GetInliers(const Solver &solver, const Model &model,
                 const double squared_inlier_threshold,
                 std::vector<int> *inliers) const {
    const int kNumData = solver.num_data();
    if (inliers == nullptr) {
      int num_inliers = 0;
      for (int i = 0; i < kNumData; ++i) {
        double squared_error = solver.EvaluateModelOnPoint(model, i);
        if (squared_error < squared_inlier_threshold) {
          ++num_inliers;
        }
      }
      return num_inliers;
    } else {
      inliers->clear();
      int num_inliers = 0;
      for (int i = 0; i < kNumData; ++i) {
        double squared_error = solver.EvaluateModelOnPoint(model, i);
        if (squared_error < squared_inlier_threshold) {
          ++num_inliers;
          inliers->push_back(i);
        }
      }
      return num_inliers;
    }
  }

  // See algorithms 2 and 3 in Lebeda et al.
  // The input model is overwritten with the refined model if the latter is
  // better, i.e., has a lower score.
  void LocalOptimization(const LORansacOptions &options, const Solver &solver,
                         std::mt19937 *rng, Model *best_minimal_model,
                         double *score_best_minimal_model) const {
    const int kNumData = solver.num_data();
    // kMinNonMinSampleSize stores how many data points are required for a
    // non-minimal sample. For example, consider the case of pose estimation
    // for a calibrated camera. A minimal sample has size 3, while the
    // smallest non-minimal sample has size 4.
    const int kMinNonMinSampleSize = solver.non_minimal_sample_size();
    if (kMinNonMinSampleSize > kNumData)
      return;

    const int kMinSampleSize = solver.min_sample_size();

    const double kSqInThresh = options.squared_inlier_threshold_;
    const double kThreshMult = options.threshold_multiplier_;

    // Performs an initial least squares fit of the best model found by the
    // minimal solver so far and then determines the inliers to that model
    // under a (slightly) relaxed inlier threshold.
    Model m_init = *best_minimal_model;
    LeastSquaresFit(options, kSqInThresh * kThreshMult, solver, rng, &m_init);

    double score = std::numeric_limits<double>::max();
    ScoreModel(solver, m_init, kSqInThresh, &score);
    UpdateBestModel(score, m_init, score_best_minimal_model,
                    best_minimal_model);

    std::vector<int> inliers_base;
    GetInliers(solver, m_init, kSqInThresh * kThreshMult, &inliers_base);

    // Determines the size of the non-miminal samples drawn in each LO step.
    const int kNonMinSampleSize =
        std::max(kMinNonMinSampleSize,
                 std::min(kMinSampleSize * options.non_min_sample_multiplier_,
                          static_cast<int>(inliers_base.size()) / 2));

    // Performs the actual local optimization (LO).
    std::vector<int> sample;
    for (int r = 0; r < options.num_lo_steps_; ++r) {
      sample = inliers_base;
      utils::RandomShuffleAndResize(kNonMinSampleSize, rng, &sample);

      Model m_non_min;
      if (!solver.NonMinimalSolver(sample, &m_non_min))
        continue;

      ScoreModel(solver, m_non_min, kSqInThresh, &score);
      UpdateBestModel(score, m_non_min, score_best_minimal_model,
                      best_minimal_model);

      // Iterative least squares refinement.
      LeastSquaresFit(options, kSqInThresh, solver, rng, &m_non_min);

      // The current threshold multiplier and its update.
      double thresh = kThreshMult * kSqInThresh;
      double thresh_mult_update =
          (kThreshMult - 1.0) * kSqInThresh /
          static_cast<int>(options.num_lsq_iterations_ - 1);
      for (int i = 0; i < options.num_lsq_iterations_; ++i) {
        LeastSquaresFit(options, thresh, solver, rng, &m_non_min);

        ScoreModel(solver, m_non_min, kSqInThresh, &score);
        UpdateBestModel(score, m_non_min, score_best_minimal_model,
                        best_minimal_model);
        thresh -= thresh_mult_update;
      }
    }
  }

  void LeastSquaresFit(const LORansacOptions &options, const double thresh,
                       const Solver &solver, std::mt19937 *rng,
                       Model *model) const {
    const int kLSqSampleSize =
        options.min_sample_multiplicator_ * solver.min_sample_size();
    std::vector<int> inliers;
    int num_inliers = GetInliers(solver, *model, thresh, &inliers);
    if (num_inliers < solver.min_sample_size())
      return;
    int lsq_data_size = std::min(kLSqSampleSize, num_inliers);
    utils::RandomShuffleAndResize(lsq_data_size, rng, &inliers);
    solver.LeastSquares(inliers, model);
  }

  inline void UpdateBestModel(const double score_curr, const Model &m_curr,
                              double *score_best, Model *m_best) const {
    if (score_curr < *score_best) {
      *score_best = score_curr;
      *m_best = m_curr;
    }
  }
};

} // namespace absolute_pose

} // namespace estimators

} // namespace limap

#endif
