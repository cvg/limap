import logging
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.utils import check_random_state
from sklearn.utils._random import sample_without_replacement
from sklearn.utils.validation import check_is_fitted

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LinearModelEstimator import LinearModelEstimator

class RANSACRegressor3D:
    """
    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    """

    def __init__(self, base_estimator=None, min_samples=None,
                 residual_threshold=None, is_data_valid=None,
                 is_model_valid=None, max_trials=100, max_skips=np.inf,
                 stop_n_inliers=np.inf, stop_score=np.inf,
                 stop_probability=0.99, residual_metric=None,
                 loss='absolute_loss', random_state=None):

        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.residual_metric = residual_metric
        self.random_state = random_state
        self.loss = loss

    def fit(self, points):
        assert len(points.shape) == 2
        assert points.shape[0] >= 2
        assert points.shape[1] >= 2

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = LinearModelEstimator()

        if self.min_samples is None:
            # assume linear model by default
            min_samples = points.shape[1]  # + 1
        elif 0 < self.min_samples < 1:
            min_samples = np.ceil(self.min_samples * points.shape[0])
        elif self.min_samples >= 1:
            if self.min_samples % 1 != 0:
                raise ValueError("Absolute number of samples must be an "
                                 "integer value.")
            min_samples = self.min_samples
        else:
            raise ValueError("Value for `min_samples` must be scalar and "
                             "positive.")
        if min_samples > points.shape[0]:
            raise ValueError("`min_samples` may not be larger than number "
                             "of samples ``X.shape[0]``.")

        if self.stop_probability < 0 or self.stop_probability > 1:
            raise ValueError("`stop_probability` must be in range [0, 1].")

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(points - np.median(points)))
        else:
            residual_threshold = self.residual_threshold

        random_state = check_random_state(self.random_state)

        try:  # Not all estimator accept a random_state
            base_estimator.set_params(random_state=random_state)
        except ValueError:
            pass

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        points_inlier_best = None
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0

        # number of data samples
        n_samples = points.shape[0]
        sample_idxs = np.arange(n_samples)

        logging.debug("-->\t Starting RANSAC Loop with params" +
                      "\n\t\tresidual_threshold:" + str(residual_threshold) +
                      "\n\t\tmax_skips: " + str(self.max_skips) +
                      "\n\t\tmax_trials: " + str(self.max_trials))

        self.n_trials_ = 0
        max_trials = self.max_trials
        while self.n_trials_ < max_trials:
            logging.debug("-->\t Ransac trial: " + str(self.n_trials_))
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            points_subset = points[subset_idxs]

            # check if random sample set is valid
            if self.is_data_valid is not None and \
                    not self.is_data_valid(points_subset):
                self.n_skips_invalid_data_ += 1
                continue

            try:
                base_estimator.fit(points_subset)
            except np.linalg.LinAlgError:
                continue

            # check if estimated model is valid
            if self.is_model_valid is not None and \
                    not self.is_model_valid(base_estimator, points_subset):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            residuals_subset = base_estimator.residues(points)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            points_inlier_subset = points[inlier_idxs_subset]
            # score of inlier data set
            score_subset = base_estimator.score(points_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            points_inlier_best = points_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                    score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            if ((self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                 self.n_skips_invalid_model_) > self.max_skips):
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*).")
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*).")
        else:
            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
                warnings.warn("RANSAC found a valid consensus set but exited"
                              " early due to skipping more iterations than"
                              " `max_skips`. See estimator attributes for"
                              " diagnostics (n_skips*).",
                              UserWarning)

        logging.debug("-->\t Found " + str(points_inlier_best.shape[0]) + " inliers of "
                      + str(points.shape[0]) + " samples.")
        # estimate final model using all inliers
        base_estimator.fit(points_inlier_best)

        self.estimator_ = base_estimator
        self.inlier_mask_ = inlier_mask_best
        return self

    def score(self, points):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.score(points)

