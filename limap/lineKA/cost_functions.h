#ifndef LIMAP_LINEKA_COST_FUNCTIONS_H_
#define LIMAP_LINEKA_COST_FUNCTIONS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/linebase.h"
#include "base/line2d_parameterization.h"
#include "base/camera_view.h"
#include "base/featurepatch.h"
#include "util/types.h"

#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

namespace lineKA {

////////////////////////////////////////////////////////////
// Heatmap maximizer
////////////////////////////////////////////////////////////

template <typename DTYPE>
struct MaxHeatmap2DFunctor {
public:
    MaxHeatmap2DFunctor(std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator, 
                                         const std::shared_ptr<Line2d_2DOF>& line2d,
                                         const std::vector<double>& t_array):
        interpolator_(interpolator), line2d_(line2d), t_array_(t_array) {}

    static ceres::CostFunction* Create(std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator,
                                       const std::shared_ptr<Line2d_2DOF>& line2d,
                                       const std::vector<double>& t_array) {
        return new ceres::AutoDiffCostFunction<MaxHeatmap2DFunctor, ceres::DYNAMIC, 2>(new MaxHeatmap2DFunctor(interpolator, line2d, t_array), t_array.size());
    }

    template <typename T>
    bool Onepoint(const T* const input, const double val, T* residuals) const {
        T xy[2];
        line2d_->GetPoint<T>(input, val, xy);
        T heatmap_val;
        interpolator_->Evaluate(xy, &heatmap_val);
        residuals[0] = T(1.0) - heatmap_val;
        return true;
    }

    template <typename T>
    bool operator()(const T* const input, T* residuals) const {
        size_t n_samples = t_array_.size();
        for (size_t i = 0; i < n_samples; ++i) {
            double val = t_array_[i];
            Onepoint(input, val, &residuals[i]);
        }
        return true;
    }

protected:
    std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator_;
    std::shared_ptr<Line2d_2DOF> line2d_;
    std::vector<double> t_array_;
};

////////////////////////////////////////////////////////////
// Feature consistency
////////////////////////////////////////////////////////////
template <typename INTERPOLATOR, int CHANNELS>
struct FeatureConsis2DFunctor {
public:
    FeatureConsis2DFunctor(std::unique_ptr<INTERPOLATOR>& interpolator_ref,
                                    const std::shared_ptr<Line2d_2DOF>& ref_line2d,
                                    const MinimalPinholeCamera& camera_ref,
                                    std::unique_ptr<INTERPOLATOR>& interpolator_tgt,
                                    const std::shared_ptr<Line2d_2DOF>& tgt_line2d,
                                    const MinimalPinholeCamera& camera_tgt,
                                    const std::vector<double>& t_array,
                                    const double& weight):
        interp_ref_(interpolator_ref), ref_line2d_(ref_line2d), cam_ref_(camera_ref),
        interp_tgt_(interpolator_tgt), tgt_line2d_(tgt_line2d), cam_tgt_(camera_tgt), 
        t_array_(t_array), weight_(weight) {}

    static ceres::CostFunction* Create(std::unique_ptr<INTERPOLATOR>& interpolator_ref,
                                       const std::shared_ptr<Line2d_2DOF>& ref_line2d,
                                       const MinimalPinholeCamera& camera_ref,
                                       std::unique_ptr<INTERPOLATOR>& interpolator_tgt,
                                       const std::shared_ptr<Line2d_2DOF>& tgt_line2d,
                                       const MinimalPinholeCamera& camera_tgt,
                                       const std::vector<double>& t_array,
                                       const double& weight) {
        return new ceres::AutoDiffCostFunction<FeatureConsis2DFunctor, ceres::DYNAMIC, 2, 2>(new FeatureConsis2DFunctor(interpolator_ref, ref_line2d, camera_ref, interpolator_tgt, tgt_line2d, camera_tgt, t_array, weight), CHANNELS * t_array.size());
    }

    template <typename T>
    bool Onepoint(const T* const ref_input, const T* const tgt_input, const double val, T* residuals) const {
        const double* kvec_ref_ = cam_ref_.kvec.data();
        T kvec_ref[4] = {T(kvec_ref_[0]), T(kvec_ref_[1]), T(kvec_ref_[2]), T(kvec_ref_[3])};
        const double* qvec_ref_ = cam_ref_.qvec.data();
        T qvec_ref[4] = {T(qvec_ref_[0]), T(qvec_ref_[1]), T(qvec_ref_[2]), T(qvec_ref_[3])};
        const double* tvec_ref_ = cam_ref_.tvec.data();
        T tvec_ref[3] = {T(tvec_ref_[0]), T(tvec_ref_[1]), T(tvec_ref_[2])};
        const double* kvec_tgt_ = cam_tgt_.kvec.data();
        T kvec_tgt[4] = {T(kvec_tgt_[0]), T(kvec_tgt_[1]), T(kvec_tgt_[2]), T(kvec_tgt_[3])};
        const double* qvec_tgt_ = cam_tgt_.qvec.data();
        T qvec_tgt[4] = {T(qvec_tgt_[0]), T(qvec_tgt_[1]), T(qvec_tgt_[2]), T(qvec_tgt_[3])};
        const double* tvec_tgt_ = cam_tgt_.tvec.data();
        T tvec_tgt[3] = {T(tvec_tgt_[0]), T(tvec_tgt_[1]), T(tvec_tgt_[2])};
    
        // get feature at reference image
        T xy_ref[2];
        ref_line2d_->GetPoint<T>(ref_input, val, xy_ref);
        T feature_ref[CHANNELS];
        interp_ref_->Evaluate(xy_ref, feature_ref);

        // get feature at target image
        T epiline_coord[3];
        GetEpipolarLineCoordinate<T>(kvec_ref, qvec_ref, tvec_ref, kvec_tgt, qvec_tgt, tvec_tgt, xy_ref, epiline_coord);
        T p2d[2], dir2d[2];
        tgt_line2d_->GetInfiniteLine2d<T>(tgt_input, p2d, dir2d);
        T coord[3];
        CeresGetLineCoordinate<T>(p2d, dir2d, coord);
        T xy_tgt[2];
        CeresIntersect_LineCoordinates(epiline_coord, coord, xy_tgt);
        T feature_tgt[CHANNELS];
        interp_tgt_->Evaluate(xy_tgt, feature_tgt);
       
        // compute residuals
        for (int i = 0; i < CHANNELS; ++i) {
            residuals[i] = weight_ * (feature_ref[i] - feature_tgt[i]);
        }
        return true;
    }

    template <typename T>
    bool operator()(const T* const ref_input, const T* const tgt_input, T* residuals) const {
        size_t n_samples = t_array_.size();
        for (size_t i = 0; i < n_samples; ++i) {
            double val = t_array_[i];
            Onepoint(ref_input, tgt_input, val, &residuals[i * CHANNELS]);
        }
        return true;
    }

protected:
    // feature interpolator
    std::unique_ptr<INTERPOLATOR>& interp_ref_;
    std::shared_ptr<Line2d_2DOF> ref_line2d_;
    MinimalPinholeCamera cam_ref_;

    std::unique_ptr<INTERPOLATOR>& interp_tgt_;
    std::shared_ptr<Line2d_2DOF> tgt_line2d_;
    MinimalPinholeCamera cam_tgt_;

    std::vector<double> t_array_;
    double weight_;
};

} // namespace lineKA 

} // namespace limap

#endif

