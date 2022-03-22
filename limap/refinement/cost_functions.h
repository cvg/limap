#ifndef LIMAP_REFINEMENT_COST_FUNCTIONS_H_
#define LIMAP_REFINEMENT_COST_FUNCTIONS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/camera.h"
#include "base/infinite_line.h"
#include "base/featuremap.h"
#include "base/featurepatch.h"
#include "util/types.h"

#include <ceres/ceres.h>
#include "ceresbase/line_projection.h"
#include "ceresbase/line_dists.h"

namespace py = pybind11;

namespace limap {

namespace refinement {

////////////////////////////////////////////////////////////
// VP Constraints 
////////////////////////////////////////////////////////////

struct VPConstraintsFunctor {
public:
    VPConstraintsFunctor(const V3D& VP, const double* kvec = NULL, const double* qvec = NULL):
        VP_(VP), kvec_(kvec), qvec_(qvec) {}
    
    static ceres::CostFunction* Create(const V3D& VP, const double* kvec = NULL, const double* qvec = NULL) {
        if (!kvec && !qvec)
            return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2, 4, 4>(new VPConstraintsFunctor(VP, NULL, NULL));
        else
            return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2>(new VPConstraintsFunctor(VP, kvec, qvec));
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(kvec_);
        CHECK_NOTNULL(qvec_);

        T kvec[4] = {T(kvec_[0]), T(kvec_[1]), T(kvec_[2]), T(kvec_[3])};
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        return (*this)(uvec, wvec, kvec, qvec, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, const T* const kvec, const T* const qvec, T* residuals) const {
        // project to 2d
        T p3d[3], dir3d[3];
        MinimalPluckerToNormal<T>(uvec, wvec, p3d, dir3d);
        T dir3d_rotated[3];
        ceres::QuaternionRotatePoint(qvec, dir3d, dir3d_rotated);

        const V3D& vp = VP_;
        T vpvec[3] = {T(vp[0]), T(vp[1]), T(vp[2])};
        T direc[3];
        GetDirectionFromVP<T>(vpvec, kvec, direc);
        T innerp = dir3d_rotated[0] * direc[0] + dir3d_rotated[1] * direc[1] + dir3d_rotated[2] * direc[2];
        residuals[0] = T(1.0) - innerp;
        return true;
    }

protected:
    V3D VP_;
    const double* kvec_;
    const double* qvec_;
};

////////////////////////////////////////////////////////////
// Geometric Refinement
////////////////////////////////////////////////////////////

template <typename T>
T Ceres_GeometricRefinementResidual(const T p2d[2], const T dir2d[2], const T p1[2], const T p2[2], const double& alpha) {
    // (dist1 + dist2) * exp(alpha * (1 - cosine))
    const T alpha_t = T(alpha);
    T direc[2];
    direc[0] = p2[0] - p1[0];
    direc[1] = p2[1] - p1[1];
    T cosine = CeresComputeDist_cosine(dir2d, direc);
    T dist1 = CeresComputeDist_squaredperp(p2d, dir2d, p1);
    T dist2 = CeresComputeDist_squaredperp(p2d, dir2d, p2);
    T res = (dist1 + dist2) * ceres::exp(alpha_t * (T(1.0) - cosine));
    return res;
}

struct GeometricRefinementFunctor {
public:
    GeometricRefinementFunctor(const Line2d& line2d, const double* kvec = NULL, const double* qvec = NULL, const double* tvec = NULL, const double alpha = 10.0):
        line2d_(line2d), kvec_(kvec), qvec_(qvec), tvec_(tvec), alpha_(alpha) {}
    
    static ceres::CostFunction* Create(const Line2d& line2d, const double* kvec = NULL, const double* qvec = NULL, const double* tvec = NULL, const double alpha = 10.0) {
        if (!kvec && !qvec && !tvec)
            return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 1, 4, 2, 4, 4, 3>(new GeometricRefinementFunctor(line2d, NULL, NULL, NULL, alpha));
        else
            return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 1, 4, 2>(new GeometricRefinementFunctor(line2d, kvec, qvec, tvec, alpha));
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(kvec_);
        CHECK_NOTNULL(qvec_);
        CHECK_NOTNULL(tvec_);
        
        T kvec[4] = {T(kvec_[0]), T(kvec_[1]), T(kvec_[2]), T(kvec_[3])};
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
        return (*this)(uvec, wvec, kvec, qvec, tvec, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, 
                    const T* const kvec, const T* const qvec, const T* const tvec,
                    T* residuals) const {
        // project to 2d
        T p3d[3], dir3d[3];
        MinimalPluckerToNormal<T>(uvec, wvec, p3d, dir3d);
        T p2d[2], dir2d[2];
        Lines_WorldToPixel<T>(kvec, qvec, tvec, p3d, dir3d, p2d, dir2d);

        const Line2d& line = line2d_;
        T p1[2] = {T(line.start(0)), T(line.start(1))};
        T p2[2] = {T(line.end(0)), T(line.end(1))};
        residuals[0] = Ceres_GeometricRefinementResidual(p2d, dir2d, p1, p2, alpha_);
        return true;
    }

protected:
    Line2d line2d_;
    const double* kvec_;
    const double* qvec_;
    const double* tvec_;
    double alpha_; // for weighting angle
};

////////////////////////////////////////////////////////////
// Heatmap maximizer
////////////////////////////////////////////////////////////

template <typename DTYPE>
struct MaxHeatmapFunctor {
public:
    MaxHeatmapFunctor(std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator,  
                              const std::vector<InfiniteLine2d>& samples,
                              const double* kvec = NULL,
                              const double* qvec = NULL,
                              const double* tvec = NULL):
        interpolator_(interpolator), samples_(samples), kvec_(kvec), qvec_(qvec), tvec_(tvec) {}

    static ceres::CostFunction* Create(std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator,
                                       const std::vector<InfiniteLine2d>& samples,
                                       const double* kvec = NULL,
                                       const double* qvec = NULL,
                                       const double* tvec = NULL) {
        if (!kvec && !qvec && !tvec)
            return new ceres::AutoDiffCostFunction<MaxHeatmapFunctor, ceres::DYNAMIC, 4, 2, 4, 4, 3>(new MaxHeatmapFunctor(interpolator, samples, NULL, NULL, NULL), samples.size());
        else
            return new ceres::AutoDiffCostFunction<MaxHeatmapFunctor, ceres::DYNAMIC, 4, 2>(new MaxHeatmapFunctor(interpolator, samples, kvec, qvec, tvec), samples.size());
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(kvec_);
        CHECK_NOTNULL(qvec_);
        CHECK_NOTNULL(tvec_);

        T kvec[4] = {T(kvec_[0]), T(kvec_[1]), T(kvec_[2]), T(kvec_[3])};
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
        return (*this)(uvec, wvec, kvec, qvec, tvec, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, const T* const kvec, const T* const qvec, const T* const tvec, T* residuals) const {
        bool is_inside = true;
        size_t n_residuals = samples_.size();
        for (size_t i = 0; i < n_residuals; ++i) {
            const InfiniteLine2d& line = samples_[i];
            T p_sample[2] = {T(line.p(0)), T(line.p(1))};
            T dir_sample[2] = {T(line.direc(0)), T(line.direc(1))};
            T xy[2];
            bool res = GetIntersection2d<T>(uvec, wvec, kvec, qvec, tvec, p_sample, dir_sample, xy);
            is_inside = is_inside && res;
            T heatmap_val;
            interpolator_->Evaluate(xy, &heatmap_val);
            residuals[i] = T(1.0) - heatmap_val;
        }
        return is_inside;
    }

protected:
    std::unique_ptr<FeatureInterpolator<DTYPE, 1>>& interpolator_;
    std::vector<InfiniteLine2d> samples_;
    const double* kvec_;
    const double* qvec_;
    const double* tvec_;
};

////////////////////////////////////////////////////////////
// Feature consistency
////////////////////////////////////////////////////////////
template <typename INTERPOLATOR, int CHANNELS>
struct FeatureConsisSrcFunctor {
public:
    FeatureConsisSrcFunctor(std::unique_ptr<INTERPOLATOR>& interpolator,
                         const InfiniteLine2d& sample,
                         const double* ref_descriptor,
                         const double* kvec = NULL,
                         const double* qvec = NULL,
                         const double* tvec = NULL):
        interpolator_(interpolator), sample_(sample), ref_descriptor_(ref_descriptor),
        kvec_(kvec), qvec_(qvec), tvec_(tvec) {}

    static ceres::CostFunction* Create(std::unique_ptr<INTERPOLATOR>& interpolator,
                                       const InfiniteLine2d& sample,
                                       const double* ref_descriptor,
                                       const double* kvec = NULL,
                                       const double* qvec = NULL,
                                       const double* tvec = NULL) {
        if (!kvec && !qvec && !tvec)
            return new ceres::AutoDiffCostFunction<FeatureConsisSrcFunctor, ceres::DYNAMIC, 4, 2, 4, 4, 3>(new FeatureConsisSrcFunctor(interpolator, sample, ref_descriptor, NULL, NULL, NULL), CHANNELS);
        else
            return new ceres::AutoDiffCostFunction<FeatureConsisSrcFunctor, ceres::DYNAMIC, 4, 2>(new FeatureConsisSrcFunctor(interpolator, sample, ref_descriptor, kvec, qvec, tvec), CHANNELS);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(kvec_);
        CHECK_NOTNULL(qvec_);
        CHECK_NOTNULL(tvec_);

        T kvec[4] = {T(kvec_[0]), T(kvec_[1]), T(kvec_[2]), T(kvec_[3])};
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
        return (*this) (uvec, wvec, kvec, qvec, tvec, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, 
                    const T* const kvec, const T* const qvec, const T* const tvec,
                    T* residuals) const {
        const InfiniteLine2d& line = sample_;
        T p_sample[2] = {T(line.p(0)), T(line.p(1))};
        T dir_sample[2] = {T(line.direc(0)), T(line.direc(1))};

        // get feature at reference image
        bool is_inside = true;
        T xy[2];
        bool res = GetIntersection2d<T>(uvec, wvec, kvec, qvec, tvec, p_sample, dir_sample, xy);
        is_inside = is_inside && res;
        T feature[CHANNELS];
        interpolator_->Evaluate(xy, feature);

        // compute residuals
        for (int i = 0; i < CHANNELS; ++i) {
            residuals[i] = feature[i] - ref_descriptor_[i];
        }
        return is_inside;
    }
    
protected:
    // feature interpolator
    std::unique_ptr<INTERPOLATOR>& interpolator_;
    InfiniteLine2d sample_;
    const double* ref_descriptor_;

    const double* kvec_;
    const double* qvec_;
    const double* tvec_;
};

template <typename INTERPOLATOR, int CHANNELS>
struct FeatureConsisTgtFunctor {
public:
    FeatureConsisTgtFunctor(std::unique_ptr<INTERPOLATOR>& interpolator_ref,
                                    std::unique_ptr<INTERPOLATOR>& interpolator_tgt,
                                    const InfiniteLine2d& sample,
                                    const double* ref_descriptor = NULL,
                                    const double* kvec_ref = NULL,
                                    const double* qvec_ref = NULL,
                                    const double* tvec_ref = NULL,
                                    const double* kvec_tgt = NULL,
                                    const double* qvec_tgt = NULL,
                                    const double* tvec_tgt = NULL):
        interp_ref_(interpolator_ref), interp_tgt_(interpolator_tgt), sample_(sample),
        ref_descriptor_(ref_descriptor),
        kvec_ref_(kvec_ref), qvec_ref_(qvec_ref), tvec_ref_(tvec_ref),
        kvec_tgt_(kvec_tgt), qvec_tgt_(qvec_tgt), tvec_tgt_(tvec_tgt) {}

    static ceres::CostFunction* Create(std::unique_ptr<INTERPOLATOR>& interpolator_ref,
                                       std::unique_ptr<INTERPOLATOR>& interpolator_tgt,
                                       const InfiniteLine2d& sample,
                                       const double* ref_descriptor = NULL,
                                       const double* kvec_ref = NULL,
                                       const double* qvec_ref = NULL,
                                       const double* tvec_ref = NULL,
                                       const double* kvec_tgt = NULL,
                                       const double* qvec_tgt = NULL,
                                       const double* tvec_tgt = NULL) {
        if (!kvec_ref && !qvec_ref && !tvec_ref && !kvec_tgt && !qvec_tgt && !tvec_tgt)
            return new ceres::AutoDiffCostFunction<FeatureConsisTgtFunctor, ceres::DYNAMIC, 4, 2, 4, 4, 3, 4, 4, 3>(new FeatureConsisTgtFunctor(interpolator_ref, interpolator_tgt, sample, ref_descriptor, NULL, NULL, NULL, NULL, NULL, NULL), CHANNELS);
        else
            return new ceres::AutoDiffCostFunction<FeatureConsisTgtFunctor, ceres::DYNAMIC, 4, 2>(new FeatureConsisTgtFunctor(interpolator_ref, interpolator_tgt, sample, ref_descriptor, kvec_ref, qvec_ref, tvec_ref, kvec_tgt, qvec_tgt, tvec_tgt), CHANNELS);
    }

    // @debug
    template <typename T>
    bool intersect(const T* const uvec, const T* const wvec, T* intersection) const {
        T kvec_ref[4] = {T(kvec_ref_[0]), T(kvec_ref_[1]), T(kvec_ref_[2]), T(kvec_ref_[3])};
        T qvec_ref[4] = {T(qvec_ref_[0]), T(qvec_ref_[1]), T(qvec_ref_[2]), T(qvec_ref_[3])};
        T tvec_ref[3] = {T(tvec_ref_[0]), T(tvec_ref_[1]), T(tvec_ref_[2])};

        const auto& line = sample_;
        T p_sample[2] = {T(line.p(0)), T(line.p(1))};
        T dir_sample[2] = {T(line.direc(0)), T(line.direc(1))};
        GetIntersection2d<T>(uvec, wvec, kvec_ref, qvec_ref, tvec_ref, p_sample, dir_sample, intersection);
        return true;
    }

    // @debug
    template <typename T>
    bool intersect_epipolar(const T* const uvec, const T* wvec,
                            const T* const xy, T* intersection) const {
        CHECK_NOTNULL(kvec_ref_);
        CHECK_NOTNULL(qvec_ref_);
        CHECK_NOTNULL(tvec_ref_);
        CHECK_NOTNULL(kvec_tgt_);
        CHECK_NOTNULL(qvec_tgt_);
        CHECK_NOTNULL(tvec_tgt_);
        
        T kvec_ref[4] = {T(kvec_ref_[0]), T(kvec_ref_[1]), T(kvec_ref_[2]), T(kvec_ref_[3])};
        T qvec_ref[4] = {T(qvec_ref_[0]), T(qvec_ref_[1]), T(qvec_ref_[2]), T(qvec_ref_[3])};
        T tvec_ref[3] = {T(tvec_ref_[0]), T(tvec_ref_[1]), T(tvec_ref_[2])};
        T kvec_tgt[4] = {T(kvec_tgt_[0]), T(kvec_tgt_[1]), T(kvec_tgt_[2]), T(kvec_tgt_[3])};
        T qvec_tgt[4] = {T(qvec_tgt_[0]), T(qvec_tgt_[1]), T(qvec_tgt_[2]), T(qvec_tgt_[3])};
        T tvec_tgt[3] = {T(tvec_tgt_[0]), T(tvec_tgt_[1]), T(tvec_tgt_[2])};

        T epiline_coord[3];
        GetEpipolarLineCoordinate<T>(kvec_ref, qvec_ref, tvec_ref, kvec_tgt, qvec_tgt, tvec_tgt, xy, epiline_coord);
        GetIntersection2d_line_coordinate<T>(uvec, wvec, kvec_tgt, qvec_tgt, tvec_tgt, epiline_coord, intersection);
        return true; 
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(kvec_ref_);
        CHECK_NOTNULL(qvec_ref_);
        CHECK_NOTNULL(tvec_ref_);
        CHECK_NOTNULL(kvec_tgt_);
        CHECK_NOTNULL(qvec_tgt_);
        CHECK_NOTNULL(tvec_tgt_);

        T kvec_ref[4] = {T(kvec_ref_[0]), T(kvec_ref_[1]), T(kvec_ref_[2]), T(kvec_ref_[3])};
        T qvec_ref[4] = {T(qvec_ref_[0]), T(qvec_ref_[1]), T(qvec_ref_[2]), T(qvec_ref_[3])};
        T tvec_ref[3] = {T(tvec_ref_[0]), T(tvec_ref_[1]), T(tvec_ref_[2])};
        T kvec_tgt[4] = {T(kvec_tgt_[0]), T(kvec_tgt_[1]), T(kvec_tgt_[2]), T(kvec_tgt_[3])};
        T qvec_tgt[4] = {T(qvec_tgt_[0]), T(qvec_tgt_[1]), T(qvec_tgt_[2]), T(qvec_tgt_[3])};
        T tvec_tgt[3] = {T(tvec_tgt_[0]), T(tvec_tgt_[1]), T(tvec_tgt_[2])};
        return (*this) (uvec, wvec, kvec_ref, qvec_ref, tvec_ref, kvec_tgt, qvec_tgt, tvec_tgt, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, 
                    const T* const kvec_ref, const T* const qvec_ref, const T* const tvec_ref,
                    const T* const kvec_tgt, const T* const qvec_tgt, const T* const tvec_tgt,
                    T* residuals) const {
        const InfiniteLine2d& line = sample_;
        T p_sample[2] = {T(line.p(0)), T(line.p(1))};
        T dir_sample[2] = {T(line.direc(0)), T(line.direc(1))};

        // get feature at reference image
        bool is_inside = true;
        T xy_ref[2];
        bool res = GetIntersection2d<T>(uvec, wvec, kvec_ref, qvec_ref, tvec_ref, p_sample, dir_sample, xy_ref);
        is_inside = is_inside && res;
        T feature_ref[CHANNELS];
        interp_ref_->Evaluate(xy_ref, feature_ref);

        // get feature at target image
        T xy_tgt[2];
        T epiline_coord[3];
        GetEpipolarLineCoordinate<T>(kvec_ref, qvec_ref, tvec_ref, kvec_tgt, qvec_tgt, tvec_tgt, xy_ref, epiline_coord);
        res = GetIntersection2d_line_coordinate<T>(uvec, wvec, kvec_tgt, qvec_tgt, tvec_tgt, epiline_coord, xy_tgt);
        is_inside = is_inside && res;
        T feature_tgt[CHANNELS];
        interp_tgt_->Evaluate(xy_tgt, feature_tgt);

        // compute residuals
        for (int i = 0; i < CHANNELS; ++i) {
            if (!ref_descriptor_)
                residuals[i] = feature_tgt[i] - feature_ref[i];
            else
                residuals[i] = feature_tgt[i] - ref_descriptor_[i];
        }
        return is_inside;
    }
    
protected:
    // feature interpolator
    std::unique_ptr<INTERPOLATOR>& interp_ref_;
    std::unique_ptr<INTERPOLATOR>& interp_tgt_;
    InfiniteLine2d sample_;
    const double* ref_descriptor_ = NULL;

    const double* kvec_ref_;
    const double* qvec_ref_;
    const double* tvec_ref_;
    const double* kvec_tgt_;
    const double* qvec_tgt_;
    const double* tvec_tgt_;
};

} // namespace refinement

} // namespace limap

#endif

