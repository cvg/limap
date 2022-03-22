#ifndef LIMAP_CERESBASE_LINE_DISTS_H
#define LIMAP_CERESBASE_LINE_DISTS_H

#include <ceres/ceres.h>

namespace limap {

template <typename T>
T CeresComputeDist_cosine(const T dir1[2], const T dir2[2]) {
    T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + EPS);
    T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + EPS);
    T cosine = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (dir1_norm * dir2_norm);
    cosine = ceres::abs(cosine);
    if (cosine > T(1.0))
        cosine = T(1.0);
    return cosine;
}

template <typename T>
T CeresComputeDist_angle(const T dir1[2], const T dir2[2]) {
    T cosine = CeresComputeDist_cosine<T>(dir1, dir2);
    T angle = ceres::acos(cosine);
    if (ceres::IsNaN(angle) || ceres::IsInfinite(angle))
        angle = T(0.0);
    return angle;
}

// squared perpendicular distance
template <typename T>
T CeresComputeDist_squaredperp(const T p2d[2], const T dir2d[2], const T point[2]) {
    T dir2d_squaredNorm = dir2d[0] * dir2d[0] + dir2d[1] * dir2d[1];
    T disp[2];
    disp[0] = point[0] - p2d[0];
    disp[1] = point[1] - p2d[1];
    T disp_squaredNorm = disp[0] * disp[0] + disp[1] * disp[1];
    T innerp = disp[0] * dir2d[0] + disp[1] * dir2d[1];
    T proj_squaredNorm = innerp * innerp / dir2d_squaredNorm;
    T distsquared = disp_squaredNorm - proj_squaredNorm;
    return distsquared;
}

} // namespace limap

#endif

