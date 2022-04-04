#ifndef LIMAP_BASE_INFINITE_LINE_H_
#define LIMAP_BASE_INFINITE_LINE_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/linebase.h"
#include "base/camera_view.h"
#include "util/types.h"

#include <ceres/ceres.h>
#include "ceresbase/line_projection.h"

namespace py = pybind11;

namespace limap {

class InfiniteLine2d {
public:
    InfiniteLine2d() {}
    InfiniteLine2d(const V2D& p_, const V2D& direc_): p(p_), direc(direc_) {};
    InfiniteLine2d(const Line2d& line);
    V2D point_projection(const V2D& p2d) const;
    V3D GetLineCoordinate() const; // get homogenous line coordinate

    V2D p;
    V2D direc;
};

std::pair<V2D, bool> Intersect_InfiniteLine2d(const InfiniteLine2d& l1, const InfiniteLine2d& l2);

class InfiniteLine3d {
public:
    InfiniteLine3d() {}
    InfiniteLine3d(const V3D& p_, const V3D& direc_): p(p_), direc(direc_) {};
    InfiniteLine3d(const Line3d& line);
    V3D point_projection(const V3D& p3d) const;
    InfiniteLine2d projection(const CameraView& view) const;
    V3D unprojection(const V2D& p2d, const CameraView& view) const;

    V3D p;
    V3D direc;
};

// minimal Plucker coordinate used for ceres optimization
class MinimalInfiniteLine3d {
public:
    MinimalInfiniteLine3d() {}
    MinimalInfiniteLine3d(const Line3d& line): MinimalInfiniteLine3d(InfiniteLine3d(line)) {};
    MinimalInfiniteLine3d(const InfiniteLine3d& inf_line);
    MinimalInfiniteLine3d(const std::vector<double>& values);
    InfiniteLine3d GetInfiniteLine() const;

    V4D uvec; // quaternion vector for SO(3)
    V2D wvec; // homogenous vector for SO(2)
};

Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d& inf_line, const std::vector<CameraView>& views, const std::vector<Line2d>& line2ds, const int num_outliers = 2); // views.size() == line2ds.size()
Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d& inf_line, const std::vector<Line3d>& line3ds, const int num_outliers = 2);

////////////////////////////////////////////
// ceres functions start here
////////////////////////////////////////////

template <typename T>
void MinimalPluckerToNormal(const T uvec[4], const T wvec[2], T p[3], T direc[3]) {
    T rotmat[3 * 3];
    ceres::QuaternionToRotation(uvec, rotmat);
    T w1, w2;
    w1 = ceres::abs(wvec[0]);
    w2 = ceres::abs(wvec[1]);

    // direc = a = Q.col(0) * w1
    // b = Q.col(1) * w2
    direc[0] = rotmat[0] * w1;
    direc[1] = rotmat[3] * w1;
    direc[2] = rotmat[6] * w1;
    T b[3];
    b[0] = rotmat[1] * w2;
    b[1] = rotmat[4] * w2;
    b[2] = rotmat[7] * w2;

    // normalize Plucker coordinate
    T norm = ceres::sqrt(direc[0] * direc[0] + direc[1] * direc[1] + direc[2] * direc[2]);
    direc[0] /= norm;
    direc[1] /= norm;
    direc[2] /= norm;
    b[0] /= norm;
    b[1] /= norm;
    b[2] /= norm;

    // get position
    ceres::CrossProduct(direc, b, p);
}

template <typename T>
void CeresGetLineCoordinate(const T p[2], const T direc[2], T coor[3]) {
    coor[0] = direc[1];
    coor[1] = -direc[0];
    coor[2] = -direc[1] * p[0] + direc[0] * p[1];
    T norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] + coor[2] * coor[2]);
    coor[0] /= norm;
    coor[1] /= norm;
    coor[2] /= norm;
}

template <typename T>
bool CeresIntersect_LineCoordinates(const T coor1[3], const T coor2[3], T xy[2]) {
    T p_homo[3];
    ceres::CrossProduct(coor1, coor2, p_homo);
    T norm = ceres::sqrt(p_homo[0] * p_homo[0] + p_homo[1] * p_homo[1] + p_homo[2] * p_homo[2]);
    p_homo[0] /= norm;
    p_homo[1] /= norm;
    p_homo[2] /= norm;
    T eps(EPS);
    if (ceres::abs(p_homo[2]) < eps)
        return false;
    else {
        xy[0] = p_homo[0] / p_homo[2];
        xy[1] = p_homo[1] / p_homo[2];
    }
    return true;
}

template <typename T>
bool CeresIntersect_InfiniteLine2d(const T p1[2], const T dir1[2], const T p2[2], const T dir2[2], T xy[2]) {
    T coor1[3], coor2[3];
    CeresGetLineCoordinate<T>(p1, dir1, coor1);
    CeresGetLineCoordinate<T>(p2, dir2, coor2);
    return CeresIntersect_LineCoordinates<T>(coor1, coor2, xy);
}

template <typename T>
bool GetIntersection2d(const T uvec[4], const T wvec[2], // MinimalLine
                       const T kvec[4], const T qvec[4], const T tvec[3], // CameraView 
                       const T p_sample[2], const T dir_sample[2], // InfiniteLine2d sample
                       T xy[2]) 
{
    T p3d[3], dir3d[3];
    MinimalPluckerToNormal<T>(uvec, wvec, p3d, dir3d);
    T p2d[2], dir2d[2];
    Lines_WorldToPixel<T>(kvec, qvec, tvec, p3d, dir3d, p2d, dir2d);
    return CeresIntersect_InfiniteLine2d(p2d, dir2d, p_sample, dir_sample, xy);
} 

template <typename T>
bool GetIntersection2d_line_coordinate(const T uvec[4], const T wvec[2], // MinimalLine
                                       const T kvec[4], const T qvec[4], const T tvec[3], // CameraView
                                       const T coor[3], // the line coordinate of the sample
                                       T xy[2]) 
{
    T p3d[3], dir3d[3];
    MinimalPluckerToNormal<T>(uvec, wvec, p3d, dir3d);
    T p2d[2], dir2d[2];
    Lines_WorldToPixel<T>(kvec, qvec, tvec, p3d, dir3d, p2d, dir2d);
    T coor_proj[3];
    CeresGetLineCoordinate<T>(p2d, dir2d, coor_proj);
    return CeresIntersect_LineCoordinates(coor_proj, coor, xy);
}

} // namespace limap

#endif


