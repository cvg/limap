#include "base/infinite_line.h"

#include <cmath>
#include <colmap/base/pose.h>

namespace limap {

InfiniteLine2d::InfiniteLine2d(const Line2d& line) {
    p = line.start;
    direc = line.end - line.start;
}

V3D InfiniteLine2d::GetLineCoordinate() const {
    V3D coor;
    coor(0) = direc(1);
    coor(1) = -direc(0);
    coor(2) = - direc(1) * p(0) + direc(0) * p(1);
    return coor.normalized();
}

V2D InfiniteLine2d::point_projection(const V2D& p2d) const {
    V2D dir = direc.normalized();
    return p + (p2d - p).dot(dir) * dir;
}

std::pair<V2D, bool> Intersect_InfiniteLine2d(const InfiniteLine2d& l1, const InfiniteLine2d& l2) {
    V3D coor1 = l1.GetLineCoordinate();
    V3D coor2 = l2.GetLineCoordinate();
    V3D p_homo = coor1.cross(coor2).normalized();
    V2D p(0, 0);
    if (std::abs(p_homo(2)) < EPS)
        return std::make_pair(p, false);
    else {
        p(0) = p_homo(0) / p_homo(2);
        p(1) = p_homo(1) / p_homo(2);
        return std::make_pair(p, true);
    }
}

InfiniteLine3d::InfiniteLine3d(const Line3d& line) {
    p = line.start;
    direc = line.end - line.start;
}

V3D InfiniteLine3d::point_projection(const V3D& p3d) const {
    V3D dir = direc.normalized();
    return p + (p3d - p).dot(dir) * dir;
}

InfiniteLine2d InfiniteLine3d::projection(const PinholeCamera& camera) const {
    InfiniteLine2d inf_line2d;
    V2D p2d  = camera.projection(p);
    V2D p2d_prime = camera.projection(p + direc * 1.0);
    inf_line2d.p = p2d;
    inf_line2d.direc = (p2d_prime - p2d).normalized();
}

V3D InfiniteLine3d::unprojection(const V2D& p2d, const PinholeCamera& camera) const {
    // take the closest point along the 3D lines towards the camera ray
    // min |C0 + C1 * t1 + C2 * t2|^2, C0 = p1 - p2
    V3D p1, p2;
    V3D C0, C1, C2;
    p1 = p; 
    p2 = camera.GetPosition(); 
    C0 = p1 - p2;
    C1 = direc;
    C2 = camera.GetCameraRay(p2d);
    C1 = C1.normalized(); C2 = C2.normalized();
    
    double A11, A12, A21, A22, B1, B2;
    A11 = A22 = 1.0;
    A12 = A21 = C1.dot(C2);
    B1 = -C0.dot(C1); B2 = -C0.dot(C2);
    double det = A11 * A22 - A12 * A21;
    double t;
    if (det < EPS) // l1 and l2 nearly collinear
    {
        t = B1 / (A11 + EPS);
    }
    else
        t = (B1 * A22 - B2 * A12) / det;
    return p + t * direc;
}

MinimalInfiniteLine3d::MinimalInfiniteLine3d(const std::vector<double>& values) {
    THROW_CHECK_EQ(values.size(), 6);
    uvec[0] = values[0];
    uvec[1] = values[1];
    uvec[2] = values[2];
    uvec[3] = values[3];
    wvec[0] = values[4];
    wvec[1] = values[5];
}

MinimalInfiniteLine3d::MinimalInfiniteLine3d(const InfiniteLine3d& inf_line) {
    // [LINK] https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    // [LINK] https://hal.archives-ouvertes.fr/hal-00092589/document

    // get Plucker coordinate
    V3D a = inf_line.direc;
    V3D b = inf_line.p.cross(inf_line.direc);

    // orthonormal representation
    // SO(2)
    double w1, w2;
    w1 = a.norm(); w2 = b.norm();
    wvec(0) = w1 / (w1 + w2);
    wvec(1) = w2 / (w1 + w2);

    // SO(3)
    M3D Q;
    Q.col(0) = a / a.norm();
    if (b.norm() > EPS) {
        Q.col(1) = b / b.norm();
        V3D axb = a.cross(b);
        Q.col(2) = axb / axb.norm();
    }
    else {
        int best_index = 0;
        if (std::abs(a(1)) > std::abs(a(0)))
            best_index = 1;
        if (std::abs(a(2)) > std::abs(a(best_index)))
            best_index = 2;
        int i1 = (best_index + 1) % 3;
        int i2 = (best_index + 2) % 3;
        V3D bprime;
        bprime(i1) = 1.0; bprime(i2) = 1.0;
        bprime(best_index) = - (a(i1) * bprime(i1) + a(i2) * bprime(i2)) / a(best_index);
        Q.col(1) = bprime / bprime.norm();
        V3D axb = a.cross(bprime);
        Q.col(2) = axb / axb.norm();
    }
    uvec = colmap::RotationMatrixToQuaternion(Q);
}

InfiniteLine3d MinimalInfiniteLine3d::GetInfiniteLine() const {
    // get plucker coordinate
    M3D Q = colmap::QuaternionToRotationMatrix(uvec);
    V3D a = std::abs(wvec(0)) * Q.col(0);
    V3D b = std::abs(wvec(1)) * Q.col(1);

    // transform plucker coordniate
    InfiniteLine3d inf_line;
    inf_line.direc = a;
    inf_line.p = a.cross(b) / (a.squaredNorm());
    return inf_line;
}

Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d& inf_line, const std::vector<PinholeCamera>& cameras, const std::vector<Line2d>& line2ds, const int num_outliers) {
    // TODO: not sure this function is working.
    THROW_CHECK_EQ(cameras.size(), line2ds.size());
    int n_lines = line2ds.size();
    V3D dir = inf_line.direc.normalized();

    std::vector<double> values;
    for (int i = 0; i < n_lines; ++i) {
        const auto& cam = cameras[i];
        const Line2d& line2d = line2ds[i];
        InfiniteLine2d inf_line2d_proj = inf_line.projection(cam);
        // project the two 2D endpoints to the 2d projection of the infinite line
        V2D pstart_2d = inf_line2d_proj.point_projection(line2d.start);
        V3D pstart_3d = inf_line.unprojection(pstart_2d, cam);
        double tstart = (pstart_3d - inf_line.p).dot(dir);
        V2D pend_2d = inf_line2d_proj.point_projection(line2d.end);
        V3D pend_3d = inf_line.unprojection(pend_2d, cam);
        double tend = (pend_3d - inf_line.p).dot(dir);

        values.push_back(tstart);
        values.push_back(tend);
    }
    std::sort(values.begin(), values.end());
    Line3d final_line;
    final_line.start = inf_line.p + dir * values[num_outliers];
    final_line.end = inf_line.p + dir * values[n_lines * 2 - 1 - num_outliers];
    return final_line;
}

Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d& inf_line, const std::vector<Line3d>& line3ds, const int num_outliers) {
    int n_lines = line3ds.size();
    V3D dir = inf_line.direc.normalized();

    std::vector<double> values;
    for (int i = 0; i < n_lines; ++i) {
        const Line3d& line3d = line3ds[i];
        double tstart = (line3d.start - inf_line.p).dot(dir);
        double tend = (line3d.end - inf_line.p).dot(dir);
        values.push_back(tstart);
        values.push_back(tend);
    }
    std::sort(values.begin(), values.end());
    Line3d final_line;
    final_line.start = inf_line.p + dir * values[num_outliers];
    final_line.end = inf_line.p + dir * values[n_lines * 2 - 1 - num_outliers];
    return final_line;
}

} // namespace limap

