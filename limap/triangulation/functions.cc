#include "triangulation/functions.h"

namespace limap {

namespace triangulation {

bool test_line_inside_ranges(const Line3d& line, const std::pair<V3D, V3D>& ranges) {
    // test start
    if (line.start[0] < ranges.first[0] || (line.start[0] > ranges.second[0]))
        return false;
    if (line.start[1] < ranges.first[1] || (line.start[1] > ranges.second[1]))
        return false;
    if (line.start[2] < ranges.first[2] || (line.start[2] > ranges.second[2]))
        return false;

    // test end
    if (line.end[0] < ranges.first[0] || (line.end[0] > ranges.second[0]))
        return false;
    if (line.end[1] < ranges.first[1] || (line.end[1] > ranges.second[1]))
        return false;
    if (line.end[2] < ranges.first[2] || (line.end[2] > ranges.second[2]))
        return false;
    return true;
}

V3D getNormalDirection(const Line2d& l, const CameraView& view) {
    const M3D K_inv = view.K_inv();
    const M3D R = view.R();
    V3D c_start = R.transpose() * K_inv * V3D(l.start[0], l.start[1], 1);
    V3D c_end = R.transpose() * K_inv * V3D(l.end[0], l.end[1], 1);
    V3D n = c_start.cross(c_end);
    return n.normalized();
}

V3D getDirectionFromVP(const V3D& vp, const CameraView& view) {
    const M3D K_inv = view.K_inv();
    const M3D R = view.R();
    V3D direc = R.transpose() * K_inv * vp;
    return direc.normalized();
}

double compute_epipolar_IoU(const Line2d& l1, const CameraView& view1,
                            const Line2d& l2, const CameraView& view2) 
{
    const M3D K1_inv = view1.K_inv();
    const M3D R1 = view1.R();
    const V3D T1 = view1.T();

    const M3D K2_inv = view2.K_inv();
    const M3D R2 = view2.R();
    const V3D T2 = view2.T();

    // compose relative pose
    M3D relR = R2 * R1.transpose();
    V3D relT = T2 - relR * T1;

    // essential matrix and fundamental matrix
    M3D tskew;
    tskew(0, 0) = 0.0; tskew(0, 1) = -relT[2]; tskew(0, 2) = relT[1];
    tskew(1, 0) = relT[2]; tskew(1, 1) = 0.0; tskew(1, 2) = -relT[0];
    tskew(2, 0) = -relT[1]; tskew(2, 1) = relT[0]; tskew(2, 2) = 0.0;
    M3D E = tskew * relR;
    M3D F = K2_inv.transpose() * E * K1_inv;

    // epipolar lines
    V3D coor_l2 = l2.coords();
    V3D coor_epline_start = (F * V3D(l1.start[0], l1.start[1], 1)).normalized();
    V3D homo_c_start = coor_l2.cross(coor_epline_start);
    V2D c_start = dehomogeneous(homo_c_start);
    V3D coor_epline_end = (F * V3D(l1.end[0], l1.end[1], 1)).normalized();
    V3D homo_c_end = coor_l2.cross(coor_epline_end);
    V2D c_end = dehomogeneous(homo_c_end);

    // compute IoU
    double c1 = (c_start - l2.start).dot(l2.direction()) / l2.length();
    double c2 = (c_end - l2.start).dot(l2.direction()) / l2.length();
    if (c1 > c2)
        std::swap(c1, c2);
    double IoU = (std::min(c2, 1.0) - std::max(c1, 0.0)) / (std::max(c2, 1.0) - std::min(c1, 0.0));
    return IoU;
}

V3D point_triangulation(const V2D& p1, const CameraView& view1,
                        const V2D& p2, const CameraView& view2) 
{
    V3D C1 = view1.pose.center();
    V3D C2 = view2.pose.center();
    V3D n1e = view1.ray_direction(p1);
    V3D n2e = view2.ray_direction(p2);
    M2D A; A << n1e.dot(n1e), -n1e.dot(n2e), -n2e.dot(n1e), n2e.dot(n2e);
    V2D b; b(0) = n1e.dot(C2 - C1); b(1) = n2e.dot(C1 - C2);
    V2D res = A.ldlt().solve(b);
    V3D point = 0.5 * (n1e * res[0] + C1 + n2e * res[1] + C2);
    return point;
}

// Triangulating endpoints for triangulation
Line3d triangulate_endpoints(const Line2d& l1, const CameraView& view1,
                             const Line2d& l2, const CameraView& view2) 
{
    V3D pstart = point_triangulation(l1.start, view1, l2.start, view2);
    double z_start = view1.pose.projdepth(pstart);
    V3D pend = point_triangulation(l1.end, view1, l2.end, view2);
    double z_end = view1.pose.projdepth(pend);
    return Line3d(pstart, pend, 1.0, z_start, z_end);
}

// Asymmetric perspective to (view1, l1)
// Triangulation by plane intersection
Line3d triangulate(const Line2d& l1, const CameraView& view1,
                   const Line2d& l2, const CameraView& view2) 
{
    V3D c1_start = view1.ray_direction(l1.start);
    V3D c1_end = view1.ray_direction(l1.end);
    V3D c2_start = view2.ray_direction(l2.start);
    V3D c2_end = view2.ray_direction(l2.end);
    V3D B = view2.pose.center() - view1.pose.center();

    // start point
    M3D A_start; A_start << c1_start, -c2_start, -c2_end;
    auto res_start = A_start.inverse() * B;
    double z_start = res_start[0];
    V3D l3d_start = c1_start * z_start + view1.pose.center();

    // end point
    M3D A_end; A_end << c1_end, -c2_start, -c2_end;
    auto res_end = A_end.inverse() * B;
    double z_end = res_end[0];
    V3D l3d_end = c1_end * z_end + view1.pose.center();
    
    // check
    if (z_start < EPS || z_end < EPS)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
    double d21, d22;
    d21 = view2.pose.projdepth(l3d_start);
    d22 = view2.pose.projdepth(l3d_end);
    if (d21 < EPS || d22 < EPS)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);

    // return the triangulated line
    if (std::isnan(l3d_start[0]))
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0); // give it a -1.0 IoU
    else
        return Line3d(l3d_start, l3d_end, 1.0, z_start, z_end);
}

// Asymmetric perspective to (view1, l1)
// Triangulation with known direction
Line3d triangulate_with_direction(const Line2d& l1, const CameraView& view1,
                                  const Line2d& l2, const CameraView& view2,
                                  const V3D& direction) 
{
    const M3D K1_inv = view1.K_inv();
    const M3D R1 = view1.R();
    const V3D T1 = view1.T();

    const M3D K2_inv = view2.K_inv();
    const M3D R2 = view2.R();
    const V3D T2 = view2.T();

    // Step 1: project direction onto plane 1
    V3D n1 = getNormalDirection(l1, view1);
    V3D direc = direction - (n1.dot(direction)) * n1;
    if (direc.norm() < EPS)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
    direc = direc.normalized();

    // Step 2: parameterize on plane 1 (a1s * d1s - a1e * d1e = 0)
    V3D perp_direc = n1.cross(direc);
    V3D v1s = R1.transpose() * K1_inv * V3D(l1.start[0], l1.start[1], 1);
    double a1s = v1s.dot(perp_direc);
    V3D v1e = R1.transpose() * K1_inv * V3D(l1.end[0], l1.end[1], 1);
    double a1e = v1e.dot(perp_direc);
    const double MIN_VALUE = 0.001;
    if (a1s < 0) {
        a1s *= -1; a1e *= -1;
    }
    if (a1s < MIN_VALUE || a1e < MIN_VALUE)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);

    // Step 3: min [(c1s * d1s - b)^2 + (c1e * d1e - b)^2]
    V3D C1 = -R1.transpose() * T1;
    V3D C2 = -R2.transpose() * T2;
    V3D n2 = getNormalDirection(l2, view2);
    double c1s = n2.dot(v1s);
    double c1e = n2.dot(v1e);
    double b = n2.dot(C2 - C1);
    
    // Optimal solution
    double c1 = c1s;
    double c2 = c1e * a1s / a1e;
    double d1s_num = (c1 + c2) * b;
    double d1s_denom = (c1 * c1  + c2 * c2);
    double d1s = d1s_num / d1s_denom;
    double d1e = d1s * a1s / a1e;

    // check
    if (d1s < EPS || d1e < EPS)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
    V3D lstart = d1s * v1s + C1;
    V3D lend = d1e * v1e + C1;
    double d21, d22;
    d21 = view2.pose.projdepth(lstart);
    d22 = view2.pose.projdepth(lend);
    if (d21 < EPS || d22 < EPS)
        return Line3d(V3D(0, 0, 0), V3D(1, 1, 1), -1.0);
    return Line3d(lstart, lend, 1.0, d1s, d1e);
}

} // namespace triangulation

} // namespace limap

