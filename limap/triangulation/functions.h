#ifndef LIMAP_TRIANGULATION_FUNCTIONS_H_
#define LIMAP_TRIANGULATION_FUNCTIONS_H_

#include "base/camera_view.h"
#include "base/infinite_line.h"
#include "base/linebase.h"
#include "util/types.h"
#include <tuple>

namespace limap {

namespace triangulation {

bool test_line_inside_ranges(const Line3d &line,
                             const std::pair<V3D, V3D> &ranges);

V3D getNormalDirection(const Line2d &l, const CameraView &view);

V3D getDirectionFromVP(const V3D &vp, const CameraView &view);

// weak epipolar constraints
M3D compute_essential_matrix(const CameraView &view1, const CameraView &view2);
M3D compute_fundamental_matrix(const CameraView &view1,
                               const CameraView &view2);

// intersect epipolar lines with the matched line on image 2
double compute_epipolar_IoU(const Line2d &l1, const CameraView &view1,
                            const Line2d &l2, const CameraView &view2);

// point triangulation
std::pair<V3D, bool> triangulate_point(const V2D &p1, const CameraView &view1,
                                       const V2D &p2, const CameraView &view2);

Eigen::Matrix3d
point_triangulation_covariance(const V2D &p1, const CameraView &view1,
                               const V2D &p2, const CameraView &view2,
                               const Eigen::Matrix4d &covariance);

// Triangulating endpoints for triangulation
Line3d triangulate_line_by_endpoints(const Line2d &l1, const CameraView &view1,
                                     const Line2d &l2, const CameraView &view2);

// Asymmetric perspective to (view1, l1)
// Triangulation by plane intersection
std::pair<Line3d, bool> line_triangulation(const Line2d &l1,
                                           const CameraView &view1,
                                           const Line2d &l2,
                                           const CameraView &view2);

M6D line_triangulation_covariance(const Line2d &l1, const CameraView &view1,
                                  const Line2d &l2, const CameraView &view2,
                                  const M8D &covariance);

// Asymmetric perspective to (view1, l1)
// Algebraic line triangulation
Line3d triangulate_line(const Line2d &l1, const CameraView &view1,
                        const Line2d &l2, const CameraView &view2);

// unproject endpoints with known infinite line
Line3d triangulate_line_with_infinite_line(const Line2d &l1,
                                           const CameraView &view1,
                                           const InfiniteLine3d &inf_line);

// Asymmetric perspective to (view1, l1)
// Triangulation with a known point
Line3d triangulate_line_with_one_point(const Line2d &l1,
                                       const CameraView &view1,
                                       const Line2d &l2,
                                       const CameraView &view2,
                                       const V3D &point);

// Asymmetric perspective to (view1, l1)
// Triangulation with known direction
Line3d triangulate_line_with_direction(const Line2d &l1,
                                       const CameraView &view1,
                                       const Line2d &l2,
                                       const CameraView &view2,
                                       const V3D &direction);

} // namespace triangulation

} // namespace limap

#endif
