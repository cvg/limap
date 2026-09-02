#pragma once

#include <ceres/jet.h>

#include "limap/geometry/ceres_line_functions.h"

namespace limap {

// Plane projection: params [a,b,c,d] where ax+by+cz+d=0, ||(a,b,c)||=1
struct PlaneProjection {
  static constexpr int kNumParams = 4;
  static constexpr bool kSupportsLineProjection = true;

  // p_proj = p - (n·p + d) * n
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *plane_params,
                           T *projected_point) {
    T nx = plane_params[0];
    T ny = plane_params[1];
    T nz = plane_params[2];
    T d = plane_params[3];

    T dist = nx * point3D[0] + ny * point3D[1] + nz * point3D[2] + d;

    projected_point[0] = point3D[0] - dist * nx;
    projected_point[1] = point3D[1] - dist * ny;
    projected_point[2] = point3D[2] - dist * nz;
  }

  // Project Plücker line to plane. Direct formula without computing point.
  // m' = (1 - α²/||d||²)m + (αγ/||d||²)d + (w - δ/||d||²)(d × n)
  template <typename T>
  static void ProjectLine(const T *dvec, const T *mvec, const T *plane_params,
                          T *proj_dvec, T *proj_mvec) {
    T nx = plane_params[0];
    T ny = plane_params[1];
    T nz = plane_params[2];
    T w = plane_params[3];

    T d_norm_sq =
        dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2] + T(1e-12);
    T alpha = dvec[0] * nx + dvec[1] * ny + dvec[2] * nz;

    proj_dvec[0] = dvec[0] - alpha * nx;
    proj_dvec[1] = dvec[1] - alpha * ny;
    proj_dvec[2] = dvec[2] - alpha * nz;

    T dxn[3];
    dxn[0] = dvec[1] * nz - dvec[2] * ny;
    dxn[1] = dvec[2] * nx - dvec[0] * nz;
    dxn[2] = dvec[0] * ny - dvec[1] * nx;

    T gamma = mvec[0] * nx + mvec[1] * ny + mvec[2] * nz;
    T delta = dxn[0] * mvec[0] + dxn[1] * mvec[1] + dxn[2] * mvec[2];

    T inv_d = T(1) / d_norm_sq;
    T coeff_m = T(1) - alpha * alpha * inv_d;
    T coeff_d = alpha * gamma * inv_d;
    T coeff_dxn = w - delta * inv_d;

    proj_mvec[0] = coeff_m * mvec[0] + coeff_d * dvec[0] + coeff_dxn * dxn[0];
    proj_mvec[1] = coeff_m * mvec[1] + coeff_d * dvec[1] + coeff_dxn * dxn[1];
    proj_mvec[2] = coeff_m * mvec[2] + coeff_d * dvec[2] + coeff_dxn * dxn[2];
  }
};

// Sphere projection: params [cx,cy,cz,log_r], r = exp(log_r)
struct SphereProjection {
  static constexpr int kNumParams = 4;
  static constexpr bool kSupportsLineProjection = false;

  // p_proj = c + r * normalize(p - c)
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *sphere_params,
                           T *projected_point) {
    T cx = sphere_params[0];
    T cy = sphere_params[1];
    T cz = sphere_params[2];
    T r = ceres::exp(sphere_params[3]);

    T dx = point3D[0] - cx;
    T dy = point3D[1] - cy;
    T dz = point3D[2] - cz;

    T dist = ceres::sqrt(dx * dx + dy * dy + dz * dz + T(1e-12));
    T scale = r / dist;
    projected_point[0] = cx + scale * dx;
    projected_point[1] = cy + scale * dy;
    projected_point[2] = cz + scale * dz;
  }

  template <typename T>
  static void ProjectLine(const T *, const T *, const T *, T *, T *) {
    static_assert(sizeof(T) == 0,
                  "Line projection is not supported for SphereProjection");
  }
};

// Cylinder projection: params [quat(4), wvec(2), log_r(1)]
// Axis encoded via MinimalInfiniteLine3d, r = exp(log_r)
struct CylinderProjection {
  static constexpr int kNumParams = 7;
  static constexpr bool kSupportsLineProjection = false;

  // Project to cylinder surface: radial projection to distance r from axis
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *cylinder_params,
                           T *projected_point) {
    T qx = cylinder_params[0];
    T qy = cylinder_params[1];
    T qz = cylinder_params[2];
    T qw = cylinder_params[3];

    // Direction d = R.col(0) from quaternion
    T d[3];
    d[0] = T(1) - T(2) * (qy * qy + qz * qz);
    d[1] = T(2) * (qx * qy + qz * qw);
    d[2] = T(2) * (qx * qz - qy * qw);

    // Moment direction m_dir = R.col(1)
    T m_dir[3];
    m_dir[0] = T(2) * (qx * qy - qz * qw);
    m_dir[1] = T(1) - T(2) * (qx * qx + qz * qz);
    m_dir[2] = T(2) * (qy * qz + qx * qw);

    // Moment magnitude from wvec
    T w0 = cylinder_params[4];
    T w1 = cylinder_params[5];
    T m_mag = w1 / w0;

    T m[3];
    m[0] = m_mag * m_dir[0];
    m[1] = m_mag * m_dir[1];
    m[2] = m_mag * m_dir[2];

    // Point on axis: p_axis = d × m
    T p_axis[3];
    p_axis[0] = d[1] * m[2] - d[2] * m[1];
    p_axis[1] = d[2] * m[0] - d[0] * m[2];
    p_axis[2] = d[0] * m[1] - d[1] * m[0];

    T r = ceres::exp(cylinder_params[6]);

    T v[3];
    v[0] = point3D[0] - p_axis[0];
    v[1] = point3D[1] - p_axis[1];
    v[2] = point3D[2] - p_axis[2];

    T v_dot_d = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];

    T v_perp[3];
    v_perp[0] = v[0] - v_dot_d * d[0];
    v_perp[1] = v[1] - v_dot_d * d[1];
    v_perp[2] = v[2] - v_dot_d * d[2];

    T dist_to_axis = ceres::sqrt(v_perp[0] * v_perp[0] + v_perp[1] * v_perp[1] +
                                 v_perp[2] * v_perp[2] + T(1e-12));

    T scale = r / dist_to_axis;
    projected_point[0] = p_axis[0] + v_dot_d * d[0] + scale * v_perp[0];
    projected_point[1] = p_axis[1] + v_dot_d * d[1] + scale * v_perp[1];
    projected_point[2] = p_axis[2] + v_dot_d * d[2] + scale * v_perp[2];
  }

  template <typename T>
  static void ProjectLine(const T *, const T *, const T *, T *, T *) {
    static_assert(sizeof(T) == 0,
                  "Line projection is not supported for CylinderProjection");
  }
};

// Ellipsoid projection: params [quat(4), center(3), log_scales(3)]
// Ellipsoid: (x-c)^T R^T D R (x-c) = 1, D = diag(1/sx^2, 1/sy^2, 1/sz^2)
struct EllipsoidProjection {
  static constexpr int kNumParams = 10;
  static constexpr bool kSupportsLineProjection = false;

  // Scaled-radial projection (fast, not closest point).
  // Transform to local frame, scale to unit sphere, normalize, scale back.
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *ellipsoid_params,
                           T *projected_point) {
    T qx = ellipsoid_params[0];
    T qy = ellipsoid_params[1];
    T qz = ellipsoid_params[2];
    T qw = ellipsoid_params[3];

    T cx = ellipsoid_params[4];
    T cy = ellipsoid_params[5];
    T cz = ellipsoid_params[6];

    T sx = ceres::exp(ellipsoid_params[7]);
    T sy = ceres::exp(ellipsoid_params[8]);
    T sz = ceres::exp(ellipsoid_params[9]);

    // Rotation matrix from quaternion
    T r00 = T(1) - T(2) * (qy * qy + qz * qz);
    T r10 = T(2) * (qx * qy + qz * qw);
    T r20 = T(2) * (qx * qz - qy * qw);
    T r01 = T(2) * (qx * qy - qz * qw);
    T r11 = T(1) - T(2) * (qx * qx + qz * qz);
    T r21 = T(2) * (qy * qz + qx * qw);
    T r02 = T(2) * (qx * qz + qy * qw);
    T r12 = T(2) * (qy * qz - qx * qw);
    T r22 = T(1) - T(2) * (qx * qx + qy * qy);

    // Transform to local frame: p_local = R^T (p - c)
    T dx = point3D[0] - cx;
    T dy = point3D[1] - cy;
    T dz = point3D[2] - cz;

    T px = r00 * dx + r10 * dy + r20 * dz;
    T py = r01 * dx + r11 * dy + r21 * dz;
    T pz = r02 * dx + r12 * dy + r22 * dz;

    // Scale to unit sphere, normalize, scale back
    T nx = px / sx;
    T ny = py / sy;
    T nz = pz / sz;

    T norm = ceres::sqrt(nx * nx + ny * ny + nz * nz + T(1e-12));
    nx /= norm;
    ny /= norm;
    nz /= norm;

    T local_proj_x = nx * sx;
    T local_proj_y = ny * sy;
    T local_proj_z = nz * sz;

    // Transform back to world frame: p_world = R * p_local + c
    projected_point[0] =
        r00 * local_proj_x + r01 * local_proj_y + r02 * local_proj_z + cx;
    projected_point[1] =
        r10 * local_proj_x + r11 * local_proj_y + r12 * local_proj_z + cy;
    projected_point[2] =
        r20 * local_proj_x + r21 * local_proj_y + r22 * local_proj_z + cz;
  }

  template <typename T>
  static void ProjectLine(const T *, const T *, const T *, T *, T *) {
    static_assert(sizeof(T) == 0,
                  "Line projection is not supported for EllipsoidProjection");
  }
};

// Cuboid projection: params [quat(4), d(6)]
// 6 face planes defined by rotation R (from quaternion) and signed offsets d.
// Projects to nearest face plane.
struct CuboidProjection {
  static constexpr int kNumParams = 10;
  static constexpr bool kSupportsLineProjection = true;

  // Extract R columns from cuboid quaternion params
  template <typename T>
  static void QuatToRColumns(const T *cuboid_params, T *r0, T *r1, T *r2) {
    T qx = cuboid_params[0];
    T qy = cuboid_params[1];
    T qz = cuboid_params[2];
    T qw = cuboid_params[3];

    r0[0] = T(1) - T(2) * (qy * qy + qz * qz);
    r0[1] = T(2) * (qx * qy + qz * qw);
    r0[2] = T(2) * (qx * qz - qy * qw);

    r1[0] = T(2) * (qx * qy - qz * qw);
    r1[1] = T(1) - T(2) * (qx * qx + qz * qz);
    r1[2] = T(2) * (qy * qz + qx * qw);

    r2[0] = T(2) * (qx * qz + qy * qw);
    r2[1] = T(2) * (qy * qz - qx * qw);
    r2[2] = T(1) - T(2) * (qx * qx + qy * qy);
  }

  // Compute signed distance from a 3D point to each of the 6 cuboid faces.
  // r0, r1, r2 are the R columns; cuboid_params[4:10] are the d offsets.
  template <typename T>
  static void ComputeFaceDistances(const T *point, const T *r0, const T *r1,
                                   const T *r2, const T *cuboid_params,
                                   T *dist) {
    T dot0 = r0[0] * point[0] + r0[1] * point[1] + r0[2] * point[2];
    T dot1 = r1[0] * point[0] + r1[1] * point[1] + r1[2] * point[2];
    T dot2 = r2[0] * point[0] + r2[1] * point[1] + r2[2] * point[2];

    dist[0] = dot0 - cuboid_params[4];  // +x face
    dist[1] = -dot0 - cuboid_params[5]; // -x face
    dist[2] = dot1 - cuboid_params[6];  // +y face
    dist[3] = -dot1 - cuboid_params[7]; // -y face
    dist[4] = dot2 - cuboid_params[8];  // +z face
    dist[5] = -dot2 - cuboid_params[9]; // -z face
  }

  // Find the face index with minimum |distance|
  template <typename T> static int FindNearestFace(const T *dist) {
    int min_face = 0;
    T min_abs = ceres::abs(dist[0]);
    for (int i = 1; i < 6; ++i) {
      T abs_d = ceres::abs(dist[i]);
      if (abs_d < min_abs) {
        min_abs = abs_d;
        min_face = i;
      }
    }
    return min_face;
  }

  // Get the outward normal and signed distance for a given face index.
  template <typename T>
  static void GetFaceNormalAndDist(int face, const T *r0, const T *r1,
                                   const T *r2, const T *dist, T *normal,
                                   T *signed_dist) {
    switch (face) {
    case 0:
      normal[0] = r0[0];
      normal[1] = r0[1];
      normal[2] = r0[2];
      *signed_dist = dist[0];
      break;
    case 1:
      normal[0] = -r0[0];
      normal[1] = -r0[1];
      normal[2] = -r0[2];
      *signed_dist = dist[1];
      break;
    case 2:
      normal[0] = r1[0];
      normal[1] = r1[1];
      normal[2] = r1[2];
      *signed_dist = dist[2];
      break;
    case 3:
      normal[0] = -r1[0];
      normal[1] = -r1[1];
      normal[2] = -r1[2];
      *signed_dist = dist[3];
      break;
    case 4:
      normal[0] = r2[0];
      normal[1] = r2[1];
      normal[2] = r2[2];
      *signed_dist = dist[4];
      break;
    case 5:
    default:
      normal[0] = -r2[0];
      normal[1] = -r2[1];
      normal[2] = -r2[2];
      *signed_dist = dist[5];
      break;
    }
  }

  // Construct PlaneProjection-compatible params [nx, ny, nz, w] for a face.
  // PlaneProjection uses n·p + w = 0, so w = -offset.
  template <typename T>
  static void GetFacePlaneParams(int face, const T *r0, const T *r1,
                                 const T *r2, const T *cuboid_params,
                                 T *plane_params) {
    switch (face) {
    case 0:
      plane_params[0] = r0[0];
      plane_params[1] = r0[1];
      plane_params[2] = r0[2];
      plane_params[3] = -cuboid_params[4];
      break;
    case 1:
      plane_params[0] = -r0[0];
      plane_params[1] = -r0[1];
      plane_params[2] = -r0[2];
      plane_params[3] = -cuboid_params[5];
      break;
    case 2:
      plane_params[0] = r1[0];
      plane_params[1] = r1[1];
      plane_params[2] = r1[2];
      plane_params[3] = -cuboid_params[6];
      break;
    case 3:
      plane_params[0] = -r1[0];
      plane_params[1] = -r1[1];
      plane_params[2] = -r1[2];
      plane_params[3] = -cuboid_params[7];
      break;
    case 4:
      plane_params[0] = r2[0];
      plane_params[1] = r2[1];
      plane_params[2] = r2[2];
      plane_params[3] = -cuboid_params[8];
      break;
    case 5:
    default:
      plane_params[0] = -r2[0];
      plane_params[1] = -r2[1];
      plane_params[2] = -r2[2];
      plane_params[3] = -cuboid_params[9];
      break;
    }
  }

  // Project point onto nearest cuboid face plane.
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *cuboid_params,
                           T *projected_point) {
    T r0[3], r1[3], r2[3];
    QuatToRColumns(cuboid_params, r0, r1, r2);

    T dist[6];
    ComputeFaceDistances(point3D, r0, r1, r2, cuboid_params, dist);

    int min_face = FindNearestFace(dist);

    T normal[3], d;
    GetFaceNormalAndDist(min_face, r0, r1, r2, dist, normal, &d);

    projected_point[0] = point3D[0] - d * normal[0];
    projected_point[1] = point3D[1] - d * normal[1];
    projected_point[2] = point3D[2] - d * normal[2];
  }

  // Project Plücker line onto nearest cuboid face plane.
  // Face selection uses the point on the line closest to the cuboid center.
  // This is geometrically correct because a line on a face is parallel to it,
  // so its distance to that face is constant — any representative point works.
  // The cuboid center is an intrinsic reference that avoids coordinate bias.
  template <typename T>
  static void ProjectLine(const T *dvec, const T *mvec, const T *cuboid_params,
                          T *proj_dvec, T *proj_mvec) {
    T r0[3], r1[3], r2[3];
    QuatToRColumns(cuboid_params, r0, r1, r2);

    // Cuboid center in world frame:
    //   center_local = [(d0-d1)/2, (d2-d3)/2, (d4-d5)/2]
    //   center_world = R * center_local
    T cl0 = (cuboid_params[4] - cuboid_params[5]) * T(0.5);
    T cl1 = (cuboid_params[6] - cuboid_params[7]) * T(0.5);
    T cl2 = (cuboid_params[8] - cuboid_params[9]) * T(0.5);

    T center[3];
    center[0] = r0[0] * cl0 + r1[0] * cl1 + r2[0] * cl2;
    center[1] = r0[1] * cl0 + r1[1] * cl1 + r2[1] * cl2;
    center[2] = r0[2] * cl0 + r1[2] * cl1 + r2[2] * cl2;

    // Closest point on line to cuboid center:
    //   p0 = (d × m) / ||d||²    (a point on the line)
    //   t = (center - p0) · d / ||d||²
    //   p_closest = p0 + t * d
    // Simplifies to: p_closest = (d × m) / ||d||² + (center · d / ||d||²) * d
    T d_norm_sq =
        dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2] + T(1e-12);
    T inv_d2 = T(1) / d_norm_sq;

    // d × m
    T dxm[3];
    dxm[0] = dvec[1] * mvec[2] - dvec[2] * mvec[1];
    dxm[1] = dvec[2] * mvec[0] - dvec[0] * mvec[2];
    dxm[2] = dvec[0] * mvec[1] - dvec[1] * mvec[0];

    T c_dot_d = center[0] * dvec[0] + center[1] * dvec[1] + center[2] * dvec[2];

    T p_closest[3];
    p_closest[0] = dxm[0] * inv_d2 + c_dot_d * inv_d2 * dvec[0];
    p_closest[1] = dxm[1] * inv_d2 + c_dot_d * inv_d2 * dvec[1];
    p_closest[2] = dxm[2] * inv_d2 + c_dot_d * inv_d2 * dvec[2];

    // Select nearest face using p_closest
    T dist[6];
    ComputeFaceDistances(p_closest, r0, r1, r2, cuboid_params, dist);
    int min_face = FindNearestFace(dist);

    // Construct plane params for the selected face and delegate
    T plane_params[4];
    GetFacePlaneParams(min_face, r0, r1, r2, cuboid_params, plane_params);

    PlaneProjection::ProjectLine(dvec, mvec, plane_params, proj_dvec,
                                 proj_mvec);
  }
};

// Cone projection: params [apex(3), direction(3), log_tan_alpha(1)]
// Double cone (both sides of apex): surface is r = |h| * tan(alpha)
struct ConeProjection {
  static constexpr int kNumParams = 7;
  static constexpr bool kSupportsLineProjection = false;

  // Project point onto nearest point on double cone surface
  template <typename T>
  static void ProjectPoint(const T *point3D, const T *cone_params,
                           T *projected_point) {
    T apex_x = cone_params[0];
    T apex_y = cone_params[1];
    T apex_z = cone_params[2];
    T dx = cone_params[3];
    T dy = cone_params[4];
    T dz = cone_params[5];

    T tan_a = ceres::exp(cone_params[6]);
    T tan_a_sq = tan_a * tan_a;
    T cos_a = T(1) / ceres::sqrt(T(1) + tan_a_sq);
    T sin_a = tan_a * cos_a;

    // Vector from apex to point
    T vx = point3D[0] - apex_x;
    T vy = point3D[1] - apex_y;
    T vz = point3D[2] - apex_z;

    // Height along axis (signed)
    T h = vx * dx + vy * dy + vz * dz;

    // Perpendicular component
    T perp_x = vx - h * dx;
    T perp_y = vy - h * dy;
    T perp_z = vz - h * dz;

    T r = ceres::sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z +
                      T(1e-12));

    // In the 2D (h, r) plane, double cone is a V: r = |h| * tan_a
    // Project onto nearer arm
    T t_right = h * cos_a + r * sin_a; // right arm (h > 0 side)
    T t_left = -h * cos_a + r * sin_a; // left arm (h < 0 side)

    T h_proj, r_proj;
    // Use smooth selection: t_right >= t_left iff h >= 0
    if (t_right >= t_left) {
      h_proj = t_right * cos_a;
      r_proj = t_right * sin_a;
    } else {
      h_proj = -t_left * cos_a;
      r_proj = t_left * sin_a;
    }

    // Reconstruct 3D projected point
    T inv_r = T(1) / r;
    projected_point[0] = apex_x + h_proj * dx + r_proj * perp_x * inv_r;
    projected_point[1] = apex_y + h_proj * dy + r_proj * perp_y * inv_r;
    projected_point[2] = apex_z + h_proj * dz + r_proj * perp_z * inv_r;
  }

  template <typename T>
  static void ProjectLine(const T *, const T *, const T *, T *, T *) {
    static_assert(sizeof(T) == 0,
                  "Line projection is not supported for ConeProjection");
  }
};

} // namespace limap
