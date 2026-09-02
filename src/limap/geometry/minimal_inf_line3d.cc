#include "limap/geometry/minimal_inf_line3d.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {

MinimalInfiniteLine3d::MinimalInfiniteLine3d(
    const std::vector<double> &values) {
  THROW_CHECK_EQ(values.size(), 6);
  for (int i = 0; i < 6; ++i)
    data[i] = values[i];
}

MinimalInfiniteLine3d::MinimalInfiniteLine3d(const InfiniteLine3d &inf_line) {
  // [LINK]
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  // [LINK] https://hal.archives-ouvertes.fr/hal-00092589/document

  V3D a = inf_line.d;
  V3D b = inf_line.m;

  // SO(2) part (2D unit vector)
  double w1 = 1.0;
  double w2 = b.norm();
  double denom = V2D(w1, w2).norm();
  double wv0 = w1 / denom;
  double wv1 = w2 / denom;

  // SO(3) part (rotation)
  M3D Q;
  Q.col(0) = a / a.norm();
  if (b.norm() > 1e-8) {
    Q.col(1) = b / b.norm();
    V3D axb = a.cross(b);
    Q.col(2) = axb / axb.norm();
  } else {
    int best_index = 0;
    if (std::abs(a(1)) > std::abs(a(0)))
      best_index = 1;
    if (std::abs(a(2)) > std::abs(a(best_index)))
      best_index = 2;
    int i1 = (best_index + 1) % 3;
    int i2 = (best_index + 2) % 3;
    V3D bprime;
    bprime(i1) = 1.0;
    bprime(i2) = 1.0;
    bprime(best_index) =
        -(a(i1) * bprime(i1) + a(i2) * bprime(i2)) / a(best_index);
    Q.col(1) = bprime / bprime.norm();
    V3D axb = a.cross(bprime);
    Q.col(2) = axb / axb.norm();
  }

  Eigen::Quaterniond q(Q);

  // Store [uvec (x,y,z,w), wvec(0), wvec(1)] continuously
  data[0] = q.x();
  data[1] = q.y();
  data[2] = q.z();
  data[3] = q.w();
  data[4] = wv0;
  data[5] = wv1;
}

InfiniteLine3d MinimalInfiniteLine3d::GetInfiniteLine() const {
  // Extract quaternion and SO(2) vector
  Eigen::Quaterniond q = uvec();
  V2D wv = wvec();

  // Convert back to Plücker representation.
  // The quaternion may have drifted from unit norm during BA optimization
  // (Ceres manifolds project updates but don't re-normalize), so normalize
  // before converting to a rotation matrix.
  q.normalize();
  M3D Q = q.toRotationMatrix();
  V3D d = Q.col(0);
  V3D m = wv(1) / wv(0) * Q.col(1);
  return InfiniteLine3d(d, m, false);
}

} // namespace limap
