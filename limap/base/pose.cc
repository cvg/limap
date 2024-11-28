#include "base/pose.h"

#include <Eigen/Eigenvalues>

namespace limap {

Eigen::Vector4d RotationMatrixToQuaternion(const Eigen::Matrix3d &rot_mat) {
  const Eigen::Quaterniond quat(rot_mat);
  return Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
}

Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d &qvec) {
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(qvec);
  const Eigen::Quaterniond quat(normalized_qvec(0), normalized_qvec(1),
                                normalized_qvec(2), normalized_qvec(3));
  return quat.toRotationMatrix();
}

Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d &qvec) {
  const double norm = qvec.norm();
  if (norm == 0) {
    // We do not just use (1, 0, 0, 0) because that is a constant and when used
    // for automatic differentiation that would lead to a zero derivative.
    return Eigen::Vector4d(1.0, qvec(1), qvec(2), qvec(3));
  } else {
    return qvec / norm;
  }
}

} // namespace limap
