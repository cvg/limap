#ifndef LIMAP_BASE_TRANSFORMS_H_
#define LIMAP_BASE_TRANSFORMS_H_

#include "base/camera.h"
#include "base/pose.h"

namespace limap {

class SimilarityTransform3 {
  /*
   * t_prime = R @ (s * t) + T
   */
public:
  SimilarityTransform3() {}
  SimilarityTransform3(V4D qqvec, V3D ttvec, double s = 1.0)
      : qvec(qqvec), tvec(ttvec), scale(s) {}
  SimilarityTransform3(M3D R, V3D T, double s = 1.0) : tvec(T), scale(s) {
    qvec = RotationMatrixToQuaternion(R);
  }
  V4D qvec = V4D(1., 0., 0., 0.);
  V3D tvec = V3D::Zero();
  double scale = 1.0;

  M3D R() const { return QuaternionToRotationMatrix(qvec); }
  V3D T() const { return tvec; }
  double s() const { return scale; }
};

CameraPose pose_similarity_transform(const CameraPose &pose,
                                     const SimilarityTransform3 &transform);

} // namespace limap

#endif
