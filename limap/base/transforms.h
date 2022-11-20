#ifndef LIMAP_BASE_TRANSFORMS_H_
#define LIMAP_BASE_TRANSFORMS_H_

#include <colmap/base/pose.h>

#include "base/camera.h"

namespace limap {

class SimilarityTransform3 {
/*
 * t_prime = R @ (s * t) + T
 */
public:
    SimilarityTransform3() {}
    SimilarityTransform3(V4D qqvec, V3D ttvec, double s = 1.0): qvec(qqvec), tvec(ttvec), scale(s) {}
    SimilarityTransform3(M3D R, V3D T, double s = 1.0): tvec(T), scale(s) { qvec = colmap::RotationMatrixToQuaternion(R); }
    V4D qvec;
    V3D tvec;
    double scale;

    M3D R() const { return colmap::QuaternionToRotationMatrix(qvec); }
    V3D T() const { return tvec; }
    double s() const { return scale; }
};

CameraPose pose_similarity_transform(const CameraPose& pose, const SimilarityTransform3& transform);

} // namespace limap

#endif

