#include "base/transforms.h"

namespace limap {

CameraPose pose_similarity_transform(const CameraPose &pose,
                                     const SimilarityTransform3 &transform) {
  M3D new_R = pose.R() * transform.R().transpose();
  V3D new_T = transform.s() * pose.T() - new_R * transform.T();
  return CameraPose(new_R, new_T);
}

} // namespace limap
