#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <colmap/math/union_find.h>
#include <colmap/scene/correspondence_graph.h>
#include <colmap/util/base_controller.h>

#include "limap/geometry/groups.h"
#include "limap/scene/group2d.h"
#include "limap/scene/group3d.h"
#include "limap/scene/holistic_reconstruction.h"
#include "limap/util/types.h"

namespace limap {

struct GlobalGroupTriangulationOptions {
  // Group types to triangulate. If empty, triangulates all valid group types.
  std::vector<GroupType> group_types = {};

  // RANSAC inlier threshold for group parameter initialization, per group type.
  // Units depend on group type: radians for VP, world units for Plane/Sphere.
  // If a group type is not in the map or its value is <= 0, uses the group
  // type's default threshold.
  std::map<GroupType, double> init_ransac_thresh = {};

  // minimum cluster size
  int min_cluster_size = 3;

  // 2D group connectivity threshold: minimum total shared 3D features
  // (sum of shared points + shared lines) between two 2D groups.
  int min_shared_features_for_group_connection = 2;

  // 3D feature association threshold: minimum number of track elements
  // that must contain a 3D feature (point or line) to associate it with the
  // 3D group.
  int min_num_supports_for_association = 2;

  // === Group Verification Options ===
  // Enable group verification after parameter initialization.
  // Groups that fail verification are rejected (not added to groups3D).
  bool enable_verification = true;

  // Inlier threshold for verification, per group type.
  // Units depend on group type: degrees for VP, pixels for Plane/Sphere/etc.
  // If a group type is not in the map or its value is <= 0, uses the group
  // type's default threshold.
  std::map<GroupType, double> verification_threshold = {};

  // Minimum inlier count to validate group.
  size_t verification_min_num_inliers = 3;

  // Minimum inlier ratio to validate group.
  double verification_min_inlier_ratio = 0.5;

  // For multi-view features: ratio of observations that must pass.
  // Feature is inlier if >= obs_inlier_ratio of its views pass.
  double verification_obs_inlier_ratio = 0.8;

  // Whether to filter outlier associations from validated groups.
  bool verification_filter_outliers = true;

  // Default thresholds when verification_threshold is not set for a group type.
  double verification_default_vp_threshold = 10.0;    // degrees
  double verification_default_reproj_threshold = 3.0; // pixels

  // Pixel uncertainty threshold for line classification in
  // GlobalTriangulateStructure. Lines with pixel uncertainty (std dev) above
  // this threshold are classified as unreliable and refined separately in
  // 2-step BA. Set to 0 to disable classification.
  double pixel_uncertainty_threshold = 30.0;

  bool Check() const {
    return min_cluster_size >= 1 &&
           min_shared_features_for_group_connection >= 1 &&
           min_num_supports_for_association >= 1;
  }

  // Returns effective group types to process (all valid types if empty).
  std::vector<GroupType> GetEffectiveGroupTypes() const {
    if (!group_types.empty()) {
      return group_types;
    }
    return GetAllGroupTypes();
  }

  // Returns the RANSAC threshold for a specific group type.
  // Returns -1.0 if not set (meaning use default).
  double GetInitRansacThresh(GroupType type) const {
    auto it = init_ransac_thresh.find(type);
    if (it != init_ransac_thresh.end()) {
      return it->second;
    }
    return -1.0;
  }

  // Returns the verification threshold for a specific group type.
  // Returns -1.0 if not set (meaning use default).
  double GetVerificationThresh(GroupType type) const {
    auto it = verification_threshold.find(type);
    if (it != verification_threshold.end()) {
      return it->second;
    }
    return -1.0;
  }
};

class GlobalGroupTriangulationController : public colmap::BaseController {
public:
  GlobalGroupTriangulationController(
      const GlobalGroupTriangulationOptions &options,
      const std::shared_ptr<HolisticReconstruction> &recon,
      const colmap::CorrespondenceGraph &group_corr_graph);

  void Run() override;

private:
  void BuildConnectivity();
  void CreateGroup3DObjects();
  void Extract3DPoints(const Node2d &g,
                       std::vector<colmap::point3D_t> &pts) const;
  void Extract3DLines(const Node2d &g, std::vector<line3D_t> &lines) const;
  bool IsValidGroupType(GroupType type) const;

  GlobalGroupTriangulationOptions options_;
  std::set<GroupType> valid_group_types_;

  std::shared_ptr<HolisticReconstruction> recon_;
  const colmap::CorrespondenceGraph &corr_graph_;

  colmap::UnionFind<Node2d, colmap::PairHash> uf_;
};

} // namespace limap
