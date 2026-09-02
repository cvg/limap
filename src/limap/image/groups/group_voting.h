#pragma once

#include "limap/scene/structure2d.h"
#include <Eigen/Core>
#include <set>
#include <vector>

namespace limap {

struct GroupVotingOptions {
  double point_weight = 1.0;
  double line_weight = 1.0;
  bool use_feature_weights = true;
  int min_num_votes = 3;
};

// Match unmatched groups by voting from matched points and lines.
// Returns Nx2 matrix of group match pairs (local indices within each image).
Eigen::MatrixX2i MatchGroupsByVoting(
    const GroupVotingOptions &options,
    const Eigen::Ref<const Eigen::MatrixX2i> &point_matches,
    const Eigen::Ref<const Eigen::MatrixX2i> &line_matches,
    const Structure2d &structure1, const Structure2d &structure2,
    const std::set<int> &matched_groups1, const std::set<int> &matched_groups2);

} // namespace limap
