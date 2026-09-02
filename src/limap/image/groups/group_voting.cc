#include "limap/image/groups/group_voting.h"

#include <utility>
#include <vector>

namespace limap {

Eigen::MatrixX2i
MatchGroupsByVoting(const GroupVotingOptions &options,
                    const Eigen::Ref<const Eigen::MatrixX2i> &point_matches,
                    const Eigen::Ref<const Eigen::MatrixX2i> &line_matches,
                    const Structure2d &structure1,
                    const Structure2d &structure2,
                    const std::set<int> &matched_groups1,
                    const std::set<int> &matched_groups2) {

  const size_t N1 = structure1.NumGroups();
  const size_t N2 = structure2.NumGroups();

  // Early return if no groups
  if (N1 == 0 || N2 == 0) {
    return Eigen::MatrixX2i(0, 2);
  }

  // Determine which group types have already been matched
  // Skip voting for entire group types that have any matches
  std::set<GroupType> matched_types;
  for (int gid : matched_groups1) {
    if (gid >= 0 && static_cast<size_t>(gid) < N1) {
      matched_types.insert(structure1.Group(gid).type);
    }
  }
  for (int gid : matched_groups2) {
    if (gid >= 0 && static_cast<size_t>(gid) < N2) {
      matched_types.insert(structure2.Group(gid).type);
    }
  }

  // Early return if all group types are already matched
  if (matched_types.size() > 0) {
    bool all_types_matched = true;
    for (size_t i = 0; i < N1; ++i) {
      if (matched_types.count(structure1.Group(i).type) == 0) {
        all_types_matched = false;
        break;
      }
    }
    if (all_types_matched) {
      return Eigen::MatrixX2i(0, 2);
    }
  }

  // Build feature-to-group lookups
  // point_to_groups[pid] = vector of {gid, weight}
  const size_t num_points1 = structure1.NumPoints();
  const size_t num_points2 = structure2.NumPoints();
  const size_t num_lines1 = structure1.NumLines();
  const size_t num_lines2 = structure2.NumLines();

  std::vector<std::vector<std::pair<int, double>>> point_to_groups1(
      num_points1);
  std::vector<std::vector<std::pair<int, double>>> point_to_groups2(
      num_points2);
  std::vector<std::vector<std::pair<int, double>>> line_to_groups1(num_lines1);
  std::vector<std::vector<std::pair<int, double>>> line_to_groups2(num_lines2);

  for (size_t gid = 0; gid < N1; ++gid) {
    const auto &group = structure1.Group(gid);
    for (const auto &pt : group.points) {
      if (pt.idx < num_points1) {
        double w = options.use_feature_weights ? pt.w : 1.0;
        point_to_groups1[pt.idx].push_back({static_cast<int>(gid), w});
      }
    }
    for (const auto &ln : group.lines) {
      if (ln.idx < num_lines1) {
        double w = options.use_feature_weights ? ln.w : 1.0;
        line_to_groups1[ln.idx].push_back({static_cast<int>(gid), w});
      }
    }
  }

  for (size_t gid = 0; gid < N2; ++gid) {
    const auto &group = structure2.Group(gid);
    for (const auto &pt : group.points) {
      if (pt.idx < num_points2) {
        double w = options.use_feature_weights ? pt.w : 1.0;
        point_to_groups2[pt.idx].push_back({static_cast<int>(gid), w});
      }
    }
    for (const auto &ln : group.lines) {
      if (ln.idx < num_lines2) {
        double w = options.use_feature_weights ? ln.w : 1.0;
        line_to_groups2[ln.idx].push_back({static_cast<int>(gid), w});
      }
    }
  }

  // Initialize vote matrices
  Eigen::MatrixXd vote_score = Eigen::MatrixXd::Zero(N1, N2);
  Eigen::MatrixXi vote_count = Eigen::MatrixXi::Zero(N1, N2);

  // Accumulate votes from point matches
  for (int i = 0; i < point_matches.rows(); ++i) {
    const int p1 = point_matches(i, 0);
    const int p2 = point_matches(i, 1);

    if (p1 < 0 || static_cast<size_t>(p1) >= num_points1 || p2 < 0 ||
        static_cast<size_t>(p2) >= num_points2) {
      continue;
    }

    for (const auto &[g1, w1] : point_to_groups1[p1]) {
      const auto type1 = structure1.Group(g1).type;
      // Skip if this group type has already been matched
      if (matched_types.count(type1))
        continue;
      for (const auto &[g2, w2] : point_to_groups2[p2]) {
        // Only match groups of the same type
        if (structure2.Group(g2).type != type1)
          continue;
        vote_score(g1, g2) += options.point_weight * w1 * w2;
        vote_count(g1, g2) += 1;
      }
    }
  }

  // Accumulate votes from line matches
  for (int i = 0; i < line_matches.rows(); ++i) {
    const int l1 = line_matches(i, 0);
    const int l2 = line_matches(i, 1);

    if (l1 < 0 || static_cast<size_t>(l1) >= num_lines1 || l2 < 0 ||
        static_cast<size_t>(l2) >= num_lines2) {
      continue;
    }

    for (const auto &[g1, w1] : line_to_groups1[l1]) {
      const auto type1 = structure1.Group(g1).type;
      // Skip if this group type has already been matched
      if (matched_types.count(type1))
        continue;
      for (const auto &[g2, w2] : line_to_groups2[l2]) {
        // Only match groups of the same type
        if (structure2.Group(g2).type != type1)
          continue;
        vote_score(g1, g2) += options.line_weight * w1 * w2;
        vote_count(g1, g2) += 1;
      }
    }
  }

  // Mutual nearest neighbor on vote_score
  Eigen::VectorXi nn12(N1); // best match for each group in img1
  Eigen::VectorXi nn21(N2); // best match for each group in img2

  for (size_t i = 0; i < N1; ++i) {
    Eigen::Index max_idx;
    vote_score.row(i).maxCoeff(&max_idx);
    nn12(i) = static_cast<int>(max_idx);
  }

  for (size_t j = 0; j < N2; ++j) {
    Eigen::Index max_idx;
    vote_score.col(j).maxCoeff(&max_idx);
    nn21(j) = static_cast<int>(max_idx);
  }

  // Filter by mutual consistency + vote_count threshold
  std::vector<std::pair<int, int>> matches;
  for (size_t g1 = 0; g1 < N1; ++g1) {
    const auto type1 = structure1.Group(g1).type;

    // Skip if this group type has already been matched
    if (matched_types.count(type1))
      continue;

    const int g2 = nn12(g1);

    // Check type match, mutual consistency and vote threshold
    if (structure2.Group(g2).type == type1 &&
        nn21(g2) == static_cast<int>(g1) &&
        vote_count(g1, g2) >= options.min_num_votes) {
      matches.push_back({static_cast<int>(g1), g2});
    }
  }

  // Convert to Eigen matrix
  Eigen::MatrixX2i result(matches.size(), 2);
  for (size_t i = 0; i < matches.size(); ++i) {
    result(i, 0) = matches[i].first;
    result(i, 1) = matches[i].second;
  }

  return result;
}

} // namespace limap
