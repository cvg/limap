#pragma once

#include "limap/geometry/groups.h"
#include "limap/util/types.h"

#include <optional>
#include <vector>

namespace limap {

struct AssociatedFeature2d {
  feature2D_t idx;
  double w = 1.0;

  // Constructors
  AssociatedFeature2d() = default;
  AssociatedFeature2d(feature2D_t idx_, double w_ = 1.0);
};

class Group2d {
public:
  GroupType type = GroupType::INVALID;
  std::vector<AssociatedFeature2d> points;
  std::vector<AssociatedFeature2d> lines;
  group3D_t group3D_id = kInvalidGroup3dId;

  // Constructors
  Group2d();
  explicit Group2d(GroupType t);
  Group2d(GroupType t, const std::vector<AssociatedFeature2d> &pts,
          const std::vector<AssociatedFeature2d> &lns = {});

  // Get max id. Return null if empty
  std::optional<feature2D_t> GetMaxPointId() const;
  std::optional<feature2D_t> GetMaxLineId() const;

  // Methods to add points/lines
  void AddPoint(const AssociatedFeature2d &pt);
  void AddLine(const AssociatedFeature2d &ln);
  void AddPoints(const std::vector<AssociatedFeature2d> &pts);
  void AddLines(const std::vector<AssociatedFeature2d> &lns);

  void AddPoint(const feature2D_t &f);
  void AddLine(const feature2D_t &f);
  void AddPoints(const std::vector<feature2D_t> &feats,
                 const std::vector<double> &weights = std::vector<double>());
  void AddLines(const std::vector<feature2D_t> &feats,
                const std::vector<double> &weights = std::vector<double>());

  // params accessor
  const std::vector<double> &GetParams() const;
  void SetParams(const std::vector<double> &new_params);

private:
  std::vector<double> params; // optional parameters
};

} // namespace limap
