#include "limap/scene/group2d.h"

#include <colmap/util/logging.h>

namespace limap {

namespace {
std::optional<feature2D_t>
GetMaxId(const std::vector<AssociatedFeature2d> &features) {
  if (features.empty())
    return std::nullopt;

  auto max_it = std::max_element(
      features.begin(), features.end(),
      [](const AssociatedFeature2d &a, const AssociatedFeature2d &b) {
        return a.idx < b.idx;
      });
  return max_it->idx;
}
} // namespace

AssociatedFeature2d::AssociatedFeature2d(feature2D_t idx_, double w_)
    : idx(idx_), w(w_) {}

Group2d::Group2d() : type(GroupType::INVALID) {}

Group2d::Group2d(GroupType t) : type(t) { params = GetDefaultGroupParams2D(t); }

Group2d::Group2d(GroupType t, const std::vector<AssociatedFeature2d> &pts,
                 const std::vector<AssociatedFeature2d> &lns)
    : type(t), points(pts), lines(lns) {
  if (t == GroupType::VP) {
    THROW_CHECK(points.empty())
        << "Vanishing points can only linked with line features";
  }
  params = GetDefaultGroupParams2D(t);
}

std::optional<feature2D_t> Group2d::GetMaxPointId() const {
  return GetMaxId(points);
}

std::optional<feature2D_t> Group2d::GetMaxLineId() const {
  return GetMaxId(lines);
}

void Group2d::AddPoint(const AssociatedFeature2d &pt) { points.push_back(pt); }

void Group2d::AddLine(const AssociatedFeature2d &ln) { lines.push_back(ln); }

void Group2d::AddPoints(const std::vector<AssociatedFeature2d> &pts) {
  points.insert(points.end(), pts.begin(), pts.end());
}

void Group2d::AddLines(const std::vector<AssociatedFeature2d> &lns) {
  lines.insert(lines.end(), lns.begin(), lns.end());
}

void Group2d::AddPoint(const feature2D_t &f) { points.emplace_back(f, 1.0); }

void Group2d::AddLine(const feature2D_t &f) { lines.emplace_back(f, 1.0); }

void Group2d::AddPoints(const std::vector<feature2D_t> &feats,
                        const std::vector<double> &weights) {
  if (!weights.empty() && feats.size() != weights.size()) {
    throw std::runtime_error(
        "AddPoints: feats and weights must have same size");
  }
  for (size_t i = 0; i < feats.size(); ++i) {
    double weight = 1.0;
    if (!weights.empty()) {
      weight = weights[i];
    }
    points.emplace_back(feats[i], weight);
  }
}

void Group2d::AddLines(const std::vector<feature2D_t> &feats,
                       const std::vector<double> &weights) {
  if (!weights.empty() && feats.size() != weights.size()) {
    throw std::runtime_error(
        "AddPoints: feats and weights must have same size");
  }
  for (size_t i = 0; i < feats.size(); ++i) {
    double weight = 1.0;
    if (!weights.empty()) {
      weight = weights[i];
    }
    lines.emplace_back(feats[i], weight);
  }
}

const std::vector<double> &Group2d::GetParams() const { return params; }

void Group2d::SetParams(const std::vector<double> &new_params) {
  const size_t expected_size = GetNumParamsIn2DByGroupType(type);

  if (new_params.size() != expected_size) {
    throw std::invalid_argument(
        "Group2d::SetParams: parameter size mismatch (got " +
        std::to_string(new_params.size()) + ", expected " +
        std::to_string(expected_size) + ")");
  }

  params = new_params;
}

} // namespace limap
