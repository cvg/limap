#include "limap/scene/group3d.h"

#include "limap/geometry/groups.h"

namespace limap {

AssociatedFeature3d::AssociatedFeature3d(feature3D_t idx_, double w_)
    : idx(idx_), w(w_) {}

Group3d::Group3d() : type(GroupType::INVALID) {}

Group3d::Group3d(GroupType t) : type(t) { params = GetDefaultGroupParams3D(t); }

Group3d::Group3d(GroupType t, const std::vector<double> &p)
    : type(t), params(p) {
  THROW_CHECK_EQ(GetNumParamsIn3DByGroupType(t), params.size());
  NormalizeParams();
}

Group3d::Group3d(GroupType t, const std::vector<double> &p,
                 const std::vector<AssociatedFeature3d> &pts,
                 const std::vector<AssociatedFeature3d> &lns)
    : type(t), params(p), points(pts), lines(lns) {
  THROW_CHECK_EQ(GetNumParamsIn3DByGroupType(t), params.size());
  if (t == GroupType::VP) {
    THROW_CHECK(points.empty())
        << "Vanishing points can only linked with line features";
  }
  NormalizeParams();
}

void Group3d::AddPoint(const AssociatedFeature3d &pt) { points.push_back(pt); }

void Group3d::AddLine(const AssociatedFeature3d &ln) { lines.push_back(ln); }

void Group3d::AddPoints(const std::vector<AssociatedFeature3d> &pts) {
  points.insert(points.end(), pts.begin(), pts.end());
}

void Group3d::AddLines(const std::vector<AssociatedFeature3d> &lns) {
  lines.insert(lines.end(), lns.begin(), lns.end());
}

void Group3d::AddPoint(const feature3D_t &f) { points.emplace_back(f, 1.0); }

void Group3d::AddLine(const feature3D_t &f) { lines.emplace_back(f, 1.0); }

void Group3d::AddPoints(const std::vector<feature3D_t> &feats,
                        const std::vector<double> &weights) {
  THROW_CHECK_NE(feats.size(), weights.size())
      << "AddPoints: feats and weights must have same size";
  for (size_t i = 0; i < feats.size(); ++i) {
    points.emplace_back(feats[i], weights[i]);
  }
}

void Group3d::AddLines(const std::vector<feature3D_t> &feats,
                       const std::vector<double> &weights) {
  if (feats.size() != weights.size()) {
    throw std::runtime_error("AddLines: feats and weights must have same size");
  }
  for (size_t i = 0; i < feats.size(); ++i) {
    lines.emplace_back(feats[i], weights[i]);
  }
}

const std::vector<double> &Group3d::GetParams() const { return params; }

std::vector<double> &Group3d::GetParams() { return params; }

double *Group3d::GetParamsData() { return params.data(); }

void Group3d::SetParams(const std::vector<double> &new_params) {
  const size_t expected_size = GetNumParamsIn3DByGroupType(type);

  if (new_params.size() != expected_size) {
    throw std::invalid_argument(
        "Group3d::SetParams: parameter size mismatch (got " +
        std::to_string(new_params.size()) + ", expected " +
        std::to_string(expected_size) + ")");
  }

  params = new_params;
  NormalizeParams();
}

void Group3d::NormalizeParams() {
  if (type == GroupType::INVALID || params.empty()) {
    return;
  }
  auto group_impl = GetGroup(type);
  if (group_impl) {
    group_impl->NormalizeParams3D(params.data());
  }
}

bool Group3d::CheckParams(double tol) const {
  if (type == GroupType::INVALID || params.empty()) {
    return false;
  }
  auto group_impl = GetGroup(type);
  if (!group_impl) {
    return false;
  }
  return group_impl->CheckParams3D(params.data(), tol);
}

} // namespace limap
