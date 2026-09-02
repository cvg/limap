#pragma once

#include <colmap/scene/track.h>

#include "limap/geometry/groups.h"
#include "limap/scene/group2d.h"
#include "limap/util/types.h"

#include <optional>
#include <vector>

namespace limap {

struct AssociatedFeature3d {
  feature3D_t idx;
  double w = 1.0;

  // Constructors
  AssociatedFeature3d() = default;
  AssociatedFeature3d(feature3D_t idx_, double w_ = 1.0);
};

class Group3d {
public:
  GroupType type = GroupType::INVALID;
  std::vector<AssociatedFeature3d> points;
  std::vector<AssociatedFeature3d> lines;
  colmap::Track track;

  Group3d();
  explicit Group3d(GroupType t);
  Group3d(GroupType t, const std::vector<double> &p);
  Group3d(GroupType t, const std::vector<double> &p,
          const std::vector<AssociatedFeature3d> &pts,
          const std::vector<AssociatedFeature3d> &lns = {});

  // Methods to add points/lines
  void AddPoint(const AssociatedFeature3d &pt);
  void AddLine(const AssociatedFeature3d &ln);
  void AddPoints(const std::vector<AssociatedFeature3d> &pts);
  void AddLines(const std::vector<AssociatedFeature3d> &lns);

  void AddPoint(const feature3D_t &f);
  void AddLine(const feature3D_t &f);
  void AddPoints(const std::vector<feature3D_t> &feats,
                 const std::vector<double> &weights);
  void AddLines(const std::vector<feature3D_t> &feats,
                const std::vector<double> &weights);

  // params accessor
  const std::vector<double> &GetParams() const;
  std::vector<double> &GetParams();
  double *GetParamsData();
  // Sets params and normalizes them to canonical form
  void SetParams(const std::vector<double> &new_params);

  // Normalizes params in-place to canonical form:
  // - VP: unit 3-vector direction (SphereManifold<3>)
  // - PLANE: (a,b,c,d) where ||(a,b,c)|| = 1, d is signed distance to origin
  //          (ProductManifold<SphereManifold<3>, EuclideanManifold<1>>)
  // Called automatically by constructors and SetParams.
  // During optimization, manifolds maintain these constraints.
  void NormalizeParams();

  // Check if params are valid and normalized (within tolerance).
  // Returns true if valid, false otherwise.
  bool CheckParams(double tol = 1e-6) const;

private:
  std::vector<double> params;
};

} // namespace limap
