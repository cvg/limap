#include "limap/geometry/groups.h"
#include "limap/geometry/groups/cone.h"
#include "limap/geometry/groups/cuboid.h"
#include "limap/geometry/groups/cylinder.h"
#include "limap/geometry/groups/ellipsoid.h"
#include "limap/geometry/groups/plane.h"
#include "limap/geometry/groups/sphere.h"
#include "limap/geometry/groups/vp.h"

#include <colmap/util/logging.h>

namespace limap {

std::unique_ptr<BaseGroup> GetGroup(GroupType type) {
  switch (type) {
  case GroupType::VP:
    return std::make_unique<VPGroup>();
  case GroupType::PLANE:
    return std::make_unique<PlaneGroup>();
  case GroupType::SPHERE:
    return std::make_unique<SphereGroup>();
  case GroupType::CYLINDER:
    return std::make_unique<CylinderGroup>();
  case GroupType::ELLIPSOID:
    return std::make_unique<EllipsoidGroup>();
  case GroupType::CUBOID:
    return std::make_unique<CuboidGroup>();
  case GroupType::CONE:
    return std::make_unique<ConeGroup>();
  case GroupType::INVALID:
    LOG(WARNING) << "GetGroup: Cannot create group for INVALID group type";
    return nullptr;
  }
  return nullptr;
}

size_t GetNumParamsIn2DByGroupType(const GroupType type) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(FATAL) << "Group type has not been set";
  }
  return impl->GetNumParamsIn2D();
}

size_t GetNumParamsIn3DByGroupType(const GroupType type) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(FATAL) << "Group type has not been set";
  }
  return impl->GetNumParamsIn3D();
}

void NormalizeGroupParams3D(GroupType type, double *params) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(WARNING)
        << "NormalizeGroupParams3D: Cannot get implementation for group type";
    return;
  }
  impl->NormalizeParams3D(params);
}

bool CheckGroupParams3D(GroupType type, const double *params, double tol) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(WARNING)
        << "CheckGroupParams3D: Cannot get implementation for group type";
    return false;
  }
  return impl->CheckParams3D(params, tol);
}

std::vector<double> GetDefaultGroupParams2D(GroupType type) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(WARNING)
        << "GetDefaultGroupParams2D: Cannot get implementation for group type";
    return {};
  }
  return impl->GetDefaultParams2D();
}

std::vector<double> GetDefaultGroupParams3D(GroupType type) {
  auto impl = GetGroup(type);
  if (!impl) {
    LOG(WARNING)
        << "GetDefaultGroupParams3D: Cannot get implementation for group type";
    return {};
  }
  return impl->GetDefaultParams3D();
}

} // namespace limap
