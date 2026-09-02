#pragma once

#include "limap/scene/structure2d.h"
#include "limap/scene/structure_correspondence_graph.h"
#include "limap/scene/structure_database.h"

namespace limap {

class StructureDatabaseCache {
public:
  StructureDatabaseCache() = default;
  static std::shared_ptr<StructureDatabaseCache>
  Create(const StructureDatabase &struct_db);

  inline class Structure2d &Structure2d(colmap::image_t image_id);
  inline const class Structure2d &Structure2d(colmap::image_t image_id) const;
  inline const NodeHashMap<colmap::image_t, class Structure2d> &
  Structures2d() const;

  inline std::shared_ptr<const class StructureCorrespondenceGraph>
  StructureCorrespondenceGraph() const;

private:
  NodeHashMap<colmap::image_t, class Structure2d> structures2d_;
  std::shared_ptr<class StructureCorrespondenceGraph>
      structure_correspondence_graph_;
};

class Structure2d &
StructureDatabaseCache::Structure2d(colmap::image_t image_id) {
  return structures2d_.at(image_id);
}

const class Structure2d &
StructureDatabaseCache::Structure2d(colmap::image_t image_id) const {
  return structures2d_.at(image_id);
}

const NodeHashMap<colmap::image_t, class Structure2d> &
StructureDatabaseCache::Structures2d() const {
  return structures2d_;
}

std::shared_ptr<const class StructureCorrespondenceGraph>
StructureDatabaseCache::StructureCorrespondenceGraph() const {
  THROW_CHECK(structure_correspondence_graph_)
      << "GetStructureCorrespondenceGraph: structure_correspondence_graph_ is "
         "null";
  return structure_correspondence_graph_;
}

} // namespace limap
