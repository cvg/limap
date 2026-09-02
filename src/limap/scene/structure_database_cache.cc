#include "limap/scene/structure_database_cache.h"

namespace limap {

std::shared_ptr<StructureDatabaseCache>
StructureDatabaseCache::Create(const StructureDatabase &structure_db) {
  auto cache = std::make_shared<StructureDatabaseCache>();

  // Read all Structure2d from structure database
  cache->structures2d_ = structure_db.ReadAllStructures2d();

  // Construct structure correspondence graph
  cache->structure_correspondence_graph_ =
      std::make_shared<class StructureCorrespondenceGraph>();
  for (const auto &[image_id, _] : cache->structures2d_) {
    size_t num_lines = cache->Structure2d(image_id).NumLines();
    cache->structure_correspondence_graph_->LineGraph().AddImage(image_id,
                                                                 num_lines);
    size_t num_groups = cache->Structure2d(image_id).NumGroups();
    cache->structure_correspondence_graph_->GroupGraph().AddImage(image_id,
                                                                  num_groups);
  }
  std::vector<std::pair<colmap::image_pair_t, LineMatches>> line_matches =
      structure_db.ReadAllLineMatches();
  for (const auto &[pair_id, matches] : line_matches) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    cache->structure_correspondence_graph_->AddLineMatches(image_id1, image_id2,
                                                           matches);
  }
  cache->structure_correspondence_graph_->LineGraph().Finalize();
  std::vector<std::pair<colmap::image_pair_t, GroupMatches>> group_matches =
      structure_db.ReadAllGroupMatches();
  for (const auto &[pair_id, matches] : group_matches) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    cache->structure_correspondence_graph_->AddGroupMatches(image_id1,
                                                            image_id2, matches);
  }
  cache->structure_correspondence_graph_->GroupGraph().Finalize();
  return cache;
}

} // namespace limap
