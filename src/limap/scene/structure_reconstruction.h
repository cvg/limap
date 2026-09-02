#pragma once

#include <filesystem>

#include <colmap/scene/reconstruction.h>

#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/line3d.h"
#include "limap/scene/group3d.h"
#include "limap/scene/group3d_with_active_labels.h"
#include "limap/scene/line3d_with_active_labels.h"
#include "limap/scene/structure2d.h"
#include "limap/scene/structure_database_cache.h"
#include "limap/scene/wireframe.h"

namespace limap {

// Options for wireframe 3D construction via voting
// (Defined outside class to allow use as default argument)
struct WireframeVotingOptions {
  int min_num_votes = 3; // Minimum images supporting the edge
  double min_weight_sum =
      1.5; // Minimum accumulated weight (half of min_num_votes)
};

class StructureReconstruction {
public:
  StructureReconstruction(const colmap::Reconstruction &point_recon);

  // Read data from text or binary file. Prefer binary data if it exists.
  void Read(const std::filesystem::path &path);
  void Write(const std::filesystem::path &path) const;

  // Read data from binary/text file.
  void ReadText(const std::filesystem::path &path);
  void ReadBinary(const std::filesystem::path &path);

  // Write data from binary/text file.
  void WriteText(const std::filesystem::path &path) const;
  void WriteBinary(const std::filesystem::path &path) const;

  // Get a reference to the point reconstruction
  inline const colmap::Reconstruction &PointRecon() const;

  inline size_t NumLines3D() const;
  inline size_t NumGroups3D() const;

  inline const class Structure2d &
  Structure2d(const colmap::image_t image_id) const;
  inline class Structure2d &Structure2d(const colmap::image_t image_id);
  inline const NodeHashMap<colmap::image_t, class Structure2d> &
  Structures2d() const;
  inline NodeHashMap<colmap::image_t, class Structure2d> &Structures2d();

  inline const Line3dWithActiveLabels &Line(const line3D_t line_id) const;
  inline Line3dWithActiveLabels &Line(const line3D_t line_id);
  inline const NodeHashMap<line3D_t, Line3dWithActiveLabels> &Lines3D() const;
  inline NodeHashMap<line3D_t, Line3dWithActiveLabels> &Lines3D();

  inline const Group3dWithActiveLabels &Group(const group3D_t group_id) const;
  inline Group3dWithActiveLabels &Group(const group3D_t group_id);
  inline const NodeHashMap<group3D_t, Group3dWithActiveLabels> &
  Groups3D() const;
  inline NodeHashMap<group3D_t, Group3dWithActiveLabels> &Groups3D();

  inline const Wireframe3d &Wireframe() const;
  inline Wireframe3d &Wireframe();

  // Mutation methods for lines3D (aligned with COLMAP Reconstruction interface)
  inline bool ExistsLine3D(line3D_t id) const;
  inline line3D_t AddLine3D(const Line3d &line); // Auto-generates ID
  inline void AddLine3D(line3D_t id, const Line3d &line);
  inline void AddLineObservation(line3D_t line3D_id,
                                 const colmap::TrackElement &track_el);
  inline line3D_t MergeLines3D(line3D_t line3D_id1, line3D_t line3D_id2);
  inline void DeleteLineObservation(colmap::image_t image_id,
                                    line2D_t line2D_idx);
  inline void DeleteLine3D(line3D_t id);
  inline void ClearLines3D();
  inline FlatHashSet<line3D_t> Line3DIds() const;

  // Mutation methods for structures2d
  inline bool ExistsStructure2D(colmap::image_t id) const;
  inline void AddStructure2D(colmap::image_t id, const class Structure2d &s);
  inline void DeleteStructure2D(colmap::image_t id);
  inline void ClearStructures2D();

  // Mutation methods for groups3D (aligned with COLMAP Reconstruction
  // interface)
  inline bool ExistsGroup3D(group3D_t id) const;
  inline group3D_t AddGroup3D(const Group3d &group); // Auto-generates ID
  inline void AddGroup3D(group3D_t id, const Group3d &group);
  inline void AddGroupObservation(group3D_t group3D_id,
                                  const colmap::TrackElement &track_el);
  inline group3D_t MergeGroups3D(group3D_t group3D_id1, group3D_t group3D_id2);
  inline void DeleteGroupObservation(colmap::image_t image_id,
                                     group2D_t group2D_idx);
  inline void DeleteGroup3D(group3D_t id);
  inline void ClearGroups3D();
  inline FlatHashSet<group3D_t> Group3DIds() const;

  // Load features from given structure database cache
  void Load(const StructureDatabaseCache &structure_database_cache);

  // Initialize all frames
  void InitializeAllWireframes(double threshold = 2.0);

  // Construct 3D wireframe from 2D associations via voting
  // For each image's Wireframe2d edge (point2D, line2D, weight):
  //   if point2D->point3D and line2D->line3D exist:
  //     Accumulate vote for (point3D, line3D) edge
  // After all images: add edges meeting thresholds to Wireframe3d
  void ConstructWireframe3dFrom2d(
      const WireframeVotingOptions &options = WireframeVotingOptions());

private:
  // Reference to point reconstruction (used for intrinsics and poses)
  const colmap::Reconstruction &point_recon_;

  // 2D structural data
  NodeHashMap<colmap::image_t, class Structure2d> structures2d_;

  // 3D data
  NodeHashMap<line3D_t, Line3dWithActiveLabels> lines3D_;
  NodeHashMap<group3D_t, Group3dWithActiveLabels> groups3D_;

  // ID counters for auto-generation (like COLMAP's max_point3D_id_)
  line3D_t max_line3D_id_ = 0;
  group3D_t max_group3D_id_ = 0;

  // Wireframe relations between 3D primitives
  Wireframe3d wireframe_;
};

const colmap::Reconstruction &StructureReconstruction::PointRecon() const {
  return point_recon_;
}

size_t StructureReconstruction::NumLines3D() const { return lines3D_.size(); }

size_t StructureReconstruction::NumGroups3D() const { return groups3D_.size(); }

const class Structure2d &
StructureReconstruction::Structure2d(const colmap::image_t image_id) const {
  return structures2d_.at(image_id);
}
class Structure2d &
StructureReconstruction::Structure2d(const colmap::image_t image_id) {
  return structures2d_.at(image_id);
}
const NodeHashMap<colmap::image_t, class Structure2d> &
StructureReconstruction::Structures2d() const {
  return structures2d_;
}
NodeHashMap<colmap::image_t, class Structure2d> &
StructureReconstruction::Structures2d() {
  return structures2d_;
}

const Line3dWithActiveLabels &
StructureReconstruction::Line(const line3D_t line_id) const {
  return lines3D_.at(line_id);
}
Line3dWithActiveLabels &StructureReconstruction::Line(const line3D_t line_id) {
  return lines3D_.at(line_id);
}
const NodeHashMap<line3D_t, Line3dWithActiveLabels> &
StructureReconstruction::Lines3D() const {
  return lines3D_;
}
NodeHashMap<line3D_t, Line3dWithActiveLabels> &
StructureReconstruction::Lines3D() {
  return lines3D_;
}

const Group3dWithActiveLabels &
StructureReconstruction::Group(const group3D_t group_id) const {
  return groups3D_.at(group_id);
}
Group3dWithActiveLabels &
StructureReconstruction::Group(const group3D_t group_id) {
  return groups3D_.at(group_id);
}
const NodeHashMap<group3D_t, Group3dWithActiveLabels> &
StructureReconstruction::Groups3D() const {
  return groups3D_;
}
NodeHashMap<group3D_t, Group3dWithActiveLabels> &
StructureReconstruction::Groups3D() {
  return groups3D_;
}

const Wireframe3d &StructureReconstruction::Wireframe() const {
  return wireframe_;
}
Wireframe3d &StructureReconstruction::Wireframe() { return wireframe_; }

// lines3D mutation
bool StructureReconstruction::ExistsLine3D(line3D_t id) const {
  return lines3D_.count(id) > 0;
}
line3D_t StructureReconstruction::AddLine3D(const Line3d &line) {
  const line3D_t line3D_id = ++max_line3D_id_;
  AddLine3D(line3D_id, line);
  return line3D_id;
}
void StructureReconstruction::AddLine3D(line3D_t id, const Line3d &line) {
  max_line3D_id_ = std::max(max_line3D_id_, id);

  // Update 2D->3D assignments for all track elements
  for (const auto &elem : line.track.Elements()) {
    class Structure2d &s2d = this->Structure2d(elem.image_id);
    const line2D_t line2D_idx = static_cast<line2D_t>(elem.point2D_idx);
    Line2d &line2D = s2d.Line(line2D_idx);
    if (line2D.HasLine3D()) {
      THROW_CHECK_EQ(line2D.line3D_id, id);
    } else {
      line2D.line3D_id = id;
    }
  }

  THROW_CHECK(!ExistsLine3D(id)) << "Line3D " << id << " already exists";
  lines3D_.emplace(id, Line3dWithActiveLabels(line));
}
void StructureReconstruction::DeleteLine3D(line3D_t id) {
  THROW_CHECK(ExistsLine3D(id)) << "Line3D " << id << " does not exist";

  // Clear 2D->3D assignments for all track elements
  const auto &line3d = Line(id);
  for (const auto &elem : line3d.track.Elements()) {
    if (ExistsStructure2D(elem.image_id)) {
      class Structure2d &s2d = this->Structure2d(elem.image_id);
      const line2D_t line2D_idx = static_cast<line2D_t>(elem.point2D_idx);
      if (line2D_idx < static_cast<line2D_t>(s2d.NumLines())) {
        s2d.Line(line2D_idx).line3D_id = kInvalidLine3dId;
      }
    }
  }

  lines3D_.erase(id);
}
void StructureReconstruction::ClearLines3D() {
  lines3D_.clear();
  max_line3D_id_ = 0;
}
FlatHashSet<line3D_t> StructureReconstruction::Line3DIds() const {
  FlatHashSet<line3D_t> ids;
  ids.reserve(lines3D_.size());
  for (const auto &kv : lines3D_) {
    ids.insert(kv.first);
  }
  return ids;
}
void StructureReconstruction::AddLineObservation(
    line3D_t line3D_id, const colmap::TrackElement &track_el) {
  THROW_CHECK(ExistsLine3D(line3D_id));
  auto &line3d = Line(line3D_id);
  line3d.track.AddElement(track_el);
  // Update 2D->3D assignment
  class Structure2d &s2d = this->Structure2d(track_el.image_id);
  const line2D_t line2D_idx = static_cast<line2D_t>(track_el.point2D_idx);
  THROW_CHECK(!s2d.Line(line2D_idx).HasLine3D())
      << "Line2D already has a 3D line assigned";
  s2d.Line(line2D_idx).line3D_id = line3D_id;
}
line3D_t StructureReconstruction::MergeLines3D(line3D_t line3D_id1,
                                               line3D_t line3D_id2) {
  THROW_CHECK(ExistsLine3D(line3D_id1));
  THROW_CHECK(ExistsLine3D(line3D_id2));
  THROW_CHECK_NE(line3D_id1, line3D_id2);

  const auto &line1 = Line(line3D_id1);
  const auto &line2 = Line(line3D_id2);

  // Collect 2D observations from both tracks
  std::vector<colmap::Image> images;
  std::vector<Line2d> line2ds;
  for (const auto &elem : line1.track.Elements()) {
    images.push_back(point_recon_.Image(elem.image_id));
    line2ds.push_back(Structure2d(elem.image_id).Line(elem.point2D_idx));
  }
  for (const auto &elem : line2.track.Elements()) {
    images.push_back(point_recon_.Image(elem.image_id));
    line2ds.push_back(Structure2d(elem.image_id).Line(elem.point2D_idx));
  }

  // Compute infinite line via SVD on endpoints (like AggregateLine3dList)
  std::vector<Line3d> lines_to_merge = {line1, line2};
  const int n_lines = static_cast<int>(lines_to_merge.size());
  V3D center = V3D::Zero();
  for (int i = 0; i < n_lines; ++i) {
    center += lines_to_merge[i].start;
    center += lines_to_merge[i].end;
  }
  center /= (2.0 * static_cast<double>(n_lines));

  Eigen::MatrixXd endpoints(n_lines * 2, 3);
  for (int i = 0; i < n_lines; ++i) {
    endpoints.row(2 * i) = (lines_to_merge[i].start - center).transpose();
    endpoints.row(2 * i + 1) = (lines_to_merge[i].end - center).transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(endpoints, Eigen::ComputeThinV);
  V3D direc = svd.matrixV().col(0).normalized();
  InfiniteLine3d inf_line(center, direc);

  // Get final segment from 2D supports
  Line3d merged_geom =
      GetLineSegmentFromInfiniteLine3d(inf_line, images, line2ds, 0);

  // Build merged track (save before deleting originals)
  colmap::Track merged_track;
  merged_track.Reserve(line1.track.Length() + line2.track.Length());
  merged_track.AddElements(line1.track.Elements());
  merged_track.AddElements(line2.track.Elements());
  merged_geom.track = std::move(merged_track);

  // Take minimum uncertainty
  merged_geom.uncertainty = std::min(line1.uncertainty, line2.uncertainty);

  // Save inactive state from both lines before deletion
  Line3dWithActiveLabels merged(merged_geom);
  merged.MergeInactiveFrom(line1);
  merged.MergeInactiveFrom(line2);

  // Following COLMAP's MergePoints3D pattern:
  // Delete both originals (clears all 2D->3D assignments cleanly),
  // then add the merged line (reassigns all 2D->3D from saved track).
  DeleteLine3D(line3D_id1);
  DeleteLine3D(line3D_id2);
  AddLine3D(line3D_id1, merged);

  // Restore merged inactive state (AddLine3D clears it via Line3d assignment)
  Line(line3D_id1).MergeInactiveFrom(merged);

  return line3D_id1;
}
void StructureReconstruction::DeleteLineObservation(colmap::image_t image_id,
                                                    line2D_t line2D_idx) {
  class Structure2d &s2d = this->Structure2d(image_id);
  Line2d &line2D = s2d.Line(line2D_idx);
  if (!line2D.HasLine3D()) {
    return;
  }
  const line3D_t line3D_id = line2D.line3D_id;
  auto &line3d = Line(line3D_id);

  // Remove from track
  line3d.track.DeleteElement(image_id,
                             static_cast<colmap::point2D_t>(line2D_idx));
  // Clear 2D->3D assignment
  line2D.line3D_id = kInvalidLine3dId;

  // Sync inactive set with track
  line3d.CleanupInactiveSet();

  // Delete line if track becomes too short
  if (line3d.track.Length() < 2) {
    DeleteLine3D(line3D_id);
  }
}

// structures2d mutation
bool StructureReconstruction::ExistsStructure2D(colmap::image_t id) const {
  return structures2d_.count(id) > 0;
}
void StructureReconstruction::AddStructure2D(colmap::image_t id,
                                             const class Structure2d &s) {
  THROW_CHECK(!ExistsStructure2D(id))
      << "Structure2D " << id << " already exists";
  structures2d_[id] = s;
}
void StructureReconstruction::DeleteStructure2D(colmap::image_t id) {
  THROW_CHECK(ExistsStructure2D(id))
      << "Structure2D " << id << " does not exist";
  structures2d_.erase(id);
}
void StructureReconstruction::ClearStructures2D() { structures2d_.clear(); }

// groups3D mutation
bool StructureReconstruction::ExistsGroup3D(group3D_t id) const {
  return groups3D_.count(id) > 0;
}
group3D_t StructureReconstruction::AddGroup3D(const Group3d &group) {
  const group3D_t group3D_id = ++max_group3D_id_;
  AddGroup3D(group3D_id, group);
  return group3D_id;
}
void StructureReconstruction::AddGroup3D(group3D_t id, const Group3d &group) {
  max_group3D_id_ = std::max(max_group3D_id_, id);

  // Update 2D->3D assignments for all track elements
  for (const auto &elem : group.track.Elements()) {
    class Structure2d &s2d = this->Structure2d(elem.image_id);
    const group2D_t group2D_idx = static_cast<group2D_t>(elem.point2D_idx);
    Group2d &group2D = s2d.Group(group2D_idx);
    if (group2D.group3D_id != kInvalidGroup3dId) {
      THROW_CHECK_EQ(group2D.group3D_id, id);
    } else {
      group2D.group3D_id = id;
    }
  }

  THROW_CHECK(!ExistsGroup3D(id)) << "Group3D " << id << " already exists";
  groups3D_.emplace(id, Group3dWithActiveLabels(group));
}
void StructureReconstruction::AddGroupObservation(
    group3D_t group3D_id, const colmap::TrackElement &track_el) {
  THROW_CHECK(ExistsGroup3D(group3D_id));
  Group3d &group3d = Group(group3D_id);
  group3d.track.AddElement(track_el);
  // Update 2D->3D assignment
  class Structure2d &s2d = this->Structure2d(track_el.image_id);
  const group2D_t group2D_idx = static_cast<group2D_t>(track_el.point2D_idx);
  THROW_CHECK_EQ(s2d.Group(group2D_idx).group3D_id, kInvalidGroup3dId)
      << "Group2D already has a 3D group assigned";
  s2d.Group(group2D_idx).group3D_id = group3D_id;
}
group3D_t StructureReconstruction::MergeGroups3D(group3D_t group3D_id1,
                                                 group3D_t group3D_id2) {
  THROW_CHECK(ExistsGroup3D(group3D_id1));
  THROW_CHECK(ExistsGroup3D(group3D_id2));
  THROW_CHECK_NE(group3D_id1, group3D_id2);

  const Group3d &group1 = Group(group3D_id1);
  const Group3d &group2 = Group(group3D_id2);

  // Build merged group
  Group3d merged = group1;

  // Build merged track (save before deleting originals)
  colmap::Track merged_track;
  merged_track.Reserve(group1.track.Length() + group2.track.Length());
  merged_track.AddElements(group1.track.Elements());
  merged_track.AddElements(group2.track.Elements());
  merged.track = std::move(merged_track);

  // Merge point associations (union, keeping max weight)
  FlatHashMap<feature3D_t, double> point_weights;
  for (const auto &af : group1.points) {
    double &w = point_weights[af.idx];
    w = std::max(w, af.w);
  }
  for (const auto &af : group2.points) {
    double &w = point_weights[af.idx];
    w = std::max(w, af.w);
  }
  merged.points.clear();
  for (const auto &[idx, w] : point_weights) {
    merged.AddPoint(AssociatedFeature3d(idx, w));
  }

  // Merge line associations (union, keeping max weight)
  FlatHashMap<feature3D_t, double> line_weights;
  for (const auto &af : group1.lines) {
    line_weights[af.idx] = std::max(line_weights[af.idx], af.w);
  }
  for (const auto &af : group2.lines) {
    line_weights[af.idx] = std::max(line_weights[af.idx], af.w);
  }
  merged.lines.clear();
  for (const auto &[idx, w] : line_weights) {
    merged.AddLine(AssociatedFeature3d(idx, w));
  }

  // Note: Params are NOT merged here - caller must re-initialize params
  // after merge using the combined 3D features

  // Following COLMAP's MergePoints3D pattern:
  // Delete both originals (clears all 2D->3D assignments cleanly),
  // then add the merged group (reassigns all 2D->3D from saved track).
  DeleteGroup3D(group3D_id1);
  DeleteGroup3D(group3D_id2);
  AddGroup3D(group3D_id1, merged);

  return group3D_id1;
}
void StructureReconstruction::DeleteGroupObservation(colmap::image_t image_id,
                                                     group2D_t group2D_idx) {
  class Structure2d &s2d = this->Structure2d(image_id);
  Group2d &group2D = s2d.Group(group2D_idx);
  if (group2D.group3D_id == kInvalidGroup3dId) {
    return;
  }
  const group3D_t group3D_id = group2D.group3D_id;
  Group3d &group3d = Group(group3D_id);

  // Remove from track
  group3d.track.DeleteElement(image_id,
                              static_cast<colmap::point2D_t>(group2D_idx));
  // Clear 2D->3D assignment
  group2D.group3D_id = kInvalidGroup3dId;

  // Delete group if track becomes too short
  if (group3d.track.Length() < 2) {
    DeleteGroup3D(group3D_id);
  }
}
void StructureReconstruction::DeleteGroup3D(group3D_t id) {
  THROW_CHECK(ExistsGroup3D(id)) << "Group3D " << id << " does not exist";

  // Clear 2D->3D assignments for all track elements
  const Group3d &group3d = Group(id);
  for (const auto &elem : group3d.track.Elements()) {
    if (ExistsStructure2D(elem.image_id)) {
      class Structure2d &s2d = this->Structure2d(elem.image_id);
      const group2D_t group2D_idx = static_cast<group2D_t>(elem.point2D_idx);
      if (group2D_idx < static_cast<group2D_t>(s2d.NumGroups())) {
        s2d.Group(group2D_idx).group3D_id = kInvalidGroup3dId;
      }
    }
  }

  groups3D_.erase(id);
}
void StructureReconstruction::ClearGroups3D() {
  groups3D_.clear();
  max_group3D_id_ = 0;
}
FlatHashSet<group3D_t> StructureReconstruction::Group3DIds() const {
  FlatHashSet<group3D_t> ids;
  ids.reserve(groups3D_.size());
  for (const auto &kv : groups3D_) {
    ids.insert(kv.first);
  }
  return ids;
}

} // namespace limap
