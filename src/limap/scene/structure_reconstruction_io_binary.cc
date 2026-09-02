#include "limap/scene/structure_reconstruction_io_binary.h"

#include <colmap/util/endian.h>
#include <colmap/util/file.h>
#include <fstream>

namespace limap {

namespace {

// ----------------------- Line2d -----------------------

void WriteBinaryLine2d(const Line2d &line, std::ostream &os) {
  colmap::WriteBinaryLittleEndian<double>(&os, line.start.x());
  colmap::WriteBinaryLittleEndian<double>(&os, line.start.y());
  colmap::WriteBinaryLittleEndian<double>(&os, line.end.x());
  colmap::WriteBinaryLittleEndian<double>(&os, line.end.y());
  colmap::WriteBinaryLittleEndian<double>(&os, line.score);
  colmap::WriteBinaryLittleEndian<line3D_t>(&os, line.line3D_id);
}

void ReadBinaryLine2d(Line2d &line, std::istream &is) {
  double sx = colmap::ReadBinaryLittleEndian<double>(&is);
  double sy = colmap::ReadBinaryLittleEndian<double>(&is);
  double ex = colmap::ReadBinaryLittleEndian<double>(&is);
  double ey = colmap::ReadBinaryLittleEndian<double>(&is);
  double sc = colmap::ReadBinaryLittleEndian<double>(&is);
  line3D_t lid = colmap::ReadBinaryLittleEndian<line3D_t>(&is);

  line.start = V2D(sx, sy);
  line.end = V2D(ex, ey);
  line.score = sc;
  line.line3D_id = lid;
}

// ----------------- AssociatedFeature2d -----------------

void WriteBinaryAssociatedFeature2d(const AssociatedFeature2d &af,
                                    std::ostream &os) {
  colmap::WriteBinaryLittleEndian<feature2D_t>(&os, af.idx);
  colmap::WriteBinaryLittleEndian<double>(&os, af.w);
}

void ReadBinaryAssociatedFeature2d(AssociatedFeature2d &af, std::istream &is) {
  af.idx = colmap::ReadBinaryLittleEndian<feature2D_t>(&is);
  af.w = colmap::ReadBinaryLittleEndian<double>(&is);
}

// ------------------------- Group2d ----------------------

void WriteBinaryGroup2d(const Group2d &g, std::ostream &os) {
  colmap::WriteBinaryLittleEndian<int>(&os, static_cast<int>(g.type));

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, g.points.size());
  for (const auto &p : g.points)
    WriteBinaryAssociatedFeature2d(p, os);

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, g.lines.size());
  for (const auto &l : g.lines)
    WriteBinaryAssociatedFeature2d(l, os);

  colmap::WriteBinaryLittleEndian<group3D_t>(&os, g.group3D_id);
}

void ReadBinaryGroup2d(Group2d &g, std::istream &is) {
  g.type = static_cast<GroupType>(colmap::ReadBinaryLittleEndian<int>(&is));

  uint64_t P = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  g.points.resize(P);
  for (auto &p : g.points)
    ReadBinaryAssociatedFeature2d(p, is);

  uint64_t L = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  g.lines.resize(L);
  for (auto &l : g.lines)
    ReadBinaryAssociatedFeature2d(l, is);

  g.group3D_id = colmap::ReadBinaryLittleEndian<group3D_t>(&is);
}

// ----------------------- Wireframe2d --------------------

void WriteBinaryWireframe2d(const Wireframe2d &wf, std::ostream &os) {
  auto edges = wf.GetAllEdges();
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, edges.size());

  for (const auto &e : edges) {
    colmap::WriteBinaryLittleEndian<point2D_t>(&os, e.point_idx);
    colmap::WriteBinaryLittleEndian<line2D_t>(&os, e.line_idx);
    colmap::WriteBinaryLittleEndian<double>(&os, e.w);
  }
}

void ReadBinaryWireframe2d(Wireframe2d &wf, std::istream &is) {
  wf.Clear();

  uint64_t N = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  for (uint64_t i = 0; i < N; ++i) {
    point2D_t pid = colmap::ReadBinaryLittleEndian<point2D_t>(&is);
    line2D_t lid = colmap::ReadBinaryLittleEndian<line2D_t>(&is);
    double w = colmap::ReadBinaryLittleEndian<double>(&is);
    wf.AddEdge(pid, lid, w);
  }
}

// ---------------------- Structure2d ---------------------

void WriteBinaryStructure2d(const Structure2d &s2d, std::ostream &os) {
  colmap::WriteBinaryLittleEndian<point2D_t>(&os, s2d.NumPoints());
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, s2d.NumLines());
  for (const auto &l : s2d.Lines())
    WriteBinaryLine2d(l, os);

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, s2d.NumGroups());
  for (const auto &g : s2d.Groups())
    WriteBinaryGroup2d(g, os);

  WriteBinaryWireframe2d(s2d.Wireframe(), os);
}

void ReadBinaryStructure2d(Structure2d &s2d, std::istream &is) {
  point2D_t num_pts = colmap::ReadBinaryLittleEndian<point2D_t>(&is);
  s2d.SetNumPoints(num_pts);

  uint64_t Nlines = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  std::vector<Line2d> lines(Nlines);
  for (auto &l : lines)
    ReadBinaryLine2d(l, is);
  s2d.SetLines(lines);

  uint64_t Ngroups = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  std::vector<Group2d> groups(Ngroups);
  for (auto &g : groups)
    ReadBinaryGroup2d(g, is);
  s2d.SetGroups(groups);

  Wireframe2d wf;
  ReadBinaryWireframe2d(wf, is);
  s2d.SetWireframe(wf);
}

// -------------------------- Track ------------------------

void WriteBinaryTrack(const colmap::Track &track, std::ostream &os) {
  const auto &elems = track.Elements();
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, elems.size());

  for (const auto &e : elems) {
    colmap::WriteBinaryLittleEndian<colmap::image_t>(&os, e.image_id);
    colmap::WriteBinaryLittleEndian<point2D_t>(&os, e.point2D_idx);
  }
}

// Write only active track elements (skip inactive observations)
void WriteBinaryTrackActiveOnly(
    const colmap::Track &track,
    const NodeHashSet<colmap::image_t> &inactive_images, std::ostream &os) {
  if (inactive_images.empty()) {
    WriteBinaryTrack(track, os);
    return;
  }
  const auto &elems = track.Elements();
  uint64_t active_count = 0;
  for (const auto &e : elems) {
    if (inactive_images.count(e.image_id) == 0) {
      active_count++;
    }
  }
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, active_count);
  for (const auto &e : elems) {
    if (inactive_images.count(e.image_id) == 0) {
      colmap::WriteBinaryLittleEndian<colmap::image_t>(&os, e.image_id);
      colmap::WriteBinaryLittleEndian<point2D_t>(&os, e.point2D_idx);
    }
  }
}

void ReadBinaryTrack(colmap::Track &track, std::istream &is) {
  uint64_t n = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  track = colmap::Track();
  track.Reserve(n);

  for (uint64_t i = 0; i < n; ++i) {
    colmap::image_t iid = colmap::ReadBinaryLittleEndian<colmap::image_t>(&is);
    point2D_t pid = colmap::ReadBinaryLittleEndian<point2D_t>(&is);
    track.AddElement(iid, pid);
  }
}

// -------------------------- Line3d ------------------------

void WriteBinaryLine3d(const Line3dWithActiveLabels &L, std::ostream &os) {
  colmap::WriteBinaryLittleEndian<double>(&os, L.start.x());
  colmap::WriteBinaryLittleEndian<double>(&os, L.start.y());
  colmap::WriteBinaryLittleEndian<double>(&os, L.start.z());

  colmap::WriteBinaryLittleEndian<double>(&os, L.end.x());
  colmap::WriteBinaryLittleEndian<double>(&os, L.end.y());
  colmap::WriteBinaryLittleEndian<double>(&os, L.end.z());

  WriteBinaryTrackActiveOnly(L.track, L.InactiveImages(), os);

  colmap::WriteBinaryLittleEndian<double>(&os, L.uncertainty);
}

void ReadBinaryLine3d(Line3d &L, std::istream &is) {
  double sx = colmap::ReadBinaryLittleEndian<double>(&is);
  double sy = colmap::ReadBinaryLittleEndian<double>(&is);
  double sz = colmap::ReadBinaryLittleEndian<double>(&is);
  double ex = colmap::ReadBinaryLittleEndian<double>(&is);
  double ey = colmap::ReadBinaryLittleEndian<double>(&is);
  double ez = colmap::ReadBinaryLittleEndian<double>(&is);

  L.start = V3D(sx, sy, sz);
  L.end = V3D(ex, ey, ez);

  ReadBinaryTrack(L.track, is);

  L.uncertainty = colmap::ReadBinaryLittleEndian<double>(&is);
}

// ---------------- AssociatedFeature3d --------------------

void WriteBinaryAssociatedFeature3d(const AssociatedFeature3d &af,
                                    std::ostream &os) {
  colmap::WriteBinaryLittleEndian<feature3D_t>(&os, af.idx);
  colmap::WriteBinaryLittleEndian<double>(&os, af.w);
}

void ReadBinaryAssociatedFeature3d(AssociatedFeature3d &af, std::istream &is) {
  af.idx = colmap::ReadBinaryLittleEndian<feature3D_t>(&is);
  af.w = colmap::ReadBinaryLittleEndian<double>(&is);
}

// -------------------------- Group3d ------------------------

void WriteBinaryGroup3d(const Group3dWithActiveLabels &g, std::ostream &os) {
  colmap::WriteBinaryLittleEndian<int>(&os, static_cast<int>(g.type));

  const auto &inactive_pts = g.InactivePointIds();
  const auto &inactive_lns = g.InactiveLineIds();

  // Write only active point associations
  uint64_t active_pts = 0;
  for (const auto &p : g.points) {
    if (inactive_pts.count(p.idx) == 0)
      active_pts++;
  }
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, active_pts);
  for (const auto &p : g.points) {
    if (inactive_pts.count(p.idx) == 0)
      WriteBinaryAssociatedFeature3d(p, os);
  }

  // Write only active line associations
  uint64_t active_lns = 0;
  for (const auto &l : g.lines) {
    if (inactive_lns.count(l.idx) == 0)
      active_lns++;
  }
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, active_lns);
  for (const auto &l : g.lines) {
    if (inactive_lns.count(l.idx) == 0)
      WriteBinaryAssociatedFeature3d(l, os);
  }

  WriteBinaryTrack(g.track, os);

  const auto &params = g.GetParams();
  colmap::WriteBinaryLittleEndian<uint64_t>(&os, params.size());
  for (const auto &param : params)
    colmap::WriteBinaryLittleEndian<double>(&os, param);
}

void ReadBinaryGroup3d(Group3d &g, std::istream &is) {
  g.type = static_cast<GroupType>(colmap::ReadBinaryLittleEndian<int>(&is));

  uint64_t P = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  g.points.resize(P);
  for (auto &p : g.points)
    ReadBinaryAssociatedFeature3d(p, is);

  uint64_t L = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  g.lines.resize(L);
  for (auto &l : g.lines)
    ReadBinaryAssociatedFeature3d(l, is);

  ReadBinaryTrack(g.track, is);

  uint64_t num_params = colmap::ReadBinaryLittleEndian<uint64_t>(&is);
  std::vector<double> params(num_params);
  for (auto &param : params)
    param = colmap::ReadBinaryLittleEndian<double>(&is);
  g.SetParams(params);

  THROW_CHECK(g.CheckParams()) << "Group3d has invalid params after read";
}

} // namespace

// =============================================================
// Structures2D
// =============================================================

void ReadStructures2dBinary(StructureReconstruction &recon, std::istream &is) {
  recon.Structures2d().clear();

  uint64_t N = colmap::ReadBinaryLittleEndian<uint64_t>(&is);

  for (uint64_t i = 0; i < N; ++i) {
    colmap::image_t image_id =
        colmap::ReadBinaryLittleEndian<colmap::image_t>(&is);

    Structure2d s2d;
    ReadBinaryStructure2d(s2d, is);

    recon.Structures2d()[image_id] = std::move(s2d);
  }
}

void ReadStructures2dBinary(StructureReconstruction &recon,
                            const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadStructures2dBinary(recon, file);
}

void WriteStructures2dBinary(const StructureReconstruction &recon,
                             std::ostream &os) {
  std::vector<std::pair<colmap::image_t, Structure2d>> vec(
      recon.Structures2d().begin(), recon.Structures2d().end());
  std::sort(vec.begin(), vec.end(),
            [](auto &a, auto &b) { return a.first < b.first; });

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, vec.size());

  for (const auto &kv : vec) {
    colmap::WriteBinaryLittleEndian<colmap::image_t>(&os, kv.first);
    WriteBinaryStructure2d(kv.second, os);
  }
}

void WriteStructures2dBinary(const StructureReconstruction &recon,
                             const std::string &path) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteStructures2dBinary(recon, file);
}

// =============================================================
// Lines3D
// =============================================================

void ReadLines3DBinary(StructureReconstruction &recon, std::istream &is) {
  recon.Lines3D().clear();

  uint64_t N = colmap::ReadBinaryLittleEndian<uint64_t>(&is);

  for (uint64_t i = 0; i < N; ++i) {
    line3D_t id = colmap::ReadBinaryLittleEndian<line3D_t>(&is);

    Line3d L;
    ReadBinaryLine3d(L, is);

    recon.Lines3D()[id] = std::move(L);
  }
}

void ReadLines3DBinary(StructureReconstruction &recon,
                       const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadLines3DBinary(recon, file);
}

void WriteLines3DBinary(const StructureReconstruction &recon,
                        std::ostream &os) {
  // Sort by ID for deterministic output
  std::vector<line3D_t> ids;
  ids.reserve(recon.Lines3D().size());
  for (const auto &[id, _] : recon.Lines3D()) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, ids.size());

  for (const line3D_t id : ids) {
    colmap::WriteBinaryLittleEndian<line3D_t>(&os, id);
    WriteBinaryLine3d(recon.Lines3D().at(id), os);
  }
}

void WriteLines3DBinary(const StructureReconstruction &recon,
                        const std::string &path) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteLines3DBinary(recon, file);
}

// =============================================================
// Groups3D
// =============================================================

void ReadGroups3DBinary(StructureReconstruction &recon, std::istream &is) {
  recon.Groups3D().clear();

  uint64_t N = colmap::ReadBinaryLittleEndian<uint64_t>(&is);

  for (uint64_t i = 0; i < N; ++i) {
    group3D_t id = colmap::ReadBinaryLittleEndian<group3D_t>(&is);

    Group3d g;
    ReadBinaryGroup3d(g, is);

    recon.Groups3D()[id] = std::move(g);
  }
}

void ReadGroups3DBinary(StructureReconstruction &recon,
                        const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadGroups3DBinary(recon, file);
}

void WriteGroups3DBinary(const StructureReconstruction &recon,
                         std::ostream &os) {
  // Sort by ID for deterministic output (same pattern as WriteLines3DBinary)
  std::vector<group3D_t> ids;
  ids.reserve(recon.Groups3D().size());
  for (const auto &[id, _] : recon.Groups3D()) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, ids.size());

  for (const group3D_t id : ids) {
    colmap::WriteBinaryLittleEndian<group3D_t>(&os, id);
    WriteBinaryGroup3d(recon.Groups3D().at(id), os);
  }
}

void WriteGroups3DBinary(const StructureReconstruction &recon,
                         const std::string &path) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteGroups3DBinary(recon, file);
}

// =============================================================
// Wireframe3D
// =============================================================

void ReadWireframeBinary(StructureReconstruction &recon, std::istream &is) {
  recon.Wireframe().Clear();

  uint64_t N = colmap::ReadBinaryLittleEndian<uint64_t>(&is);

  for (uint64_t i = 0; i < N; ++i) {
    point3D_t pid = colmap::ReadBinaryLittleEndian<point3D_t>(&is);
    line3D_t lid = colmap::ReadBinaryLittleEndian<line3D_t>(&is);
    double w = colmap::ReadBinaryLittleEndian<double>(&is);

    recon.Wireframe().AddEdge(pid, lid, w);
  }
}

void ReadWireframeBinary(StructureReconstruction &recon,
                         const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadWireframeBinary(recon, file);
}

void WriteWireframeBinary(const StructureReconstruction &recon,
                          std::ostream &os) {
  auto edges = recon.Wireframe().GetAllEdges();

  colmap::WriteBinaryLittleEndian<uint64_t>(&os, edges.size());

  for (const auto &e : edges) {
    colmap::WriteBinaryLittleEndian<point3D_t>(&os, e.point_idx);
    colmap::WriteBinaryLittleEndian<line3D_t>(&os, e.line_idx);
    colmap::WriteBinaryLittleEndian<double>(&os, e.w);
  }
}

void WriteWireframeBinary(const StructureReconstruction &recon,
                          const std::string &path) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteWireframeBinary(recon, file);
}

} // namespace limap
