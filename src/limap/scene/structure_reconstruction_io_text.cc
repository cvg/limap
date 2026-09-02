#include "limap/scene/structure_reconstruction_io_text.h"

#include <colmap/util/file.h>
#include <fstream>

namespace limap {

namespace {

void ReadTextLine2d(Line2d &l, std::istream &is) {
  double sx, sy, ex, ey, sc;
  is >> sx >> sy >> ex >> ey >> sc;
  l.start = V2D(sx, sy);
  l.end = V2D(ex, ey);
  l.score = sc;
}

void WriteTextLine2d(const Line2d &l, std::ostream &os) {
  os << l.start.x() << " " << l.start.y() << " " << l.end.x() << " "
     << l.end.y() << " " << l.score << "\n";
}

void ReadTextAssociatedFeature2d(AssociatedFeature2d &af, std::istream &is) {
  is >> af.idx >> af.w;
}

void WriteTextAssociatedFeature2d(const AssociatedFeature2d &af,
                                  std::ostream &os) {
  os << af.idx << " " << af.w << "\n";
}

void ReadTextGroup2d(Group2d &g, std::istream &is) {
  int type;
  size_t np, nl;
  is >> type >> np >> nl >> g.group3D_id;
  g.type = static_cast<GroupType>(type);
  g.points.resize(np);
  for (auto &p : g.points)
    ReadTextAssociatedFeature2d(p, is);
  g.lines.resize(nl);
  for (auto &l : g.lines)
    ReadTextAssociatedFeature2d(l, is);
}

void WriteTextGroup2d(const Group2d &g, std::ostream &os) {
  os << static_cast<int>(g.type) << " " << g.points.size() << " "
     << g.lines.size() << " " << g.group3D_id << "\n";
  for (const auto &p : g.points)
    WriteTextAssociatedFeature2d(p, os);
  for (const auto &l : g.lines)
    WriteTextAssociatedFeature2d(l, os);
}

void ReadTextWireframe2d(Wireframe2d &wf, std::istream &is) {
  wf.Clear();
  size_t N;
  is >> N;
  for (size_t i = 0; i < N; ++i) {
    point2D_t pid;
    line2D_t lid;
    double w;
    is >> pid >> lid >> w;
    wf.AddEdge(pid, lid, w);
  }
}

void WriteTextWireframe2d(const Wireframe2d &wf, std::ostream &os) {
  auto edges = wf.GetAllEdges();
  os << edges.size() << "\n";
  for (const auto &e : edges)
    os << e.point_idx << " " << e.line_idx << " " << e.w << "\n";
}

void ReadTextStructure2d(Structure2d &s2d, std::istream &is) {
  point2D_t P;
  size_t L, G, E;
  is >> P >> L >> G >> E;
  std::vector<Line2d> lines(L);
  for (auto &l : lines)
    ReadTextLine2d(l, is);
  s2d.SetLines(lines);
  std::vector<Group2d> groups(G);
  for (auto &g : groups)
    ReadTextGroup2d(g, is);
  s2d.SetGroups(groups);
  Wireframe2d wf;
  ReadTextWireframe2d(wf, is);
  s2d.SetWireframe(wf);
  s2d.SetNumPoints(P);
}

void WriteTextStructure2d(const Structure2d &s2d, std::ostream &os) {
  os << s2d.NumPoints() << " " << s2d.NumLines() << " " << s2d.NumGroups()
     << " " << s2d.Wireframe().CountEdges() << "\n";
  for (const auto &l : s2d.Lines())
    WriteTextLine2d(l, os);
  for (const auto &g : s2d.Groups())
    WriteTextGroup2d(g, os);
  WriteTextWireframe2d(s2d.Wireframe(), os);
}

void ReadTextTrack(colmap::Track &t, std::istream &is) {
  size_t N;
  is >> N;
  t = colmap::Track();
  t.Reserve(N);
  for (size_t i = 0; i < N; ++i) {
    colmap::image_t iid;
    point2D_t pid;
    is >> iid >> pid;
    t.AddElement(iid, pid);
  }
}

void WriteTextTrack(const colmap::Track &t, std::ostream &os) {
  const auto &elems = t.Elements();
  os << elems.size() << "\n";
  for (const auto &e : elems)
    os << e.image_id << " " << e.point2D_idx << "\n";
}

// Write only active track elements (skip inactive observations)
void WriteTextTrackActiveOnly(
    const colmap::Track &t, const NodeHashSet<colmap::image_t> &inactive_images,
    std::ostream &os) {
  if (inactive_images.empty()) {
    WriteTextTrack(t, os);
    return;
  }
  const auto &elems = t.Elements();
  size_t active_count = 0;
  for (const auto &e : elems) {
    if (inactive_images.count(e.image_id) == 0) {
      active_count++;
    }
  }
  os << active_count << "\n";
  for (const auto &e : elems) {
    if (inactive_images.count(e.image_id) == 0) {
      os << e.image_id << " " << e.point2D_idx << "\n";
    }
  }
}

void ReadTextLine3d(Line3d &L, std::istream &is) {
  double sx, sy, sz, ex, ey, ez;
  is >> sx >> sy >> sz >> ex >> ey >> ez;
  L.start = V3D(sx, sy, sz);
  L.end = V3D(ex, ey, ez);
  ReadTextTrack(L.track, is);
  is >> L.uncertainty;
}

void WriteTextLine3d(const Line3dWithActiveLabels &L, std::ostream &os) {
  os << L.start.x() << " " << L.start.y() << " " << L.start.z() << " "
     << L.end.x() << " " << L.end.y() << " " << L.end.z() << "\n";
  WriteTextTrackActiveOnly(L.track, L.InactiveImages(), os);
  os << L.uncertainty << "\n";
}

void ReadTextAssociatedFeature3d(AssociatedFeature3d &af, std::istream &is) {
  is >> af.idx >> af.w;
}

void WriteTextAssociatedFeature3d(const AssociatedFeature3d &af,
                                  std::ostream &os) {
  os << af.idx << " " << af.w << "\n";
}

void ReadTextGroup3d(Group3d &g, std::istream &is) {
  int type;
  size_t np, nl;
  is >> type >> np >> nl;
  g.type = static_cast<GroupType>(type);
  g.points.resize(np);
  for (auto &p : g.points)
    ReadTextAssociatedFeature3d(p, is);
  g.lines.resize(nl);
  for (auto &l : g.lines)
    ReadTextAssociatedFeature3d(l, is);
  ReadTextTrack(g.track, is);

  size_t num_params;
  is >> num_params;
  std::vector<double> params(num_params);
  for (auto &param : params)
    is >> param;
  g.SetParams(params);

  THROW_CHECK(g.CheckParams()) << "Group3d has invalid params after read";
}

void WriteTextGroup3d(const Group3dWithActiveLabels &g, std::ostream &os) {
  const auto &inactive_pts = g.InactivePointIds();
  const auto &inactive_lns = g.InactiveLineIds();

  size_t active_pts = 0;
  for (const auto &p : g.points) {
    if (inactive_pts.count(p.idx) == 0)
      active_pts++;
  }
  size_t active_lns = 0;
  for (const auto &l : g.lines) {
    if (inactive_lns.count(l.idx) == 0)
      active_lns++;
  }

  os << static_cast<int>(g.type) << " " << active_pts << " " << active_lns
     << "\n";
  for (const auto &p : g.points) {
    if (inactive_pts.count(p.idx) == 0)
      WriteTextAssociatedFeature3d(p, os);
  }
  for (const auto &l : g.lines) {
    if (inactive_lns.count(l.idx) == 0)
      WriteTextAssociatedFeature3d(l, os);
  }
  WriteTextTrack(g.track, os);

  const auto &params = g.GetParams();
  os << params.size();
  for (const auto &param : params)
    os << " " << param;
  os << "\n";
}

void ReadTextWireframe3d(Wireframe3d &wf, std::istream &is) {
  wf.Clear();
  size_t N;
  is >> N;
  for (size_t i = 0; i < N; ++i) {
    point3D_t pid;
    line3D_t lid;
    double w;
    is >> pid >> lid >> w;
    wf.AddEdge(pid, lid, w);
  }
}

void WriteTextWireframe3d(const Wireframe3d &wf, std::ostream &os) {
  auto edges = wf.GetAllEdges();
  os << edges.size() << "\n";
  for (const auto &e : edges)
    os << e.point_idx << " " << e.line_idx << " " << e.w << "\n";
}

} // anonymous namespace

// ===========================================================
// PUBLIC TEXT API
// ===========================================================

void ReadStructures2dText(StructureReconstruction &recon, std::istream &is) {
  recon.Structures2d().clear();
  size_t N;
  is >> N;
  for (size_t i = 0; i < N; ++i) {
    colmap::image_t image_id;
    is >> image_id;
    Structure2d s2d;
    ReadTextStructure2d(s2d, is);
    recon.Structures2d()[image_id] = std::move(s2d);
  }
}

void ReadStructures2dText(StructureReconstruction &recon,
                          const std::string &path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadStructures2dText(recon, file);
}

void WriteStructures2dText(const StructureReconstruction &recon,
                           std::ostream &os) {
  std::vector<std::pair<colmap::image_t, Structure2d>> vec(
      recon.Structures2d().begin(), recon.Structures2d().end());
  std::sort(vec.begin(), vec.end(),
            [](auto &a, auto &b) { return a.first < b.first; });

  os << vec.size() << "\n";

  for (const auto &kv : vec) {
    os << kv.first << "\n";
    WriteTextStructure2d(kv.second, os);
  }
}

void WriteStructures2dText(const StructureReconstruction &recon,
                           const std::string &path) {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteStructures2dText(recon, file);
}

void ReadLines3DText(StructureReconstruction &recon, std::istream &is) {
  recon.Lines3D().clear();
  size_t N;
  is >> N;
  for (size_t i = 0; i < N; ++i) {
    line3D_t id;
    is >> id;
    Line3d L;
    ReadTextLine3d(L, is);
    recon.Lines3D()[id] = std::move(L);
  }
}

void ReadLines3DText(StructureReconstruction &recon, const std::string &path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadLines3DText(recon, file);
}

void WriteLines3DText(const StructureReconstruction &recon, std::ostream &os) {
  // Sort by ID for deterministic output
  std::vector<line3D_t> ids;
  ids.reserve(recon.Lines3D().size());
  for (const auto &[id, _] : recon.Lines3D()) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());

  os << ids.size() << "\n";

  for (const line3D_t id : ids) {
    os << id << "\n";
    WriteTextLine3d(recon.Lines3D().at(id), os);
  }
}

void WriteLines3DText(const StructureReconstruction &recon,
                      const std::string &path) {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteLines3DText(recon, file);
}

void ReadGroups3DText(StructureReconstruction &recon, std::istream &is) {
  recon.Groups3D().clear();
  size_t N;
  is >> N;
  for (size_t i = 0; i < N; ++i) {
    group3D_t id;
    is >> id;
    Group3d g;
    ReadTextGroup3d(g, is);
    recon.Groups3D()[id] = std::move(g);
  }
}

void ReadGroups3DText(StructureReconstruction &recon, const std::string &path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadGroups3DText(recon, file);
}

void WriteGroups3DText(const StructureReconstruction &recon, std::ostream &os) {
  // Sort by ID for deterministic output (same pattern as WriteLines3DText)
  std::vector<group3D_t> ids;
  ids.reserve(recon.Groups3D().size());
  for (const auto &[id, _] : recon.Groups3D()) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());

  os << ids.size() << "\n";

  for (const group3D_t id : ids) {
    os << id << "\n";
    WriteTextGroup3d(recon.Groups3D().at(id), os);
  }
}

void WriteGroups3DText(const StructureReconstruction &recon,
                       const std::string &path) {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteGroups3DText(recon, file);
}

void ReadWireframeText(StructureReconstruction &recon, std::istream &is) {
  recon.Wireframe().Clear();
  ReadTextWireframe3d(recon.Wireframe(), is);
}

void ReadWireframeText(StructureReconstruction &recon,
                       const std::string &path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  ReadWireframeText(recon, file);
}

void WriteWireframeText(const StructureReconstruction &recon,
                        std::ostream &os) {
  WriteTextWireframe3d(recon.Wireframe(), os);
}

void WriteWireframeText(const StructureReconstruction &recon,
                        const std::string &path) {
  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  WriteWireframeText(recon, file);
}

} // namespace limap
