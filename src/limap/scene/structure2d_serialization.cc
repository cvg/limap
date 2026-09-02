#include "limap/scene/structure2d_serialization.h"
#include <cstring>
#include <stdexcept>

namespace limap {

namespace {

template <typename T>
void AppendBytes(std::vector<uint8_t> &buffer, const T &value) {
  size_t old_size = buffer.size();
  buffer.resize(old_size + sizeof(T));
  std::memcpy(buffer.data() + old_size, &value, sizeof(T));
}

template <typename T>
T ReadBytes(const std::vector<uint8_t> &buffer, size_t &offset) {
  if (offset + sizeof(T) > buffer.size()) {
    throw std::runtime_error("Deserialize: buffer overflow");
  }
  T value;
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

void AppendVectorBytes(std::vector<uint8_t> &buffer,
                       const std::vector<uint8_t> &data) {
  uint64_t size = data.size();
  AppendBytes(buffer, size);
  buffer.insert(buffer.end(), data.begin(), data.end());
}

std::vector<uint8_t> ReadVectorBytes(const std::vector<uint8_t> &buffer,
                                     size_t &offset) {
  uint64_t size = ReadBytes<uint64_t>(buffer, offset);
  if (offset + size > buffer.size())
    throw std::runtime_error("Deserialize: vector overflow");
  std::vector<uint8_t> data(buffer.begin() + offset,
                            buffer.begin() + offset + size);
  offset += size;
  return data;
}

} // namespace

std::vector<uint8_t>
SerializeAssociatedFeature2d(const AssociatedFeature2d &af) {
  std::vector<uint8_t> buffer;
  AppendBytes(buffer, af.idx);
  AppendBytes(buffer, af.w);
  return buffer;
}

AssociatedFeature2d
DeserializeAssociatedFeature2d(const std::vector<uint8_t> &data,
                               size_t &offset) {
  AssociatedFeature2d af;
  af.idx = ReadBytes<feature2D_t>(data, offset);
  af.w = ReadBytes<double>(data, offset);
  return af;
}

// ----------------- Line2d -----------------
std::vector<uint8_t> SerializeLine2d(const Line2d &line) {
  std::vector<uint8_t> buffer;
  AppendBytes(buffer, line.start[0]);
  AppendBytes(buffer, line.start[1]);
  AppendBytes(buffer, line.end[0]);
  AppendBytes(buffer, line.end[1]);
  AppendBytes(buffer, line.line3D_id);
  return buffer;
}

Line2d DeserializeLine2d(const std::vector<uint8_t> &data, size_t &offset) {
  Line2d line;
  line.start[0] = ReadBytes<double>(data, offset);
  line.start[1] = ReadBytes<double>(data, offset);
  line.end[0] = ReadBytes<double>(data, offset);
  line.end[1] = ReadBytes<double>(data, offset);
  line.line3D_id = ReadBytes<line3D_t>(data, offset);
  return line;
}

// ----------------- Wireframe2d -----------------
std::vector<uint8_t> SerializeWireframe2d(const Wireframe2d &wf) {
  std::vector<uint8_t> buffer;
  uint64_t n_edges = wf.CountEdges();
  AppendBytes(buffer, n_edges);
  for (const auto &e : wf.GetAllEdges()) {
    AppendBytes(buffer, e.point_idx);
    AppendBytes(buffer, e.line_idx);
    AppendBytes(buffer, e.w);
  }
  return buffer;
}

Wireframe2d DeserializeWireframe2d(const std::vector<uint8_t> &data,
                                   size_t &offset) {
  Wireframe2d wf;
  uint64_t n_edges = ReadBytes<uint64_t>(data, offset);
  for (uint64_t i = 0; i < n_edges; i++) {
    WireframeConnection2d c;
    c.point_idx = ReadBytes<point2D_t>(data, offset);
    c.line_idx = ReadBytes<line2D_t>(data, offset);
    c.w = ReadBytes<double>(data, offset);
    wf.AddEdge(c);
  }
  return wf;
}

// ----------------- Group2d -----------------
std::vector<uint8_t> SerializeGroup2d(const Group2d &group) {
  std::vector<uint8_t> buffer;
  AppendBytes(buffer, static_cast<int>(group.type));

  // Params
  const auto &params = group.GetParams();
  uint64_t n_params = params.size();
  AppendBytes(buffer, n_params);
  for (const auto &p : params) {
    AppendBytes(buffer, p);
  }

  // Points
  uint64_t n_points = group.points.size();
  AppendBytes(buffer, n_points);
  for (const auto &pt : group.points) {
    AppendVectorBytes(buffer, SerializeAssociatedFeature2d(pt));
  }

  // Lines
  uint64_t n_lines = group.lines.size();
  AppendBytes(buffer, n_lines);
  for (const auto &ln : group.lines) {
    AppendVectorBytes(buffer, SerializeAssociatedFeature2d(ln));
  }

  return buffer;
}

Group2d DeserializeGroup2d(const std::vector<uint8_t> &data, size_t &offset) {
  GroupType type = static_cast<GroupType>(ReadBytes<int>(data, offset));
  Group2d group(type); // Use constructor that initializes params

  // Params
  uint64_t n_params = ReadBytes<uint64_t>(data, offset);
  if (n_params > 0) {
    std::vector<double> params(n_params);
    for (uint64_t i = 0; i < n_params; i++) {
      params[i] = ReadBytes<double>(data, offset);
    }
    group.SetParams(params);
  }

  uint64_t n_points = ReadBytes<uint64_t>(data, offset);
  group.points.resize(n_points);
  for (uint64_t i = 0; i < n_points; i++) {
    auto pt_bytes = ReadVectorBytes(data, offset);
    size_t pt_offset = 0;
    group.points[i] = DeserializeAssociatedFeature2d(pt_bytes, pt_offset);
  }

  uint64_t n_lines = ReadBytes<uint64_t>(data, offset);
  group.lines.resize(n_lines);
  for (uint64_t i = 0; i < n_lines; i++) {
    auto ln_bytes = ReadVectorBytes(data, offset);
    size_t ln_offset = 0;
    group.lines[i] = DeserializeAssociatedFeature2d(ln_bytes, ln_offset);
  }

  return group;
}

// ----------------- Structure2d -----------------
std::vector<uint8_t> SerializeStructure2d(const Structure2d &structure) {
  std::vector<uint8_t> buffer;

  // Num points
  uint64_t n_points = structure.NumPoints();
  AppendBytes(buffer, n_points);

  // Lines
  uint64_t n_lines = structure.NumLines();
  AppendBytes(buffer, n_lines);
  for (const auto &line : structure.Lines()) {
    AppendVectorBytes(buffer, SerializeLine2d(line));
  }

  // Groups
  uint64_t n_groups = structure.NumGroups();
  AppendBytes(buffer, n_groups);
  for (const auto &group : structure.Groups()) {
    AppendVectorBytes(buffer, SerializeGroup2d(group));
  }

  // Wireframe
  AppendVectorBytes(buffer, SerializeWireframe2d(structure.Wireframe()));
  return buffer;
}

Structure2d DeserializeStructure2d(const std::vector<uint8_t> &data) {
  Structure2d structure;
  size_t offset = 0;

  uint64_t n_points = ReadBytes<uint64_t>(data, offset);
  structure.SetNumPoints(static_cast<point2D_t>(n_points));

  uint64_t n_lines = ReadBytes<uint64_t>(data, offset);
  structure.Lines().resize(n_lines);
  for (uint64_t i = 0; i < n_lines; i++) {
    auto line_bytes = ReadVectorBytes(data, offset);
    size_t line_offset = 0;
    structure.Line(i) = DeserializeLine2d(line_bytes, line_offset);
  }

  uint64_t n_groups = ReadBytes<uint64_t>(data, offset);
  structure.Groups().resize(n_groups);
  for (uint64_t i = 0; i < n_groups; i++) {
    auto group_bytes = ReadVectorBytes(data, offset);
    size_t group_offset = 0;
    structure.Group(i) = DeserializeGroup2d(group_bytes, group_offset);
  }

  auto wf_bytes = ReadVectorBytes(data, offset);
  size_t wf_offset = 0;
  structure.Wireframe() = DeserializeWireframe2d(wf_bytes, wf_offset);

  return structure;
}

} // namespace limap
