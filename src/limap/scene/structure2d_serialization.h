#pragma once
#include "limap/scene/structure2d.h"
#include <string>
#include <vector>

namespace limap {

// AssociatedFeature2d
std::vector<uint8_t>
SerializeAssociatedFeature2d(const AssociatedFeature2d &af);
AssociatedFeature2d
DeserializeAssociatedFeature2d(const std::vector<uint8_t> &data,
                               size_t &offset);

// Line2d
std::vector<uint8_t> SerializeLine2d(const Line2d &line);
Line2d DeserializeLine2d(const std::vector<uint8_t> &data, size_t &offset);

// Wireframe2d
std::vector<uint8_t> SerializeWireframe2d(const Wireframe2d &wf);
Wireframe2d DeserializeWireframe2d(const std::vector<uint8_t> &data,
                                   size_t &offset);

// Group2d
std::vector<uint8_t> SerializeGroup2d(const Group2d &group);
Group2d DeserializeGroup2d(const std::vector<uint8_t> &data, size_t &offset);

// Structure2d
std::vector<uint8_t> SerializeStructure2d(const Structure2d &structure);
Structure2d DeserializeStructure2d(const std::vector<uint8_t> &data);

} // namespace limap
