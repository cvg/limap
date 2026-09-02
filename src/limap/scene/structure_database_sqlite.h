#pragma once

#include "limap/scene/structure_database.h"

#include <filesystem>
#include <memory>

namespace limap {

// Can be used to construct temporary in-memory database.
constexpr inline char kInMemorySqliteStructureDatabasePath[] = ":memory:";

std::shared_ptr<StructureDatabase>
OpenSqliteStructureDatabase(const std::filesystem::path &path);

} // namespace limap
