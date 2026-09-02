#include "limap/scene/structure_database.h"
#include "limap/scene/structure_database_sqlite.h"

#include <colmap/util/logging.h>

namespace limap {

std::vector<StructureDatabase::Factory> StructureDatabase::factories_ = {
    &OpenSqliteStructureDatabase};

void StructureDatabase::Register(Factory factory) {
  factories_.push_back(std::move(factory));
}

StructureDatabase::~StructureDatabase() = default;

std::shared_ptr<StructureDatabase>
StructureDatabase::Open(const std::filesystem::path &path) {
  for (auto it = factories_.rbegin(); it != factories_.rend(); ++it) {
    try {
      return (*it)(path);
    } catch (const std::exception &e) {
      LOG(WARNING)
          << "Failed to open database with registered factory, because: "
          << e.what() << ". Trying next registered factory.";
    }
  }
  throw std::runtime_error("No registered database factory succeeded.");
}

StructureDatabaseTransaction::StructureDatabaseTransaction(
    StructureDatabase *database)
    : database_(database), database_lock_(database->transaction_mutex_) {
  THROW_CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

StructureDatabaseTransaction::~StructureDatabaseTransaction() {
  database_->EndTransaction();
}

} // namespace limap
