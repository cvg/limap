#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <sqlite3.h>

#include "limap/scene/structure2d.h"

namespace limap {

// Adapted from colmap::Database
typedef Eigen::Matrix<line2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    LineMatchesBlob;
typedef Eigen::Matrix<line2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    GroupMatchesBlob;

class StructureDatabase {
public:
  // Factory function to create a database implementation for a given path.
  // The factory should be robust to handle non-supported files and return a
  // runtime_error in that case.
  using Factory = std::function<std::shared_ptr<StructureDatabase>(
      const std::filesystem::path &)>;

  // Register a factory to open a database implementation. Database factories
  // are tried in reverse order of registration. In other words, later
  // registrations are tried first.
  static void Register(Factory factory);

  // Closes the database, if not closed before.
  virtual ~StructureDatabase();

  // Open database and throw a runtime_error if none of the factories succeeds.
  static std::shared_ptr<StructureDatabase>
  Open(const std::filesystem::path &path);

  // Explicitly close the database before destruction.
  virtual void Close() = 0;

  // Combine multiple queries into one transaction by wrapping a code section
  // into a `BeginTransaction` and `EndTransaction`. You can create a scoped
  // transaction with `DatabaseTransaction` that ends when the transaction
  // object is destructed. Depending on the database implementation, combining
  // queries results in faster transaction time due to reduced locking of the
  // database, etc.
  virtual void BeginTransaction() const = 0;
  virtual void EndTransaction() const = 0;

  // Test exists
  virtual bool ExistsStructure2d(colmap::image_t image_id) const = 0;
  virtual bool ExistsLineMatches(colmap::image_t image_id1,
                                 colmap::image_t image_id2) const = 0;
  virtual bool ExistsGroupMatches(colmap::image_t image_id1,
                                  colmap::image_t image_id2) const = 0;

  // Read
  virtual Structure2d ReadStructure2d(colmap::image_t image_id) const = 0;
  virtual NodeHashMap<colmap::image_t, Structure2d>
  ReadAllStructures2d() const = 0;
  virtual LineMatchesBlob
  ReadLineMatchesBlob(colmap::image_t image_id1,
                      colmap::image_t image_id2) const = 0;
  virtual LineMatches ReadLineMatches(colmap::image_t image_id1,
                                      colmap::image_t image_id2) const = 0;
  virtual std::vector<std::pair<colmap::image_pair_t, LineMatchesBlob>>
  ReadAllLineMatchesBlob() const = 0;
  virtual std::vector<std::pair<colmap::image_pair_t, LineMatches>>
  ReadAllLineMatches() const = 0;
  virtual GroupMatchesBlob
  ReadGroupMatchesBlob(colmap::image_t image_id1,
                       colmap::image_t image_id2) const = 0;
  virtual GroupMatches ReadGroupMatches(colmap::image_t image_id1,
                                        colmap::image_t image_id2) const = 0;
  virtual std::vector<std::pair<colmap::image_pair_t, GroupMatchesBlob>>
  ReadAllGroupMatchesBlob() const = 0;
  virtual std::vector<std::pair<colmap::image_pair_t, GroupMatches>>
  ReadAllGroupMatches() const = 0;

  // Write
  virtual void WriteStructure2d(colmap::image_t image_id,
                                const Structure2d &structure) = 0;
  virtual void WriteLineMatches(colmap::image_t image_id1,
                                colmap::image_t image_id2,
                                const LineMatches &matches) const = 0;
  virtual void WriteLineMatches(colmap::image_t image_id1,
                                colmap::image_t image_id2,
                                const LineMatchesBlob &blob) const = 0;
  virtual void WriteGroupMatches(colmap::image_t image_id1,
                                 colmap::image_t image_id2,
                                 const GroupMatches &matches) const = 0;
  virtual void WriteGroupMatches(colmap::image_t image_id1,
                                 colmap::image_t image_id2,
                                 const GroupMatchesBlob &blob) const = 0;

  // Update
  virtual void UpdateStructure2d(colmap::image_t image_id,
                                 const Structure2d &structure) const = 0;

  // Delete line/group matches.
  virtual void DeleteLineMatches(colmap::image_t image_id1,
                                 colmap::image_t image_id2) const = 0;
  virtual void DeleteGroupMatches(colmap::image_t image_id1,
                                  colmap::image_t image_id2) const = 0;

  // Clear the entire line/group matches table.
  virtual void ClearLineMatches() const = 0;
  virtual void ClearGroupMatches() const = 0;

  // Clear all database tables
  virtual void ClearAllTables() const = 0;

  // Clear
  virtual void ClearStructures2d() const = 0;

private:
  friend class StructureDatabaseTransaction;

  // Used to ensure that only one transaction is active at the same time.
  std::mutex transaction_mutex_;

  static std::vector<Factory> factories_;
};

// This class automatically manages the scope of a database transaction by
// calling `BeginTransaction` and `EndTransaction` during construction and
// destruction, respectively.
class StructureDatabaseTransaction {
public:
  explicit StructureDatabaseTransaction(StructureDatabase *database);
  ~StructureDatabaseTransaction();

private:
  NON_COPYABLE(StructureDatabaseTransaction)
  NON_MOVABLE(StructureDatabaseTransaction)
  StructureDatabase *database_;
  std::unique_lock<std::mutex> database_lock_;
};

} // namespace limap
