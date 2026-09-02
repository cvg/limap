#include "limap/scene/structure2d_serialization.h"
#include "limap/scene/structure_database.h"

#include <colmap/util/string.h>

#include <iostream>

namespace limap {
namespace {

inline int SQLite3CallHelper(int result_code, const std::string &filename,
                             int line) {
  switch (result_code) {
  case SQLITE_OK:
  case SQLITE_ROW:
  case SQLITE_DONE:
    return result_code;
  default:
    colmap::LogMessageFatalThrow<std::runtime_error>(filename.c_str(), line)
            .stream()
        << "SQLite error: " << sqlite3_errstr(result_code);
    return result_code;
  }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                                  \
  {                                                                            \
    char *err_msg = nullptr;                                                   \
    const int result_code = sqlite3_exec(THROW_CHECK_NOTNULL(database), sql,   \
                                         callback, nullptr, &err_msg);         \
    if (result_code != SQLITE_OK) {                                            \
      LOG(ERROR) << "SQLite error [" << __FILE__ << ", line " << __LINE__      \
                 << "]: " << err_msg;                                          \
      sqlite3_free(err_msg);                                                   \
    }                                                                          \
  }

struct Sqlite3StmtContext {
  explicit Sqlite3StmtContext(sqlite3_stmt *sql_stmt) : sql_stmt_(sql_stmt) {}
  ~Sqlite3StmtContext() { SQLITE3_CALL(sqlite3_reset(sql_stmt_)); }

private:
  sqlite3_stmt *sql_stmt_;
};

void SwapLineMatchesBlob(LineMatchesBlob *matches) {
  for (Eigen::Index i = 0; i < matches->rows(); ++i) {
    std::swap((*matches)(i, 0), (*matches)(i, 1));
  }
}

LineMatchesBlob LineMatchesToBlob(const LineMatches &matches) {
  const LineMatchesBlob::Index kNumCols = 2;
  LineMatchesBlob blob(matches.size(), kNumCols);
  for (size_t i = 0; i < matches.size(); ++i) {
    blob(i, 0) = matches[i].line2D_idx1;
    blob(i, 1) = matches[i].line2D_idx2;
  }
  return blob;
}

LineMatches LineMatchesFromBlob(const LineMatchesBlob &blob) {
  THROW_CHECK_EQ(blob.cols(), 2);
  LineMatches matches(static_cast<size_t>(blob.rows()));
  for (LineMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].line2D_idx1 = blob(i, 0);
    matches[i].line2D_idx2 = blob(i, 1);
  }
  return matches;
}

void SwapGroupMatchesBlob(GroupMatchesBlob *matches) {
  for (Eigen::Index i = 0; i < matches->rows(); ++i) {
    std::swap((*matches)(i, 0), (*matches)(i, 1));
  }
}

GroupMatchesBlob GroupMatchesToBlob(const GroupMatches &matches) {
  const GroupMatchesBlob::Index kNumCols = 2;
  GroupMatchesBlob blob(matches.size(), kNumCols);
  for (size_t i = 0; i < matches.size(); ++i) {
    blob(i, 0) = matches[i].group2D_idx1;
    blob(i, 1) = matches[i].group2D_idx2;
  }
  return blob;
}

GroupMatches GroupMatchesFromBlob(const GroupMatchesBlob &blob) {
  THROW_CHECK_EQ(blob.cols(), 2);
  GroupMatches matches(static_cast<size_t>(blob.rows()));
  for (GroupMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].group2D_idx1 = blob(i, 0);
    matches[i].group2D_idx2 = blob(i, 1);
  }
  return matches;
}

template <typename MatrixType>
MatrixType ReadStaticMatrixBlob(sqlite3_stmt *sql_stmt, const int rc,
                                const int col) {
  THROW_CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col));
    if (num_bytes > 0) {
      THROW_CHECK_EQ(num_bytes,
                     matrix.size() * sizeof(typename MatrixType::Scalar));
      memcpy(reinterpret_cast<char *>(matrix.data()),
             sqlite3_column_blob(sql_stmt, col), num_bytes);
    } else {
      matrix = MatrixType::Zero();
    }
  } else {
    matrix = MatrixType::Zero();
  }

  return matrix;
}

template <typename MatrixType>
MatrixType ReadDynamicMatrixBlob(sqlite3_stmt *sql_stmt, const int rc,
                                 const int col) {
  THROW_CHECK_GE(col, 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

    THROW_CHECK_GE(rows, 0);
    THROW_CHECK_GE(cols, 0);
    matrix = MatrixType(rows, cols);

    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
    THROW_CHECK_EQ(matrix.size() * sizeof(typename MatrixType::Scalar),
                   num_bytes);

    memcpy(reinterpret_cast<char *>(matrix.data()),
           sqlite3_column_blob(sql_stmt, col + 2), num_bytes);
  } else {
    const typename MatrixType::Index rows =
        (MatrixType::RowsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::RowsAtCompileTime;
    const typename MatrixType::Index cols =
        (MatrixType::ColsAtCompileTime == Eigen::Dynamic)
            ? 0
            : MatrixType::ColsAtCompileTime;
    matrix = MatrixType(rows, cols);
  }

  return matrix;
}

template <typename MatrixType>
void WriteStaticMatrixBlob(sqlite3_stmt *sql_stmt, const MatrixType &matrix,
                           const int col) {
  SQLITE3_CALL(sqlite3_bind_blob(
      sql_stmt, col, reinterpret_cast<const char *>(matrix.data()),
      static_cast<int>(matrix.size() * sizeof(typename MatrixType::Scalar)),
      SQLITE_STATIC));
}

template <typename MatrixType>
void WriteDynamicMatrixBlob(sqlite3_stmt *sql_stmt, const MatrixType &matrix,
                            const int col) {
  THROW_CHECK_GE(matrix.rows(), 0);
  THROW_CHECK_GE(matrix.cols(), 0);
  THROW_CHECK_GE(col, 0);

  const size_t num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, matrix.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, matrix.cols()));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, col + 2,
                                 reinterpret_cast<const char *>(matrix.data()),
                                 static_cast<int>(num_bytes), SQLITE_STATIC));
}

class SqliteStructureDatabase : public StructureDatabase {
public:
  SqliteStructureDatabase() : database_(nullptr) {}

  static std::shared_ptr<StructureDatabase>
  Open(const std::filesystem::path &path) {
    auto database = std::make_shared<SqliteStructureDatabase>();

    // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
    // mutex (so that we don't serialize the connection's operations).
    // Modifications to the database will still be serialized, but multiple
    // connections can read concurrently.
    try {
      SQLITE3_CALL(sqlite3_open_v2(
          colmap::PlatformToUTF8(path.string()).c_str(), &database->database_,
          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
          nullptr));
    } catch (...) {
      SQLITE3_CALL(sqlite3_close_v2(database->database_));
      throw;
    }

    // Don't wait for the operating system to write the changes to disk
    SQLITE3_EXEC(database->database_, "PRAGMA synchronous=OFF", nullptr);

    // Use faster journaling mode
    SQLITE3_EXEC(database->database_, "PRAGMA journal_mode=WAL", nullptr);

    // Store temporary tables and indices in memory
    SQLITE3_EXEC(database->database_, "PRAGMA temp_store=MEMORY", nullptr);

    // Disabled by default
    SQLITE3_EXEC(database->database_, "PRAGMA foreign_keys=ON", nullptr);

    // Enable auto vacuum to reduce DB file size
    SQLITE3_EXEC(database->database_, "PRAGMA auto_vacuum=1", nullptr);

    database->CreateTables();
    database->UpdateSchema();
    database->PrepareSQLStatements();

    return database;
  }

  void CloseImpl() {
    if (database_ != nullptr) {
      FinalizeSQLStatements();
      SQLITE3_CALL(sqlite3_close_v2(database_));
      database_ = nullptr;
    }
  }

  ~SqliteStructureDatabase() override { CloseImpl(); }

  void Close() override { CloseImpl(); }

  void BeginTransaction() const override {
    SQLITE3_EXEC(THROW_CHECK_NOTNULL(database_), "BEGIN TRANSACTION", nullptr);
  }

  void EndTransaction() const override {
    SQLITE3_EXEC(THROW_CHECK_NOTNULL(database_), "END TRANSACTION", nullptr);
  }

  bool ExistsRowId(sqlite3_stmt *sql_stmt, const sqlite3_int64 row_id) const {
    Sqlite3StmtContext context(sql_stmt);
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

    return SQLITE3_CALL(sqlite3_step(sql_stmt)) == SQLITE_ROW;
  }

  bool ExistsStructure2d(colmap::image_t image_id) const override {
    Sqlite3StmtContext stmt_guard(sql_stmt_exists_structure2d_);
    sqlite3_bind_int64(sql_stmt_exists_structure2d_, 1, image_id);
    int rc = sqlite3_step(sql_stmt_exists_structure2d_);
    return rc == SQLITE_ROW;
  }

  bool ExistsLineMatches(const colmap::image_t image_id1,
                         const colmap::image_t image_id2) const override {
    return ExistsRowId(sql_stmt_exists_line_matches_,
                       colmap::ImagePairToPairId(image_id1, image_id2));
  }

  bool ExistsGroupMatches(const colmap::image_t image_id1,
                          const colmap::image_t image_id2) const override {
    return ExistsRowId(sql_stmt_exists_group_matches_,
                       colmap::ImagePairToPairId(image_id1, image_id2));
  }

  Structure2d ReadStructure2d(colmap::image_t image_id) const override {
    Sqlite3StmtContext stmt_guard(sql_stmt_read_structure2d_);
    sqlite3_bind_int64(sql_stmt_read_structure2d_, 1, image_id);

    int rc = sqlite3_step(sql_stmt_read_structure2d_);
    if (rc != SQLITE_ROW)
      throw std::runtime_error("Structure2d not found.");

    const void *data = sqlite3_column_blob(sql_stmt_read_structure2d_, 0);
    int size = sqlite3_column_bytes(sql_stmt_read_structure2d_, 0);
    return DeserializeStructure2d(
        std::vector<uint8_t>((uint8_t *)data, (uint8_t *)data + size));
  }

  NodeHashMap<colmap::image_t, Structure2d>
  ReadAllStructures2d() const override {
    Sqlite3StmtContext context(sql_stmt_read_all_structures2d_);

    NodeHashMap<colmap::image_t, Structure2d> structures2d;

    while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_all_structures2d_)) ==
           SQLITE_ROW) {
      colmap::image_t image_id = static_cast<colmap::image_t>(
          sqlite3_column_int64(sql_stmt_read_all_structures2d_, 0));
      const void *data =
          sqlite3_column_blob(sql_stmt_read_all_structures2d_, 1);
      int size = sqlite3_column_bytes(sql_stmt_read_all_structures2d_, 1);
      Structure2d structure2d = DeserializeStructure2d(
          std::vector<uint8_t>((uint8_t *)data, (uint8_t *)data + size));
      structures2d.emplace(image_id, structure2d);
    }

    return structures2d;
  }

  LineMatchesBlob
  ReadLineMatchesBlob(colmap::image_t image_id1,
                      colmap::image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_read_line_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_line_matches_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_line_matches_));
    LineMatchesBlob blob = ReadDynamicMatrixBlob<LineMatchesBlob>(
        sql_stmt_read_line_matches_, rc, 0);

    if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
      SwapLineMatchesBlob(&blob);
    }
    return blob;
  }

  LineMatches ReadLineMatches(colmap::image_t image_id1,
                              colmap::image_t image_id2) const override {
    return LineMatchesFromBlob(ReadLineMatchesBlob(image_id1, image_id2));
  }

  std::vector<std::pair<colmap::image_pair_t, LineMatchesBlob>>
  ReadAllLineMatchesBlob() const override {
    Sqlite3StmtContext context(sql_stmt_read_line_matches_all_);

    std::vector<std::pair<colmap::image_pair_t, LineMatchesBlob>>
        all_line_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_line_matches_all_))) ==
           SQLITE_ROW) {
      const colmap::image_pair_t pair_id = static_cast<colmap::image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_line_matches_all_, 0));
      all_line_matches.emplace_back(
          pair_id, ReadDynamicMatrixBlob<LineMatchesBlob>(
                       sql_stmt_read_line_matches_all_, rc, 1));
    }

    return all_line_matches;
  }

  std::vector<std::pair<colmap::image_pair_t, LineMatches>>
  ReadAllLineMatches() const override {
    Sqlite3StmtContext context(sql_stmt_read_line_matches_all_);

    std::vector<std::pair<colmap::image_pair_t, LineMatches>> all_line_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_line_matches_all_))) ==
           SQLITE_ROW) {
      const colmap::image_pair_t pair_id = static_cast<colmap::image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_line_matches_all_, 0));
      const LineMatchesBlob blob = ReadDynamicMatrixBlob<LineMatchesBlob>(
          sql_stmt_read_line_matches_all_, rc, 1);
      all_line_matches.emplace_back(pair_id, LineMatchesFromBlob(blob));
    }

    return all_line_matches;
  }

  GroupMatchesBlob
  ReadGroupMatchesBlob(colmap::image_t image_id1,
                       colmap::image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_read_group_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_group_matches_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_group_matches_));
    GroupMatchesBlob blob = ReadDynamicMatrixBlob<GroupMatchesBlob>(
        sql_stmt_read_group_matches_, rc, 0);

    if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
      SwapGroupMatchesBlob(&blob);
    }
    return blob;
  }

  GroupMatches ReadGroupMatches(colmap::image_t image_id1,
                                colmap::image_t image_id2) const override {
    return GroupMatchesFromBlob(ReadGroupMatchesBlob(image_id1, image_id2));
  }

  std::vector<std::pair<colmap::image_pair_t, GroupMatchesBlob>>
  ReadAllGroupMatchesBlob() const override {
    Sqlite3StmtContext context(sql_stmt_read_group_matches_all_);

    std::vector<std::pair<colmap::image_pair_t, GroupMatchesBlob>>
        all_group_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(
                sql_stmt_read_group_matches_all_))) == SQLITE_ROW) {
      const colmap::image_pair_t pair_id = static_cast<colmap::image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_group_matches_all_, 0));
      all_group_matches.emplace_back(
          pair_id, ReadDynamicMatrixBlob<GroupMatchesBlob>(
                       sql_stmt_read_group_matches_all_, rc, 1));
    }

    return all_group_matches;
  }

  std::vector<std::pair<colmap::image_pair_t, GroupMatches>>
  ReadAllGroupMatches() const override {
    Sqlite3StmtContext context(sql_stmt_read_group_matches_all_);

    std::vector<std::pair<colmap::image_pair_t, GroupMatches>>
        all_group_matches;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(
                sql_stmt_read_group_matches_all_))) == SQLITE_ROW) {
      const colmap::image_pair_t pair_id = static_cast<colmap::image_pair_t>(
          sqlite3_column_int64(sql_stmt_read_group_matches_all_, 0));
      const GroupMatchesBlob blob = ReadDynamicMatrixBlob<GroupMatchesBlob>(
          sql_stmt_read_group_matches_all_, rc, 1);
      all_group_matches.emplace_back(pair_id, GroupMatchesFromBlob(blob));
    }

    return all_group_matches;
  }

  void WriteStructure2d(colmap::image_t image_id,
                        const Structure2d &structure) override {
    std::vector<uint8_t> blob = SerializeStructure2d(structure);

    Sqlite3StmtContext stmt_guard(sql_stmt_write_structure2d_);
    sqlite3_bind_int64(sql_stmt_write_structure2d_, 1, image_id);
    sqlite3_bind_blob(sql_stmt_write_structure2d_, 2, blob.data(),
                      static_cast<int>(blob.size()), SQLITE_TRANSIENT);

    if (sqlite3_step(sql_stmt_write_structure2d_) != SQLITE_DONE)
      throw std::runtime_error("Failed to write Structure2d.");
  }

  void WriteLineMatches(const colmap::image_t image_id1,
                        const colmap::image_t image_id2,
                        const LineMatches &matches) const override {
    WriteLineMatches(image_id1, image_id2, LineMatchesToBlob(matches));
  }

  void WriteLineMatches(const colmap::image_t image_id1,
                        const colmap::image_t image_id2,
                        const LineMatchesBlob &blob) const override {
    Sqlite3StmtContext context(sql_stmt_write_line_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_line_matches_, 1, pair_id));

    // Important: the swapped data must live until the query is executed.
    LineMatchesBlob swapped_blob;
    if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
      swapped_blob = blob;
      SwapLineMatchesBlob(&swapped_blob);
      WriteDynamicMatrixBlob(sql_stmt_write_line_matches_, swapped_blob, 2);
    } else {
      WriteDynamicMatrixBlob(sql_stmt_write_line_matches_, blob, 2);
    }

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_line_matches_));
  }

  void WriteGroupMatches(const colmap::image_t image_id1,
                         const colmap::image_t image_id2,
                         const GroupMatches &matches) const override {
    WriteGroupMatches(image_id1, image_id2, GroupMatchesToBlob(matches));
  }

  void WriteGroupMatches(const colmap::image_t image_id1,
                         const colmap::image_t image_id2,
                         const GroupMatchesBlob &blob) const override {
    Sqlite3StmtContext context(sql_stmt_write_group_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_group_matches_, 1, pair_id));

    // Important: the swapped data must live until the query is executed.
    GroupMatchesBlob swapped_blob;
    if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
      swapped_blob = blob;
      SwapGroupMatchesBlob(&swapped_blob);
      WriteDynamicMatrixBlob(sql_stmt_write_group_matches_, swapped_blob, 2);
    } else {
      WriteDynamicMatrixBlob(sql_stmt_write_group_matches_, blob, 2);
    }

    SQLITE3_CALL(sqlite3_step(sql_stmt_write_group_matches_));
  }

  void UpdateStructure2d(colmap::image_t image_id,
                         const Structure2d &structure) const override {
    std::vector<uint8_t> blob = SerializeStructure2d(structure);

    Sqlite3StmtContext stmt_guard(sql_stmt_update_structure2d_);
    sqlite3_bind_blob(sql_stmt_update_structure2d_, 1, blob.data(),
                      static_cast<int>(blob.size()), SQLITE_TRANSIENT);
    sqlite3_bind_int64(sql_stmt_update_structure2d_, 2, image_id);

    if (sqlite3_step(sql_stmt_update_structure2d_) != SQLITE_DONE)
      throw std::runtime_error("Failed to update Structure2d.");
  }

  void DeleteLineMatches(const colmap::image_t image_id1,
                         const colmap::image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_delete_line_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_line_matches_, 1,
                                    static_cast<sqlite3_int64>(pair_id)));
    SQLITE3_CALL(sqlite3_step(sql_stmt_delete_line_matches_));
  }

  void DeleteGroupMatches(const colmap::image_t image_id1,
                          const colmap::image_t image_id2) const override {
    Sqlite3StmtContext context(sql_stmt_delete_group_matches_);

    const colmap::image_pair_t pair_id =
        colmap::ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_delete_group_matches_, 1,
                                    static_cast<sqlite3_int64>(pair_id)));
    SQLITE3_CALL(sqlite3_step(sql_stmt_delete_group_matches_));
  }

  void ClearStructures2d() const override {
    Sqlite3StmtContext stmt_guard(sql_stmt_clear_structure2d_);
    sqlite3_step(sql_stmt_clear_structure2d_);
  }

  void ClearLineMatches() const override {
    Sqlite3StmtContext context(sql_stmt_clear_line_matches_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_line_matches_));
  }

  void ClearGroupMatches() const override {
    Sqlite3StmtContext context(sql_stmt_clear_group_matches_);
    SQLITE3_CALL(sqlite3_step(sql_stmt_clear_group_matches_));
  }

  void ClearAllTables() const override {
    ClearStructures2d();
    ClearLineMatches();
    ClearGroupMatches();
  }

  // -------------------- Schema & Tables --------------------

  void UpdateSchema() const {
    std::lock_guard<std::mutex> lock(update_schema_mutex_);

    SQLITE3_EXEC(
        database_,
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER)",
        nullptr);

    int current_version = 0;
    sqlite3_stmt *stmt = nullptr;
    SQLITE3_CALL(sqlite3_prepare_v2(
        database_, "SELECT value FROM meta WHERE key='schema_version'", -1,
        &stmt, nullptr));

    if (sqlite3_step(stmt) == SQLITE_ROW) {
      current_version = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
  }

  void CreateTables() const {
    CreateLineMatchesTable();
    CreateGroupMatchesTable();
    CreateStructure2dTable();
  }

  void CreateLineMatchesTable() const {
    const std::string sql = "CREATE TABLE IF NOT EXISTS line_matches"
                            "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
                            "    rows     INTEGER               NOT NULL,"
                            "    cols     INTEGER               NOT NULL,"
                            "    data     BLOB);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateGroupMatchesTable() const {
    const std::string sql = "CREATE TABLE IF NOT EXISTS group_matches"
                            "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
                            "    rows     INTEGER               NOT NULL,"
                            "    cols     INTEGER               NOT NULL,"
                            "    data     BLOB);";

    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
  }

  void CreateStructure2dTable() const {
    SQLITE3_EXEC(database_,
                 "CREATE TABLE IF NOT EXISTS structure2d ("
                 "image_id INTEGER PRIMARY KEY, "
                 "data BLOB NOT NULL)",
                 nullptr);
  }

  void PrepareSQLStatements() {
    sql_stmts_.clear();

    auto prepare_sql_stmt = [this](const std::string_view sql,
                                   sqlite3_stmt **sql_stmt) {
      THROW_CHECK_NOTNULL(database_);
      VLOG(3) << "Preparing SQL statement: " << sql;
      SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.data(), -1, sql_stmt, 0));
      sql_stmts_.push_back(sql_stmt);
    };

    // exists_*
    prepare_sql_stmt("SELECT 1 FROM structure2d WHERE image_id=?",
                     &sql_stmt_exists_structure2d_);
    prepare_sql_stmt("SELECT 1 FROM line_matches WHERE pair_id = ?;",
                     &sql_stmt_exists_line_matches_);
    prepare_sql_stmt("SELECT 1 FROM group_matches WHERE pair_id = ?;",
                     &sql_stmt_exists_group_matches_);

    // add_*
    prepare_sql_stmt("INSERT INTO structure2d (image_id, data) VALUES (?, ?)",
                     &sql_stmt_add_structure2d_);

    // update_*
    prepare_sql_stmt("UPDATE structure2d SET data=? WHERE image_id=?",
                     &sql_stmt_update_structure2d_);

    // read_*
    prepare_sql_stmt("SELECT data FROM structure2d WHERE image_id=?",
                     &sql_stmt_read_structure2d_);
    prepare_sql_stmt("SELECT * FROM structure2d",
                     &sql_stmt_read_all_structures2d_);
    prepare_sql_stmt(
        "SELECT rows, cols, data FROM line_matches WHERE pair_id = ?;",
        &sql_stmt_read_line_matches_);
    prepare_sql_stmt("SELECT * FROM line_matches WHERE rows > 0;",
                     &sql_stmt_read_line_matches_all_);
    prepare_sql_stmt(
        "SELECT rows, cols, data FROM group_matches WHERE pair_id = ?;",
        &sql_stmt_read_group_matches_);
    prepare_sql_stmt("SELECT * FROM group_matches WHERE rows > 0;",
                     &sql_stmt_read_group_matches_all_);

    // write_*
    prepare_sql_stmt("INSERT INTO structure2d (image_id, data) VALUES (?, ?);",
                     &sql_stmt_write_structure2d_);
    prepare_sql_stmt(
        "INSERT INTO line_matches(pair_id, rows, cols, data) VALUES(?, ?, "
        "?, ?);",
        &sql_stmt_write_line_matches_);
    prepare_sql_stmt(
        "INSERT INTO group_matches(pair_id, rows, cols, data) VALUES(?, ?, "
        "?, ?);",
        &sql_stmt_write_group_matches_);

    // delete_*
    prepare_sql_stmt("DELETE FROM structure2d WHERE image_id=?",
                     &sql_stmt_delete_structure2d_);
    prepare_sql_stmt("DELETE FROM line_matches WHERE pair_id = ?;",
                     &sql_stmt_delete_line_matches_);
    prepare_sql_stmt("DELETE FROM group_matches WHERE pair_id = ?;",
                     &sql_stmt_delete_group_matches_);

    // clear_*
    prepare_sql_stmt("DELETE FROM structure2d", &sql_stmt_clear_structure2d_);
    prepare_sql_stmt("DELETE FROM line_matches;",
                     &sql_stmt_clear_line_matches_);
    prepare_sql_stmt("DELETE FROM group_matches;",
                     &sql_stmt_clear_group_matches_);
  }

  void FinalizeSQLStatements() {
    for (sqlite3_stmt **sql_stmt : sql_stmts_) {
      SQLITE3_CALL(sqlite3_finalize(*sql_stmt));
      *sql_stmt = nullptr;
    }
  }

  sqlite3 *database_ = nullptr;

  // Ensure that only one database object at a time updates the schema of a
  // database. Since the schema is updated every time a database is opened, this
  // is to ensure that there are no race conditions ("database locked" error
  // messages) when the user actually only intends to read from the database,
  // which requires to open it.
  inline static std::mutex update_schema_mutex_;

  // A collection of all `sqlite3_stmt` objects for deletion in the destructor.
  std::vector<sqlite3_stmt **> sql_stmts_;

  // exists_*
  sqlite3_stmt *sql_stmt_exists_structure2d_ = nullptr;
  sqlite3_stmt *sql_stmt_exists_line_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_exists_group_matches_ = nullptr;

  // add_*
  sqlite3_stmt *sql_stmt_add_structure2d_ = nullptr;

  // update_*
  sqlite3_stmt *sql_stmt_update_structure2d_ = nullptr;

  // read_*
  sqlite3_stmt *sql_stmt_read_structure2d_ = nullptr;
  sqlite3_stmt *sql_stmt_read_all_structures2d_ = nullptr;
  sqlite3_stmt *sql_stmt_read_line_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_read_line_matches_all_ = nullptr;
  sqlite3_stmt *sql_stmt_read_group_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_read_group_matches_all_ = nullptr;

  // write_*
  sqlite3_stmt *sql_stmt_write_structure2d_ = nullptr;
  sqlite3_stmt *sql_stmt_write_line_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_write_group_matches_ = nullptr;

  // delete_*
  sqlite3_stmt *sql_stmt_delete_structure2d_ = nullptr;
  sqlite3_stmt *sql_stmt_delete_line_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_delete_group_matches_ = nullptr;

  // clear_*
  sqlite3_stmt *sql_stmt_clear_structure2d_ = nullptr;
  sqlite3_stmt *sql_stmt_clear_line_matches_ = nullptr;
  sqlite3_stmt *sql_stmt_clear_group_matches_ = nullptr;
};

} // namespace

std::shared_ptr<StructureDatabase>
OpenSqliteStructureDatabase(const std::filesystem::path &path) {
  return SqliteStructureDatabase::Open(path);
}

} // namespace limap
