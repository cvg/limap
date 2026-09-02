#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/structure_database.h"
#include "limap/util/types.h"
#include <thirdparty/pycolmap/pybind11_extension.h>

using namespace pybind11::literals;
namespace py = pybind11;

// [Reference]
// https://github.com/colmap/colmap/blob/main/src/pycolmap/scene/database.cc

namespace limap {
namespace {

class StructureDatabaseTransactionWrapper {
public:
  explicit StructureDatabaseTransactionWrapper(StructureDatabase *db)
      : db_(db) {}

  void Enter() { txn_ = std::make_unique<StructureDatabaseTransaction>(db_); }

  void Exit(const py::args &) { txn_.reset(); }

private:
  StructureDatabase *db_;
  std::unique_ptr<StructureDatabaseTransaction> txn_;
};

class PyStructureDatabaseImpl : public StructureDatabase,
                                py::trampoline_self_life_support {
public:
  void Close() override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, Close);
  }

  bool ExistsStructure2d(colmap::image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, StructureDatabase, ExistsStructure2d,
                           image_id);
  }
  bool ExistsLineMatches(colmap::image_t i1,
                         colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(bool, StructureDatabase, ExistsLineMatches, i1, i2);
  }
  bool ExistsGroupMatches(colmap::image_t i1,
                          colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(bool, StructureDatabase, ExistsGroupMatches, i1, i2);
  }

  // Read
  Structure2d ReadStructure2d(colmap::image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(Structure2d, StructureDatabase, ReadStructure2d,
                           image_id);
  }
  NodeHashMap<colmap::image_t, Structure2d>
  ReadAllStructures2d() const override {
    using ImageStructureMap = NodeHashMap<colmap::image_t, Structure2d>;
    PYBIND11_OVERRIDE_PURE(ImageStructureMap, StructureDatabase,
                           ReadAllStructures2d);
  }

  LineMatchesBlob ReadLineMatchesBlob(colmap::image_t i1,
                                      colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(LineMatchesBlob, StructureDatabase,
                           ReadLineMatchesBlob, i1, i2);
  }
  LineMatches ReadLineMatches(colmap::image_t i1,
                              colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(LineMatches, StructureDatabase, ReadLineMatches, i1,
                           i2);
  }
  std::vector<std::pair<colmap::image_pair_t, LineMatchesBlob>>
  ReadAllLineMatchesBlob() const override {
    using ReturnType =
        std::vector<std::pair<colmap::image_pair_t, LineMatchesBlob>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, StructureDatabase,
                           ReadAllLineMatchesBlob);
  }
  std::vector<std::pair<colmap::image_pair_t, LineMatches>>
  ReadAllLineMatches() const override {
    using ReturnType =
        std::vector<std::pair<colmap::image_pair_t, LineMatches>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, StructureDatabase, ReadAllLineMatches);
  }
  GroupMatchesBlob ReadGroupMatchesBlob(colmap::image_t i1,
                                        colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(GroupMatchesBlob, StructureDatabase,
                           ReadGroupMatchesBlob, i1, i2);
  }
  GroupMatches ReadGroupMatches(colmap::image_t i1,
                                colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(GroupMatches, StructureDatabase, ReadGroupMatches,
                           i1, i2);
  }
  std::vector<std::pair<colmap::image_pair_t, GroupMatchesBlob>>
  ReadAllGroupMatchesBlob() const override {
    using ReturnType =
        std::vector<std::pair<colmap::image_pair_t, GroupMatchesBlob>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, StructureDatabase,
                           ReadAllGroupMatchesBlob);
  }
  std::vector<std::pair<colmap::image_pair_t, GroupMatches>>
  ReadAllGroupMatches() const override {
    using ReturnType =
        std::vector<std::pair<colmap::image_pair_t, GroupMatches>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, StructureDatabase, ReadAllGroupMatches);
  }

  // Write
  void WriteStructure2d(colmap::image_t image_id,
                        const Structure2d &s) override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, WriteStructure2d, image_id,
                           s);
  }
  void WriteLineMatches(colmap::image_t i1, colmap::image_t i2,
                        const LineMatches &m) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, WriteLineMatches, i1, i2,
                           m);
  }
  void WriteLineMatches(colmap::image_t i1, colmap::image_t i2,
                        const LineMatchesBlob &b) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, WriteLineMatches, i1, i2,
                           b);
  }
  void WriteGroupMatches(colmap::image_t i1, colmap::image_t i2,
                         const GroupMatches &m) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, WriteGroupMatches, i1, i2,
                           m);
  }
  void WriteGroupMatches(colmap::image_t i1, colmap::image_t i2,
                         const GroupMatchesBlob &b) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, WriteGroupMatches, i1, i2,
                           b);
  }

  // Update
  void UpdateStructure2d(colmap::image_t image_id,
                         const Structure2d &s) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, UpdateStructure2d, image_id,
                           s);
  }

  // Delete
  void DeleteLineMatches(colmap::image_t i1,
                         colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, DeleteLineMatches, i1, i2);
  }
  void DeleteGroupMatches(colmap::image_t i1,
                          colmap::image_t i2) const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, DeleteGroupMatches, i1, i2);
  }

  // Clear
  void ClearLineMatches() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, ClearLineMatches);
  }
  void ClearGroupMatches() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, ClearGroupMatches);
  }
  void ClearAllTables() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, ClearAllTables);
  }
  void ClearStructures2d() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, ClearStructures2d);
  }

  void BeginTransaction() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, BeginTransaction);
  }

  void EndTransaction() const override {
    PYBIND11_OVERRIDE_PURE(void, StructureDatabase, EndTransaction);
  }
};

} // namespace

void BindStructureDatabase(py::module &m) {
  py::classh<LineMatch>(m, "LineMatch")
      .def(py::init<>())
      .def(py::init<line2D_t, line2D_t>(), py::arg("line2D_idx1"),
           py::arg("line2D_idx2"))
      .def_readwrite("line2D_idx1", &LineMatch::line2D_idx1)
      .def_readwrite("line2D_idx2", &LineMatch::line2D_idx2)
      .def("__repr__", [](const LineMatch &lm) {
        return "LineMatch(line2D_idx1=" + std::to_string(lm.line2D_idx1) +
               ", line2D_idx2=" + std::to_string(lm.line2D_idx2) + ")";
      });

  py::classh<GroupMatch>(m, "GroupMatch")
      .def(py::init<>())
      .def(py::init<std::size_t, std::size_t>(), py::arg("group2D_idx1"),
           py::arg("group2D_idx2"))
      .def_readwrite("group2D_idx1", &GroupMatch::group2D_idx1)
      .def_readwrite("group2D_idx2", &GroupMatch::group2D_idx2)
      .def("__repr__", [](const GroupMatch &gm) {
        return "GroupMatch(group2D_idx1=" + std::to_string(gm.group2D_idx1) +
               ", group2D_idx2=" + std::to_string(gm.group2D_idx2) + ")";
      });

  py::bind_vector<LineMatches>(m, "LineMatches", py::module_local())
      .def("__repr__", [](const LineMatches &v) {
        return "LineMatches[" + std::to_string(v.size()) + "]";
      });

  py::bind_vector<GroupMatches>(m, "GroupMatches", py::module_local())
      .def("__repr__", [](const GroupMatches &v) {
        return "GroupMatches[" + std::to_string(v.size()) + "]";
      });

  py::classh<StructureDatabase, PyStructureDatabaseImpl>(m, "StructureDatabase")
      .def(py::init<>())
      .def_static("open", &StructureDatabase::Open, "path"_a)
      .def("close", &StructureDatabase::Close)
      .def("__enter__", [](StructureDatabase &self) { return &self; })
      .def("__exit__",
           [](StructureDatabase &self, const py::args &) { self.Close(); })
      .def("exists_structure2d", &StructureDatabase::ExistsStructure2d)
      .def("exists_line_matches", &StructureDatabase::ExistsLineMatches)
      .def("exists_group_matches", &StructureDatabase::ExistsGroupMatches)
      .def("read_structure2d", &StructureDatabase::ReadStructure2d)
      .def("read_all_structures2d", &StructureDatabase::ReadAllStructures2d)
      .def("read_line_matches_blob", &StructureDatabase::ReadLineMatchesBlob)
      .def("read_line_matches", &StructureDatabase::ReadLineMatches)
      .def("read_all_line_matches_blob",
           &StructureDatabase::ReadAllLineMatchesBlob)
      .def("read_all_line_matches", &StructureDatabase::ReadAllLineMatches)
      .def("read_group_matches_blob", &StructureDatabase::ReadGroupMatchesBlob)
      .def("read_group_matches", &StructureDatabase::ReadGroupMatches)
      .def("read_all_group_matches_blob",
           &StructureDatabase::ReadAllGroupMatchesBlob)
      .def("read_all_group_matches", &StructureDatabase::ReadAllGroupMatches)
      .def("write_structure2d", &StructureDatabase::WriteStructure2d)
      .def(
          "write_line_matches",
          static_cast<void (StructureDatabase::*)(
              colmap::image_t, colmap::image_t, const LineMatchesBlob &) const>(
              &StructureDatabase::WriteLineMatches))
      .def("write_group_matches",
           static_cast<void (StructureDatabase::*)(
               colmap::image_t, colmap::image_t, const GroupMatchesBlob &)
                           const>(&StructureDatabase::WriteGroupMatches))
      .def("update_structure2d", &StructureDatabase::UpdateStructure2d)
      .def("delete_line_matches", &StructureDatabase::DeleteLineMatches)
      .def("delete_group_matches", &StructureDatabase::DeleteGroupMatches)
      .def("clear_line_matches", &StructureDatabase::ClearLineMatches)
      .def("clear_group_matches", &StructureDatabase::ClearGroupMatches)
      .def("clear_all_tables", &StructureDatabase::ClearAllTables)
      .def("clear_structures2d", &StructureDatabase::ClearStructures2d);

  py::classh<StructureDatabaseTransactionWrapper>(
      m, "StructureDatabaseTransaction")
      .def(py::init<StructureDatabase *>(), py::arg("db"))
      .def("__enter__", &StructureDatabaseTransactionWrapper::Enter,
           py::return_value_policy::reference_internal)
      .def("__exit__", &StructureDatabaseTransactionWrapper::Exit);
}

} // namespace limap
