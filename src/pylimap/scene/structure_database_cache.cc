#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/structure_database_cache.h"

namespace py = pybind11;

namespace limap {

void BindStructureDatabaseCache(py::module &m) {
  py::classh<StructureDatabaseCache>(m, "StructureDatabaseCache")
      .def(py::init<>())
      .def_static("create", &StructureDatabaseCache::Create,
                  py::arg("struct_db"))
      .def("structure2d",
           static_cast<Structure2d &(
               StructureDatabaseCache::*)(colmap::image_t)>(
               &StructureDatabaseCache::Structure2d),
           py::arg("image_id"), py::return_value_policy::reference_internal)
      .def_property_readonly("structures2d",
                             &StructureDatabaseCache::Structures2d,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "structure_correspondence_graph",
          &StructureDatabaseCache::StructureCorrespondenceGraph);
}

} // namespace limap
