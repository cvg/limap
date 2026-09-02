#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <thirdparty/pycolmap/pybind11_extension.h>

#include "limap/scene/holistic_reconstruction_manager.h"

using namespace pybind11::literals;
namespace py = pybind11;

namespace limap {

void BindHolisticReconstructionManager(py::module &m) {
  py::classh<HolisticReconstructionManager>(m, "HolisticReconstructionManager")
      .def(py::init<>())
      .def("size", &HolisticReconstructionManager::Size)
      .def("get",
           py::overload_cast<size_t>(&HolisticReconstructionManager::Get),
           "idx"_a)
      .def("add", &HolisticReconstructionManager::Add)
      .def("delete", &HolisticReconstructionManager::Delete, "idx"_a)
      .def("clear", &HolisticReconstructionManager::Clear)
      .def("read", &HolisticReconstructionManager::Read, "path"_a)
      .def("write", &HolisticReconstructionManager::Write, "path"_a);
}

} // namespace limap
