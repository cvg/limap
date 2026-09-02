#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "pylimap/helpers.h"

namespace limap {

void BindCOLMAPMVSModel(py::module &m);
void BindGroup2d(py::module &m);
void BindGroup3d(py::module &m);
void BindWireframe(py::module &m);
void BindStructure2d(py::module &m);
void BindLine3dWithActiveLabels(py::module &m);
void BindStructureDatabase(py::module &m);
void BindStructureCorrespondenceGraph(py::module &m);
void BindStructureDatabaseCache(py::module &m);
void BindStructureReconstruction(py::module &m);
void BindHolisticReconstruction(py::module &m);
void BindHolisticReconstructionManager(py::module &m);

void bind_scene(py::module &m) {
  BindCOLMAPMVSModel(m);
  BindGroup2d(m);
  BindGroup3d(m);
  BindWireframe(m);
  BindStructure2d(m);
  // Must be bound before StructureReconstruction (which uses
  // Line3dWithActiveLabels in its lines3D map)
  BindLine3dWithActiveLabels(m);
  BindStructureDatabase(m);
  BindStructureCorrespondenceGraph(m);
  BindStructureDatabaseCache(m);
  BindStructureReconstruction(m);
  BindHolisticReconstruction(m);
  BindHolisticReconstructionManager(m);
}

} // namespace limap
