#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <vector>

namespace py = pybind11;

#include "limap/estimators/triangulation/functions.h"

namespace limap {

void bind_triangulation(py::module &m) {
  using namespace estimators::triangulation;

  m.def("get_normal_direction", &GetNormalDirection);
  m.def("get_direction_from_VP", &GetDirectionFromVP);
  m.def("compute_essential_matrix", &ComputeEssentialMatrix);
  m.def("compute_fundamental_matrix", &ComputeFundamentalMatrix);
  m.def("compute_epipolar_IoU", &ComputeEpipolarIoU);
  m.def("triangulate_point", &TriangulatePoint);
  m.def("triangulate_line_by_endpoints", &TriangulateLineByEndpoints);
  m.def("triangulate_line", &TriangulateLine);
  m.def("triangulate_line_with_one_point", &TriangulateLineWithOnePoint);
  m.def("triangulate_line_with_direction", &TriangulateLineWithDirection);
}

} // namespace limap
