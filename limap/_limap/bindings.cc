#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>
#include <thread>

namespace py = pybind11;

#include "base/bindings.cc"
#include "ceresbase/bindings.cc"
#include "estimators/bindings.cc"
#include "evaluation/bindings.cc"
#include "fitting/bindings.cc"
#include "merging/bindings.cc"
#include "optimize/bindings.cc"
#include "pointsfm/bindings.cc"
#include "structures/bindings.cc"
#include "triangulation/bindings.cc"
#include "undistortion/bindings.cc"
#include "vplib/bindings.cc"
#ifdef INTERPOLATION_ENABLED
#include "features/bindings.cc"
#endif // INTERPOLATION_ENABLED

#include "_limap/helpers.h"

void bind_base(py::module &);
void bind_ceresbase(py::module &);
void bind_pointsfm(py::module &);
void bind_triangulation(py::module &);
void bind_merging(py::module &);
void bind_undistortion(py::module &);
void bind_vplib(py::module &);
void bind_evaluation(py::module &);
void bind_fitting(py::module &);
void bind_estimators(py::module &);
void bind_optimize(py::module &);
void bind_structures(py::module &);
#ifdef INTERPOLATION_ENABLED
void bind_features(py::module &);
#endif // INTERPOLATION_ENABLED

namespace limap {

PYBIND11_MODULE(_limap, m) {
  m.doc() = "A toolbox for mapping and localization with line features";
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("dev");
#endif
  m.attr("__ceres_version__") = py::str(CERES_VERSION_STRING);

  py::add_ostream_redirect(m, "ostream_redirect");

  pybind11::module_ _b = m.def_submodule("_base");
  pybind11::module_ _ceresb = m.def_submodule("_ceresbase");
  pybind11::module_ _pointsfm = m.def_submodule("_pointsfm");
  pybind11::module_ _tri = m.def_submodule("_triangulation");
  pybind11::module_ _mrg = m.def_submodule("_merging");
  pybind11::module_ _undist = m.def_submodule("_undistortion");
  pybind11::module_ _f = m.def_submodule("_features");
  pybind11::module_ _vplib = m.def_submodule("_vplib");
  pybind11::module_ _eval = m.def_submodule("_evaluation");
  pybind11::module_ _fitting = m.def_submodule("_fitting");
  pybind11::module_ _estimators = m.def_submodule("_estimators");
  pybind11::module_ _optim = m.def_submodule("_optimize");
  pybind11::module_ _structures = m.def_submodule("_structures");

  // bind modules
  bind_base(_b);
  bind_ceresbase(_ceresb);
  bind_pointsfm(_pointsfm);
  bind_triangulation(_tri);
  bind_merging(_mrg);
  bind_undistortion(_undist);
  bind_vplib(_vplib);
  bind_evaluation(_eval);
  bind_fitting(_fitting);
  bind_estimators(_estimators);
  bind_optimize(_optim);
  bind_structures(_structures);
#ifdef INTERPOLATION_ENABLED
  bind_features(_f);
#endif // INTERPOLATION_ENABLED
}

} // namespace limap
