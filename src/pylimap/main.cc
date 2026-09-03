#include <ceres/version.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace limap {

void bind_util(py::module &);
void bind_evaluation(py::module &);
void bind_estimators(py::module &);
void bind_scene(py::module &);
void bind_image(py::module &);
void bind_geometry(py::module &);
void bind_sfm(py::module &);

PYBIND11_MODULE(_limap, m) {
  m.doc() =
      "A toolbox for 3D visual mapping, localization, and SfM with structured "
      "geometric features (points, lines, planes, vanishing points, "
      "wireframes, etc.).";
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("dev");
#endif
  m.attr("__ceres_version__") = py::str(CERES_VERSION_STRING);

  py::add_ostream_redirect(m, "ostream_redirect");

  pybind11::module_ _eval = m.def_submodule("_evaluation");
  pybind11::module_ _estimators = m.def_submodule("_estimators");
  pybind11::module_ _scene = m.def_submodule("_scene");
  pybind11::module_ _image = m.def_submodule("_image");
  pybind11::module_ _geometry = m.def_submodule("_geometry");
  pybind11::module_ _util = m.def_submodule("_util");
  pybind11::module_ _sfm = m.def_submodule("_sfm");

  // bind modules
  // geometry must come before scene (Line3dWithActiveLabels depends on Line3d)
  bind_geometry(_geometry);
  bind_evaluation(_eval);
  bind_estimators(_estimators);
  bind_scene(_scene);
  bind_image(_image);
  bind_util(_util);
  bind_sfm(_sfm);
}

} // namespace limap
