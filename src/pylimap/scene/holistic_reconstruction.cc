#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "limap/scene/holistic_reconstruction.h"
#include <thirdparty/pycolmap/helpers.h>
#include <thirdparty/pycolmap/pybind11_extension.h>

namespace py = pybind11;

namespace limap {

void BindHolisticReconstruction(py::module &m) {
  using HR = HolisticReconstruction;
  py::classh<HR>(m, "HolisticReconstruction")
      .def(py::init<>())
      .def(py::init<const colmap::Reconstruction &>(), py::arg("point_recon"))
      .def(py::init([](const std::string &path) {
             auto reconstruction = std::make_shared<HolisticReconstruction>();
             reconstruction->Read(path);
             return reconstruction;
           }),
           "path"_a)
      .def("read", &HR::Read)
      .def("write", &HR::Write)
      .def("read_binary", &HR::ReadBinary)
      .def("read_text", &HR::ReadText)
      .def("write_binary", &HR::WriteBinary)
      .def("write_text", &HR::WriteText)
      .def_property_readonly(
          "point_recon",
          static_cast<colmap::Reconstruction &(HR::*)()>(&HR::PointRecon),
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "structure_recon",
          static_cast<StructureReconstruction &(HR::*)()>(&HR::StructureRecon),
          py::return_value_policy::reference_internal)
      .def("check_validity", &HolisticReconstruction::CheckValidity)
      .def_static("exists_model", &HR::ExistsModel, "path"_a,
                  "Check if a complete holistic reconstruction exists at path")
      .def("__repr__", [](const HolisticReconstruction &s) {
        std::ostringstream oss;
        oss << "HolisticReconstruction(PointRecon(num_cameras="
            << s.PointRecon().NumCameras()
            << ", num_images=" << s.PointRecon().NumRegImages()
            << ", num_points=" << s.PointRecon().NumPoints3D()
            << "), StructureRecon(num_lines3D="
            << s.StructureRecon().NumLines3D()
            << ", num_groups3D=" << s.StructureRecon().NumGroups3D()
            << ", num_wf_edges=" << s.StructureRecon().Wireframe().CountEdges()
            << "))";
        return oss.str();
      });
}

} // namespace limap
