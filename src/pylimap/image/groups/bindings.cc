#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "limap/image/groups/group_voting.h"
#include "pylimap/helpers.h"
#include <Eigen/Core>
#include <thirdparty/pycolmap/helpers.h>
#include <vector>

namespace limap {

void bind_vplib(py::module &);

void bind_group_voting(py::module &m) {
  py::classh<GroupVotingOptions> PyGroupVotingOptions(m, "GroupVotingOptions");
  PyGroupVotingOptions.def(py::init<>())
      .def_readwrite("point_weight", &GroupVotingOptions::point_weight)
      .def_readwrite("line_weight", &GroupVotingOptions::line_weight)
      .def_readwrite("use_feature_weights",
                     &GroupVotingOptions::use_feature_weights)
      .def_readwrite("min_num_votes", &GroupVotingOptions::min_num_votes);
  MakeDataclass(PyGroupVotingOptions);

  m.def("match_groups_by_voting", &MatchGroupsByVoting, py::arg("options"),
        py::arg("point_matches"), py::arg("line_matches"),
        py::arg("structure1"), py::arg("structure2"),
        py::arg("matched_groups1"), py::arg("matched_groups2"),
        "Match unmatched groups by voting from matched points and lines.");
}

void bind_groups(py::module &m) {
  pybind11::module_ _vplib = m.def_submodule("_vplib");
  bind_vplib(_vplib);
  bind_group_voting(m);
}

} // namespace limap
