#include <optional>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <thirdparty/pycolmap/helpers.h>

#include "limap/estimators/bundle_adjustment/structure_bundle_adjustment.h"
#include "limap/sfm/global_structure_triangulation.h"
#include "limap/sfm/group_verification.h"

namespace limap {

void BindGlobalStructureTriangulation(py::module &m) {
  m.def(
      "global_triangulate_structure",
      [](const StructureDatabaseCache &sdc,
         const std::shared_ptr<HolisticReconstruction> &recon,
         const GlobalLineTriangulationOptions &lo,
         const GlobalGroupTriangulationOptions &go,
         std::optional<estimators::StructureBundleAdjustmentOptions> ba,
         std::optional<GroupVerificationOptions> fo,
         const ExhaustiveMatchNeighbors &exhaustive_match_neighbors) {
        GlobalTriangulateStructure(sdc, recon, lo, go, ba ? &*ba : nullptr,
                                   fo ? &*fo : nullptr,
                                   exhaustive_match_neighbors);
      },
      "structure_db_cache"_a, "reconstruction"_a, "line_options"_a,
      "group_options"_a, "ba_options"_a = py::none(),
      "filter_options"_a = py::none(),
      "exhaustive_match_neighbors"_a = ExhaustiveMatchNeighbors());
}

} // namespace limap
