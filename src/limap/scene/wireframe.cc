#include "limap/scene/wireframe.h"
#include "limap/geometry/line_linker.h"

namespace limap {

std::shared_ptr<Wireframe2d> CreateWireframe2d(const std::vector<V2D> &points,
                                               const std::vector<Line2d> &lines,
                                               double threshold) {
  std::shared_ptr<Wireframe2d> wf = std::make_shared<Wireframe2d>();
  for (size_t pi = 0; pi < points.size(); ++pi) {
    for (size_t li = 0; li < lines.size(); ++li) {
      double d = lines[li].PointDistance(points[pi]);
      double w = ExpScore(d, threshold);
      if (w > 1e-6) {
        wf->AddEdge(pi, li, w);
      }
    }
  }
  return wf;
}

} // namespace limap
