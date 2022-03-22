#ifndef LIMAP_TRIANGULATION_BUILD_INITIAL_GRAPH_H_
#define LIMAP_TRIANGULATION_BUILD_INITIAL_GRAPH_H_

#include "base/graph.h"
#include "base/linebase.h"

namespace limap {

namespace triangulation {

void BuildInitialGraph(DirectedGraph& graph,
                       const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                       const std::vector<std::vector<int>>& neighbors);

} // namespace triangulation

} // namespace limap

#endif

