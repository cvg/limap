#include "triangulation/build_initial_graph.h"
#include "util/types.h"
#include "base/linebase.h"

namespace limap {

namespace triangulation {

void BuildInitialGraph(DirectedGraph& graph,
                       const std::vector<std::vector<Eigen::MatrixXi>>& all_matches,
                       const std::vector<std::vector<int>>& neighbors) 
{
    // check
    THROW_CHECK_EQ(all_matches.size(), neighbors.size());
    THROW_CHECK_EQ(all_matches[0].size(), neighbors[0].size());

    // construct graph
    int n_images = neighbors.size();
    for (int i = 0; i < n_images; ++i) {
        int n_neighbors = neighbors[i].size();
        for (int j = 0; j < n_neighbors; ++j) {
            int image_id = i;
            int neighbor_id = neighbors[i][j];

            // get matches
            const Eigen::MatrixXi& match_info = all_matches[i][j];
            if (match_info.rows() != 0) {
                THROW_CHECK_EQ(match_info.cols(), 2);
            }

            size_t n_matches = match_info.rows();
            std::vector<size_t> matches;
            std::vector<double> similarities;
            for (int k = 0; k < n_matches; ++k) {
                // good match exists
                matches.push_back(match_info(k, 0));
                matches.push_back(match_info(k, 1));
                similarities.push_back(1.0); // similarity does not matter in the initial graph 
            }
            graph.RegisterMatchesDirected(image_id, neighbor_id, matches.data(), similarities.data(), n_matches);
        }
    }
}

} // namespace triangulation

} // namespace limap

