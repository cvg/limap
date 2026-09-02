#pragma once

#include <unordered_map>
#include <vector>

#include <colmap/scene/correspondence_graph.h>
#include <colmap/scene/image.h>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/geometry/line_linker.h"
#include "limap/scene/structure_reconstruction.h"
#include "limap/util/types.h"

namespace limap {

// Images each image is exhaustively paired against.
using ExhaustiveMatchNeighbors =
    std::unordered_map<colmap::image_t, std::vector<colmap::image_t>>;

// Triangulation candidate for a given reference 2D line.
struct LineTriangulationProposal {
  // Triangulated 3D line
  Line3d line;

  // 2D line that generated the proposal
  Node2d supporting_node;

  // Multi-view support score
  double score = 0.0;

  // depth of line.start in ref image
  double depth_start;

  // depth of line.end in ref image
  double depth_end;

  LineTriangulationProposal() = default;

  LineTriangulationProposal(const Line3d &line_, const Node2d &supporting_node_,
                            const colmap::Rigid3d &cam_from_world)
      : line(line_), supporting_node(supporting_node_) {
    depth_start = (cam_from_world * line.start).z();
    depth_end = (cam_from_world * line.end).z();
  }
};

// Parameters for line triangulation helper functions.
// Used by both GlobalLineTriangulationOptions and
// IncrementalLineTriangulator::Options.
struct LineTriangulationParams {
  // 2D variance for uncertainty computation
  double var2d = 2.0;

  // Epipolar IoU threshold
  double IoU_threshold = 0.1;

  // Maximum sensitivity angle (degrees)
  double sensitivity_threshold = 70.0;
};

namespace internal {
namespace line_triangulation {

// Convenience alias
using ProposalList = std::vector<LineTriangulationProposal>;

// Edge for similarity graph during clustering
struct EdgeRecord {
  double weight = 0.0;
  Node2d n1;
  Node2d n2;
};

// Collect shared 3D points for two 2D lines using Wireframe2d + COLMAP.
// Only needs StructureReconstruction; internally calls srec.PointRecon().
void CollectShared3DPoints(const Node2d &node1, const Node2d &node2,
                           const StructureReconstruction &srec,
                           std::vector<V3D> &points3d);

// Try to obtain a 3D direction from VP information attached to a line.
// Only needs StructureReconstruction; internally calls srec.PointRecon().
bool GetVPDirectionForLine(const StructureReconstruction &srec,
                           const Structure2d &S, const Node2d &node,
                           V3D &direction);

// Step 1: point-based triangulation (many points / one point).
void AddPointBasedProposalsForMatch(const Node2d &node, const Line2d &l1,
                                    const colmap::Image &image1,
                                    const Node2d &ng_node, const Line2d &l2,
                                    const colmap::Image &image2,
                                    const StructureReconstruction &srec,
                                    const LineTriangulationParams &params,
                                    ProposalList &proposals);

// Step 2: triangulation with vanishing points (optional).
void AddVPBasedProposalsForMatch(const Node2d &node, const Line2d &l1,
                                 const colmap::Image &image1,
                                 const Node2d &ng_node, const Line2d &l2,
                                 const colmap::Image &image2,
                                 const Structure2d &S1, const Structure2d &S2,
                                 const StructureReconstruction &srec,
                                 const LineTriangulationParams &params,
                                 ProposalList &proposals);

// Step 3: algebraic line triangulation with epipolar IoU test.
void AddEpipolarProposalsForMatch(const Line2d &l1, const colmap::Image &image1,
                                  const Line2d &l2, const colmap::Image &image2,
                                  const Node2d &ng_node,
                                  const LineTriangulationParams &params,
                                  ProposalList &proposals);

// Support scoring for one candidate given all proposals.
double ComputeTotalSupportScore(std::size_t target_idx,
                                const ProposalList &proposals,
                                const StructureReconstruction &srec,
                                const LineLinker &linker, double th_scale_inv);

// Pick best supported proposal index and score, or {-1, best_score}.
std::pair<int, double> SelectBestSupportedProposal(
    const ProposalList &proposals, const StructureReconstruction &srec,
    const LineLinker &linker, double scale_inv_th, double fullscore_th);

// Build similarity edges between nodes (3D + 2D consistency).
// Candidate pairs come from corr_graph, or from the neighbors when those are
// supplied. Sorts edges in descending weight.
std::vector<EdgeRecord> BuildLineSimilarityEdges(
    const std::vector<Node2d> &nodes,
    const Node2dMap<LineTriangulationProposal> &best_tris,
    StructureReconstruction &srec, const LineLinker &linker_clustering,
    const colmap::CorrespondenceGraph &corr_graph, int num_threads = -1,
    const ExhaustiveMatchNeighbors &exhaustive_match_neighbors = {});

// Build UF-based clusters from edges: root -> members.
Node2dMap<std::vector<Node2d>>
BuildLineClusters(const std::vector<Node2d> &nodes,
                  const std::vector<EdgeRecord> &edges);

} // namespace line_triangulation
} // namespace internal
} // namespace limap
