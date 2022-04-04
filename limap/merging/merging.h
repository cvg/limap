#ifndef LIMAP_MERGING_MERGING_H_
#define LIMAP_MERGING_MERGING_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

namespace py = pybind11;

#include "base/graph.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/line_linker.h"
#include "base/camera_view.h"

#include <set>

namespace limap {

namespace merging {

// Modified track computation based on Kruskal 
std::vector<int> ComputeLineTrackLabelsGreedy(const Graph& graph, 
                                              const std::vector<Line3d>& line3d_list_nodes);
std::vector<int> ComputeLineTrackLabelsExhaustive(const Graph& graph, 
                                                  const std::vector<Line3d>& line3d_list_nodes, 
                                                  LineLinker3d linker3d);
std::vector<int> ComputeLineTrackLabelsAvg(const Graph& graph, 
                                           const std::vector<Line3d>& line3d_list_nodes, 
                                           LineLinker3d linker3d);

// for fitnmerge application
void MergeToLineTracks(Graph& graph,
                       std::vector<LineTrack>& linetracks,
                       const std::vector<std::vector<Line2d>>& all_lines_2d,
                       const std::vector<CameraView>& views,
                       const std::vector<std::vector<Line3d>>& all_lines_3d,
                       const std::vector<std::vector<int>>& neighbors,
                       LineLinker linker);

// remerge line tracks
std::vector<LineTrack> RemergeLineTracks(const std::vector<LineTrack>& linetracks,
                                         LineLinker3d linker3d,
                                         const int num_outliers = 2);

} // namespace merging

} // namespace limap

#endif

