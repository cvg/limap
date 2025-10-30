#pragma once

#include "limap/internal/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "limap/base/graph.h"
#include "limap/base/image_collection.h"
#include "limap/base/line_linker.h"
#include "limap/base/linebase.h"
#include "limap/base/linetrack.h"

#include <set>

namespace limap {

namespace merging {

// Modified track computation based on Kruskal
std::vector<int>
ComputeLineTrackLabelsGreedy(const Graph &graph,
                             const std::vector<Line3d> &line3d_list_nodes);
std::vector<int>
ComputeLineTrackLabelsExhaustive(const Graph &graph,
                                 const std::vector<Line3d> &line3d_list_nodes,
                                 LineLinker3d linker3d);
std::vector<int>
ComputeLineTrackLabelsAvg(const Graph &graph,
                          const std::vector<Line3d> &line3d_list_nodes,
                          LineLinker3d linker3d);

// for fitnmerge application
void MergeToLineTracks(Graph &graph, std::vector<LineTrack> &linetracks,
                       const std::map<int, std::vector<Line2d>> &all_lines_2d,
                       const ImageCollection &imagecols,
                       const std::map<int, std::vector<Line3d>> &all_lines_3d,
                       const std::map<int, std::vector<int>> &neighbors,
                       LineLinker linker);

// remerge line tracks
std::vector<LineTrack>
RemergeLineTracks(const std::vector<LineTrack> &linetracks,
                  LineLinker3d linker3d, const int num_outliers = 2);

} // namespace merging

} // namespace limap
