#ifndef LIMAP_MERGING_AGGREGATOR_H_
#define LIMAP_MERGING_AGGREGATOR_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "base/linebase.h"

namespace limap {

namespace merging {

class Aggregator {
public:
  Aggregator() {}
  static Line3d
  aggregate_line3d_list_takebest(const std::vector<Line3d> &lines,
                                 const std::vector<double> &scores);

  static Line3d
  aggregate_line3d_list_takelongest(const std::vector<Line3d> &lines,
                                    const std::vector<double> &scores);

  static Line3d aggregate_line3d_list(const std::vector<Line3d> &lines,
                                      const std::vector<double> &scores,
                                      const int num_outliers = 2);
};

} // namespace merging

} // namespace limap

#endif
