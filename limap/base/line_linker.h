#ifndef LIMAP_BASE_LINE_LINKER_H_
#define LIMAP_BASE_LINE_LINKER_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/line_dists.h"
#include "base/linebase.h"

#include <cmath>

namespace py = pybind11;

namespace limap {

double expscore(const double &val, const double &sigma);
double get_multiplier(const double &score_th);

class LineLinker2dConfig {
public:
  LineLinker2dConfig() {}
  LineLinker2dConfig(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, score_th, double)
    ASSIGN_PYDICT_ITEM(dict, th_angle, double)
    ASSIGN_PYDICT_ITEM(dict, th_overlap, double)
    ASSIGN_PYDICT_ITEM(dict, th_smartoverlap, double)
    ASSIGN_PYDICT_ITEM(dict, th_smartangle, double)
    ASSIGN_PYDICT_ITEM(dict, th_perp, double)
    ASSIGN_PYDICT_ITEM(dict, th_innerseg, double)
    ASSIGN_PYDICT_ITEM(dict, use_angle, bool)
    ASSIGN_PYDICT_ITEM(dict, use_overlap, bool)
    ASSIGN_PYDICT_ITEM(dict, use_smartangle, bool)
    ASSIGN_PYDICT_ITEM(dict, use_perp, bool)
    ASSIGN_PYDICT_ITEM(dict, use_innerseg, bool)
  }

  // for scoring
  double score_th = 0.5; // only score that is higher than 0.5 survives the test
  double multiplier() const { return get_multiplier(score_th); }

  // angle
  double th_angle = 8.0;
  bool use_angle = true;

  // overlap
  double th_overlap = 0.1;
  bool use_overlap = true;

  // smart angle
  double th_smartoverlap = 0.2;
  double th_smartangle = 1.0;
  bool use_smartangle = true;

  // perpendicular
  double th_perp = 5.0; // in pixels
  double use_perp = true;

  // innerseg (advanced perpendicular)
  double th_innerseg = 5.0; // in pixels
  bool use_innerseg = false;

  void set_to_default() {
    use_angle = true;
    use_overlap = true;
    use_perp = true;
    use_innerseg = false;
  }
};

class LineLinker2d {
public:
  LineLinker2d() {}
  LineLinker2d(const LineLinker2dConfig &config_) : config(config_) {}
  LineLinker2d(py::dict dict) : config(LineLinker2dConfig(dict)) {}
  LineLinker2dConfig config;

  bool check_connection(const Line2d &l1, const Line2d &l2) const;
  double compute_score(const Line2d &l1, const Line2d &l2) const;

private:
  bool check_connection_angle(const Line2d &l1, const Line2d &l2) const;
  bool check_connection_smartangle(const Line2d &l1, const Line2d &l2) const;
  bool check_connection_overlap(const Line2d &l1, const Line2d &l2) const;
  bool check_connection_perp(const Line2d &l1, const Line2d &l2) const;
  bool check_connection_innerseg(const Line2d &l1, const Line2d &l2) const;

  double compute_score_angle(const Line2d &l1, const Line2d &l2) const;
  double compute_score_overlap(const Line2d &l1, const Line2d &l2) const;
  double compute_score_smartangle(const Line2d &l1, const Line2d &l2) const;
  double compute_score_perp(const Line2d &l1, const Line2d &l2) const;
  double compute_score_innerseg(const Line2d &l1, const Line2d &l2) const;
};

class LineLinker3dConfig {
public:
  LineLinker3dConfig() {}
  LineLinker3dConfig(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, score_th, double)
    ASSIGN_PYDICT_ITEM(dict, th_angle, double)
    ASSIGN_PYDICT_ITEM(dict, th_overlap, double)
    ASSIGN_PYDICT_ITEM(dict, th_smartoverlap, double)
    ASSIGN_PYDICT_ITEM(dict, th_smartangle, double)
    ASSIGN_PYDICT_ITEM(dict, th_perp, double)
    ASSIGN_PYDICT_ITEM(dict, th_innerseg, double)
    ASSIGN_PYDICT_ITEM(dict, th_scaleinv, double)
    ASSIGN_PYDICT_ITEM(dict, use_angle, bool)
    ASSIGN_PYDICT_ITEM(dict, use_overlap, bool)
    ASSIGN_PYDICT_ITEM(dict, use_smartangle, bool)
    ASSIGN_PYDICT_ITEM(dict, use_perp, bool)
    ASSIGN_PYDICT_ITEM(dict, use_innerseg, bool)
    ASSIGN_PYDICT_ITEM(dict, use_scaleinv, bool)
  }

  // for scoring
  double score_th = 0.5; // only score that is higher than 0.5 survives the test
  double multiplier() const { return get_multiplier(score_th); }

  // angle
  double th_angle = 10.0;
  bool use_angle = true;

  // overlap
  double th_overlap = 0.01;
  bool use_overlap = true;

  // smart angle (use together with the overlap)
  double th_smartoverlap = 0.1;
  double th_smartangle = 1.0;
  bool use_smartangle = true;

  // perpendicular (this feature is abandoned since we have innerseg distance)
  double th_perp = 0.02;
  double use_perp = false;

  // innerseg (advanced perpendicular)
  double th_innerseg = 0.02;
  bool use_innerseg = true;

  // scale-invariance perpendicular similarity
  double th_scaleinv = 0.01;
  bool use_scaleinv = false;

  void set_to_default() { set_to_spatial_merging(); }

  void set_to_shared_parent_scoring() {
    use_angle = true;
    use_overlap = false;
    use_perp = false;
    use_innerseg = false;
    use_scaleinv = true;
  }

  void set_to_spatial_merging() {
    use_angle = true;
    use_overlap = true;
    use_perp = false;
    use_innerseg = true;
    use_scaleinv = false;
  }

  void set_to_avgtest_merging() {
    use_angle = true;
    use_overlap = false;
    use_perp = true;
    use_innerseg = false;
    use_scaleinv = false;
  }
};

class LineLinker3d {
public:
  LineLinker3d() {}
  LineLinker3d(const LineLinker3dConfig &config_) : config(config_) {}
  LineLinker3d(py::dict dict) : config(LineLinker3dConfig(dict)) {}
  LineLinker3dConfig config;

  bool check_connection(const Line3d &l1, const Line3d &l2) const;
  double compute_score(const Line3d &l1, const Line3d &l2) const;

private:
  bool check_connection_angle(const Line3d &l1, const Line3d &l2) const;
  bool check_connection_smartangle(const Line3d &l1, const Line3d &l2) const;
  bool check_connection_overlap(const Line3d &l1, const Line3d &l2) const;
  bool check_connection_perp(const Line3d &l1, const Line3d &l2) const;
  bool check_connection_innerseg(const Line3d &l1, const Line3d &l2) const;
  bool check_connection_scaleinv(const Line3d &l1, const Line3d &l2) const;

  double compute_score_angle(const Line3d &l1, const Line3d &l2) const;
  double compute_score_overlap(const Line3d &l1, const Line3d &l2) const;
  double compute_score_smartangle(const Line3d &l1, const Line3d &l2) const;
  double compute_score_perp(const Line3d &l1, const Line3d &l2) const;
  double compute_score_innerseg(const Line3d &l1, const Line3d &l2) const;
  double compute_score_scaleinv(const Line3d &l1, const Line3d &l2) const;
};

// joint 2d and 3d linker
class LineLinker {
public:
  LineLinker() {}
  LineLinker(const LineLinker2d &linker_2d_, const LineLinker3d &linker_3d_)
      : linker_2d(linker_2d_), linker_3d(linker_3d_) {}
  LineLinker(const LineLinker2dConfig &config2d,
             const LineLinker3dConfig &config3d)
      : linker_2d(config2d), linker_3d(config3d) {}
  LineLinker(py::dict dict2d, py::dict dict3d)
      : linker_2d(LineLinker2dConfig(dict2d)),
        linker_3d(LineLinker3dConfig(dict3d)) {}

  LineLinker2d linker_2d;
  LineLinker3d linker_3d;

  LineLinker2d GetLinker2d() const { return linker_2d; }
  LineLinker3d GetLinker3d() const { return linker_3d; }

  bool check_connection_2d(const Line2d &l1, const Line2d &l2) const {
    return linker_2d.check_connection(l1, l2);
  }
  double compute_score_2d(const Line2d &l1, const Line2d &l2) const {
    return linker_2d.compute_score(l1, l2);
  }
  bool check_connection_3d(const Line3d &l1, const Line3d &l2) const {
    return linker_3d.check_connection(l1, l2);
  }
  double compute_score_3d(const Line3d &l1, const Line3d &l2) const {
    return linker_3d.compute_score(l1, l2);
  }
};

} // namespace limap

#endif
