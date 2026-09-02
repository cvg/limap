#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/geometry/line_metrics.h"

namespace limap {

double GetMultiplierFromThreshold(double score_th);
double ExpScore(const double &val, const double &sigma);

class LineLinker2dOptions {
public:
  LineLinker2dOptions();

  // Component score threshold (>= keeps, < becomes 0).
  double score_th = 0.5;
  double Multiplier() const; // derived from score_th

  // Angle
  double th_angle = 5.0;
  bool use_angle = true;

  // Overlap
  double th_overlap = 0.05;
  bool use_overlap = true;

  // Smart angle (requires angle + overlap)
  double th_smartoverlap = 0.2;
  double th_smartangle = 1.0;
  bool use_smartangle = true;

  // Perpendicular (pixels)
  double th_perp = 2.0;
  bool use_perp = true;

  // Inner segment (advanced perpendicular)
  double th_innerseg = 2.0;
  bool use_innerseg = false;

  void SetToDefault();
  static LineLinker2dOptions BuildDefaultOptions(); // Same as SetToDefault()
};

class LineLinker2d {
public:
  using Pred = std::function<bool(const Line2d &, const Line2d &)>;
  using Scorer = std::function<double(const Line2d &, const Line2d &)>;

  LineLinker2d(); // builds with defaults
  explicit LineLinker2d(
      const LineLinker2dOptions &); // builds with custom options

  bool CheckConnection(const Line2d &l1, const Line2d &l2) const;
  double ComputeScore(const Line2d &l1, const Line2d &l2) const;

  // Append a custom predicate / scorer to the end of the pipeline
  void AddCheck(Pred f) { checks_.push_back(std::move(f)); }
  void AddScorer(Scorer f) { scorers_.push_back(std::move(f)); }

  // Prepend a custom predicate / scorer to the beginning (runs earlier)
  void PrependCheck(Pred f) { checks_.insert(checks_.begin(), std::move(f)); }
  void PrependScorer(Scorer f) {
    scorers_.insert(scorers_.begin(), std::move(f));
  }

  // Manage collections directly if you need full control
  void ClearChecks() { checks_.clear(); }
  void ClearScorers() { scorers_.clear(); }

private:
  std::vector<Pred> checks_;
  std::vector<Scorer> scorers_;

  void _BuildPipelines(const LineLinker2dOptions &);
};

class LineLinker3dOptions {
public:
  LineLinker3dOptions();

  double score_th = 0.5;
  double Multiplier() const;

  // Angle
  double th_angle = 10.0;
  bool use_angle = true;

  // Overlap
  double th_overlap = 0.001;
  bool use_overlap = true;

  // Smart angle (requires angle + overlap)
  double th_smartoverlap = 0.1;
  double th_smartangle = 1.0;
  bool use_smartangle = true;

  // Perpendicular (prefer innerseg for most cases)
  double th_perp = 1.0;
  bool use_perp = false;

  // Inner segment (advanced perpendicular)
  double th_innerseg = 1.0;
  bool use_innerseg = true;

  // Presets
  void SetToDefault(); // Spatial merging preset

  // For scoring proposals for each 2D node, we only check the angle and check
  // scale-invariant metric with depth information outside the linker.
  void SetToSharedParentScoring() {
    use_angle = true;
    use_overlap = false;
    use_perp = false;
    use_innerseg = false;
  }

  // For spatial merging
  void SetToSpatialMerging() {
    use_angle = true;
    use_overlap = true;
    use_perp = false;
    use_innerseg = true;
  }

  static LineLinker3dOptions BuildDefaultOptions();
  static LineLinker3dOptions BuildSharedParentScoring();
  static LineLinker3dOptions BuildSpatialMerging();
};

class LineLinker3d {
public:
  using Pred = std::function<bool(const Line3d &, const Line3d &)>;
  using Scorer = std::function<double(const Line3d &, const Line3d &)>;

  LineLinker3d(); // builds with defaults
  explicit LineLinker3d(
      const LineLinker3dOptions &); // builds with custom options

  bool CheckConnection(const Line3d &l1, const Line3d &l2) const;
  double ComputeScore(const Line3d &l1, const Line3d &l2) const;

  void AddCheck(Pred f) { checks_.push_back(std::move(f)); }
  void AddScorer(Scorer f) { scorers_.push_back(std::move(f)); }
  void PrependCheck(Pred f) { checks_.insert(checks_.begin(), std::move(f)); }
  void PrependScorer(Scorer f) {
    scorers_.insert(scorers_.begin(), std::move(f));
  }
  void ClearChecks() { checks_.clear(); }
  void ClearScorers() { scorers_.clear(); }

private:
  std::vector<Pred> checks_;
  std::vector<Scorer> scorers_;

  void _BuildPipelines(const LineLinker3dOptions &);
};

class LineLinker {
public:
  LineLinker()
      : linker_2d(LineLinker2dOptions::BuildDefaultOptions()),
        linker_3d(LineLinker3dOptions::BuildDefaultOptions()) {}

  LineLinker(const LineLinker2d &l2d, const LineLinker3d &l3d)
      : linker_2d(l2d), linker_3d(l3d) {}

  explicit LineLinker(const LineLinker2dOptions &o2d,
                      const LineLinker3dOptions &o3d)
      : linker_2d(o2d), linker_3d(o3d) {}

  LineLinker2d &Linker2d() { return linker_2d; }
  const LineLinker2d &Linker2d() const { return linker_2d; }
  LineLinker3d &Linker3d() { return linker_3d; }
  const LineLinker3d &Linker3d() const { return linker_3d; }

  bool CheckConnection2d(const Line2d &a, const Line2d &b) const {
    return linker_2d.CheckConnection(a, b);
  }
  double ComputeScore2d(const Line2d &a, const Line2d &b) const {
    return linker_2d.ComputeScore(a, b);
  }
  bool CheckConnection3d(const Line3d &a, const Line3d &b) const {
    return linker_3d.CheckConnection(a, b);
  }
  double ComputeScore3d(const Line3d &a, const Line3d &b) const {
    return linker_3d.ComputeScore(a, b);
  }

  void AddCheck2d(LineLinker2d::Pred f) { linker_2d.AddCheck(std::move(f)); }
  void AddScorer2d(LineLinker2d::Scorer f) {
    linker_2d.AddScorer(std::move(f));
  }
  void AddCheck3d(LineLinker3d::Pred f) { linker_3d.AddCheck(std::move(f)); }
  void AddScorer3d(LineLinker3d::Scorer f) {
    linker_3d.AddScorer(std::move(f));
  }

private:
  LineLinker2d linker_2d;
  LineLinker3d linker_3d;
};

} // namespace limap
