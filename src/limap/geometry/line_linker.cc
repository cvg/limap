#include "limap/geometry/line_linker.h"
#include "limap/geometry/line_metrics.h"

#include <cmath>

namespace limap {

double GetMultiplierFromThreshold(double score_th) {
  // exp(- (val / sigma)^2 / 2.0) >= T  <=>  val <= sqrt(-2 ln T) * sigma
  return 1.0 / std::sqrt(-std::log(score_th) * 2.0);
}

double ExpScore(const double &val, const double &sigma) {
  return std::exp(-std::pow(val / sigma, 2) / 2.0);
}

LineLinker2dOptions::LineLinker2dOptions() {}

double LineLinker2dOptions::Multiplier() const {
  return GetMultiplierFromThreshold(score_th);
}

void LineLinker2dOptions::SetToDefault() {
  use_angle = true;
  use_overlap = true;
  use_perp = true;
  use_innerseg = false;
}

LineLinker2dOptions LineLinker2dOptions::BuildDefaultOptions() {
  LineLinker2dOptions o;
  o.SetToDefault();
  return o;
}

LineLinker2d::LineLinker2d()
    : LineLinker2d(LineLinker2dOptions::BuildDefaultOptions()) {}

LineLinker2d::LineLinker2d(const LineLinker2dOptions &o) { _BuildPipelines(o); }

bool LineLinker2d::CheckConnection(const Line2d &l1, const Line2d &l2) const {
  for (const auto &f : checks_) {
    if (!f(l1, l2))
      return false;
  }
  return true;
}

double LineLinker2d::ComputeScore(const Line2d &l1, const Line2d &l2) const {
  double score = 1.0;
  for (const auto &s : scorers_) {
    score = std::min(score, s(l1, l2));
    if (score == 0.0)
      break;
  }
  return score;
}

void LineLinker2d::_BuildPipelines(const LineLinker2dOptions &o) {
  checks_.clear();
  scorers_.clear();

  // Angle
  if (o.use_angle) {
    checks_.emplace_back([o](const Line2d &a, const Line2d &b) {
      return ComputeAngle<Line2d>(a, b) <= o.th_angle;
    });
    scorers_.emplace_back([o](const Line2d &a, const Line2d &b) {
      const double ang = ComputeAngle<Line2d>(a, b);
      const double s = ExpScore(ang, o.th_angle * o.Multiplier());
      return (s < o.score_th) ? 0.0 : s;
    });
  }

  // Overlap
  if (o.use_overlap) {
    checks_.emplace_back([o](const Line2d &a, const Line2d &b) {
      return ComputeBioverlap<Line2d>(a, b) > o.th_overlap;
    });
    scorers_.emplace_back([o](const Line2d &a, const Line2d &b) {
      return (ComputeBioverlap<Line2d>(a, b) > o.th_overlap) ? 1.0 : 0.0;
    });
  }

  // Smart angle (requires angle + overlap)
  if (o.use_angle && o.use_overlap && o.use_smartangle) {
    auto smart_angle_sigma = [o](const Line2d &a, const Line2d &b) {
      const double overlap = ComputeBioverlap<Line2d>(a, b);
      double th = o.th_angle;
      if (overlap < o.th_smartoverlap) {
        const double r = std::min((o.th_smartoverlap - overlap) /
                                      (o.th_smartoverlap - o.th_overlap),
                                  1.0);
        th = o.th_angle - r * (o.th_angle - o.th_smartangle);
      }
      return th * o.Multiplier();
    };
    checks_.emplace_back([o, smart_angle_sigma](const Line2d &a,
                                                const Line2d &b) {
      return ExpScore(ComputeAngle<Line2d>(a, b), smart_angle_sigma(a, b)) >=
             o.score_th;
    });
    scorers_.emplace_back(
        [o, smart_angle_sigma](const Line2d &a, const Line2d &b) {
          const double s =
              ExpScore(ComputeAngle<Line2d>(a, b), smart_angle_sigma(a, b));
          return (s < o.score_th) ? 0.0 : s;
        });
  }

  // Perpendicular
  if (o.use_perp) {
    checks_.emplace_back([o](const Line2d &a, const Line2d &b) {
      const double d =
          ComputeDistance<Line2d>(a, b, LineMetricType::PERPENDICULAR);
      return ExpScore(d, o.th_perp * o.Multiplier()) >= o.score_th;
    });
    scorers_.emplace_back([o](const Line2d &a, const Line2d &b) {
      const double d =
          ComputeDistance<Line2d>(a, b, LineMetricType::PERPENDICULAR);
      const double s = ExpScore(d, o.th_perp * o.Multiplier());
      return (s < o.score_th) ? 0.0 : s;
    });
  }

  // Inner segment
  if (o.use_innerseg) {
    checks_.emplace_back([o](const Line2d &a, const Line2d &b) {
      const double d = ComputeDistance<Line2d>(a, b, LineMetricType::INNERSEG);
      return ExpScore(d, o.th_innerseg * o.Multiplier()) >= o.score_th;
    });
    scorers_.emplace_back([o](const Line2d &a, const Line2d &b) {
      const double d = ComputeDistance<Line2d>(a, b, LineMetricType::INNERSEG);
      const double s = ExpScore(d, o.th_innerseg * o.Multiplier());
      return (s < o.score_th) ? 0.0 : s;
    });
  }
}

LineLinker3dOptions::LineLinker3dOptions() {}

double LineLinker3dOptions::Multiplier() const {
  return GetMultiplierFromThreshold(score_th);
}

void LineLinker3dOptions::SetToDefault() { SetToSpatialMerging(); }

LineLinker3dOptions LineLinker3dOptions::BuildDefaultOptions() {
  LineLinker3dOptions o;
  o.SetToDefault();
  return o;
}
LineLinker3dOptions LineLinker3dOptions::BuildSharedParentScoring() {
  LineLinker3dOptions o;
  o.SetToSharedParentScoring();
  return o;
}
LineLinker3dOptions LineLinker3dOptions::BuildSpatialMerging() {
  LineLinker3dOptions o;
  o.SetToSpatialMerging();
  return o;
}

LineLinker3d::LineLinker3d()
    : LineLinker3d(LineLinker3dOptions::BuildDefaultOptions()) {}

LineLinker3d::LineLinker3d(const LineLinker3dOptions &o) { _BuildPipelines(o); }

bool LineLinker3d::CheckConnection(const Line3d &l1, const Line3d &l2) const {
  for (const auto &f : checks_) {
    if (!f(l1, l2))
      return false;
  }
  return true;
}

double LineLinker3d::ComputeScore(const Line3d &l1, const Line3d &l2) const {
  double score = 1.0;
  for (const auto &s : scorers_) {
    score = std::min(score, s(l1, l2));
    if (score == 0.0)
      break;
  }
  return score;
}

void LineLinker3d::_BuildPipelines(const LineLinker3dOptions &o) {
  checks_.clear();
  scorers_.clear();

  // Angle
  if (o.use_angle) {
    checks_.emplace_back([o](const Line3d &a, const Line3d &b) {
      return ComputeAngle<Line3d>(a, b) <= o.th_angle;
    });
    scorers_.emplace_back([o](const Line3d &a, const Line3d &b) {
      const double s =
          ExpScore(ComputeAngle<Line3d>(a, b), o.th_angle * o.Multiplier());
      return (s < o.score_th) ? 0.0 : s;
    });
  }

  // Overlap
  if (o.use_overlap) {
    checks_.emplace_back([o](const Line3d &a, const Line3d &b) {
      return ComputeBioverlap<Line3d>(a, b) > o.th_overlap;
    });
    scorers_.emplace_back([o](const Line3d &a, const Line3d &b) {
      return (ComputeBioverlap<Line3d>(a, b) > o.th_overlap) ? 1.0 : 0.0;
    });
  }

  // Smart angle
  if (o.use_angle && o.use_overlap && o.use_smartangle) {
    auto smart_angle_sigma = [o](const Line3d &a, const Line3d &b) {
      const double overlap = ComputeBioverlap<Line3d>(a, b);
      double th = o.th_angle;
      if (overlap < o.th_smartoverlap) {
        const double r = std::min((o.th_smartoverlap - overlap) /
                                      (o.th_smartoverlap - o.th_overlap),
                                  1.0);
        th = o.th_angle - r * (o.th_angle - o.th_smartangle);
      }
      return th * o.Multiplier();
    };
    checks_.emplace_back([o, smart_angle_sigma](const Line3d &a,
                                                const Line3d &b) {
      return ExpScore(ComputeAngle<Line3d>(a, b), smart_angle_sigma(a, b)) >=
             o.score_th;
    });
    scorers_.emplace_back(
        [o, smart_angle_sigma](const Line3d &a, const Line3d &b) {
          const double s =
              ExpScore(ComputeAngle<Line3d>(a, b), smart_angle_sigma(a, b));
          return (s < o.score_th) ? 0.0 : s;
        });
  }

  // Perpendicular (uncertainty-scaled)
  if (o.use_perp) {
    checks_.emplace_back([o](const Line3d &a, const Line3d &b) {
      THROW_CHECK_GT(a.uncertainty, 0);
      THROW_CHECK_GT(b.uncertainty, 0);
      const double d =
          ComputeDistance<Line3d>(a, b, LineMetricType::PERPENDICULAR);
      const double sigma =
          o.th_perp * std::min(a.uncertainty, b.uncertainty) * o.Multiplier();
      return ExpScore(d, sigma) >= o.score_th;
    });
    scorers_.emplace_back([o](const Line3d &a, const Line3d &b) {
      THROW_CHECK_GT(a.uncertainty, 0);
      THROW_CHECK_GT(b.uncertainty, 0);
      const double d =
          ComputeDistance<Line3d>(a, b, LineMetricType::PERPENDICULAR);
      const double sigma =
          o.th_perp * std::min(a.uncertainty, b.uncertainty) * o.Multiplier();
      const double s = ExpScore(d, sigma);
      return (s < o.score_th) ? 0.0 : s;
    });
  }

  // Inner segment (uncertainty-scaled)
  if (o.use_innerseg) {
    checks_.emplace_back([o](const Line3d &a, const Line3d &b) {
      THROW_CHECK_GT(a.uncertainty, 0);
      THROW_CHECK_GT(b.uncertainty, 0);
      const double d = ComputeDistance<Line3d>(a, b, LineMetricType::INNERSEG);
      const double sigma = o.th_innerseg *
                           std::min(a.uncertainty, b.uncertainty) *
                           o.Multiplier();
      return ExpScore(d, sigma) >= o.score_th;
    });
    scorers_.emplace_back([o](const Line3d &a, const Line3d &b) {
      THROW_CHECK_GT(a.uncertainty, 0);
      THROW_CHECK_GT(b.uncertainty, 0);
      const double d = ComputeDistance<Line3d>(a, b, LineMetricType::INNERSEG);
      const double sigma = o.th_innerseg *
                           std::min(a.uncertainty, b.uncertainty) *
                           o.Multiplier();
      const double s = ExpScore(d, sigma);
      return (s < o.score_th) ? 0.0 : s;
    });
  }
}

} // namespace limap
