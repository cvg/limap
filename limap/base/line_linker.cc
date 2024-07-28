#include "base/line_linker.h"
#include "base/line_dists.h"
#include <cmath>

namespace limap {

double expscore(const double &val, const double &sigma) {
  return exp(-pow(val / sigma, 2) / 2.0);
}

double get_multiplier(const double &score_th) {
  // exp(- (val / sigma)^2 / 2.0) >= 0.5 <--> val <= 1.1774100 sigma
  return 1.0 / sqrt(-log(score_th) * 2.0);
}

double LineLinker2d::compute_score_angle(const Line2d &l1,
                                         const Line2d &l2) const {
  double angle = compute_angle<Line2d>(l1, l2);
  double score = expscore(angle, config.th_angle * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

double LineLinker2d::compute_score_smartangle(const Line2d &l1,
                                              const Line2d &l2) const {
  double angle = compute_angle<Line2d>(l1, l2);
  double th_angle = config.th_angle;
  double overlap = compute_bioverlap<Line2d>(l1, l2);
  if (overlap < config.th_smartoverlap) {
    double ratio = (config.th_smartoverlap - overlap) /
                   (config.th_smartoverlap - config.th_overlap);
    ratio = std::min(ratio, 1.0);
    th_angle =
        config.th_angle - ratio * (config.th_angle - config.th_smartangle);
  }
  double score = expscore(angle, th_angle * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker2d::check_connection_angle(const Line2d &l1,
                                          const Line2d &l2) const {
  double angle = compute_angle<Line2d>(l1, l2);
  return angle <= config.th_angle;
}

bool LineLinker2d::check_connection_smartangle(const Line2d &l1,
                                               const Line2d &l2) const {
  return compute_score_smartangle(l1, l2) >= config.score_th;
}

double LineLinker2d::compute_score_overlap(const Line2d &l1,
                                           const Line2d &l2) const {
  double overlap = compute_bioverlap<Line2d>(l1, l2);
  if (overlap > config.th_overlap)
    return 1.0;
  else
    return 0.0;
}

bool LineLinker2d::check_connection_overlap(const Line2d &l1,
                                            const Line2d &l2) const {
  return compute_score_overlap(l1, l2) == 1.0;
}

double LineLinker2d::compute_score_perp(const Line2d &l1,
                                        const Line2d &l2) const {
  double dist = compute_distance<Line2d>(l1, l2, LineDistType::PERPENDICULAR);
  double score = expscore(dist, config.th_perp * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker2d::check_connection_perp(const Line2d &l1,
                                         const Line2d &l2) const {
  return compute_score_perp(l1, l2) >= config.score_th;
}

double LineLinker2d::compute_score_innerseg(const Line2d &l1,
                                            const Line2d &l2) const {
  double dist = compute_distance<Line2d>(l1, l2, LineDistType::INNERSEG);
  double score = expscore(dist, config.th_innerseg * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker2d::check_connection_innerseg(const Line2d &l1,
                                             const Line2d &l2) const {
  return compute_score_innerseg(l1, l2) >= config.score_th;
}

bool LineLinker2d::check_connection(const Line2d &l1, const Line2d &l2) const {
  if (config.use_angle)
    if (!check_connection_angle(l1, l2))
      return false;
  if (config.use_overlap)
    if (!check_connection_overlap(l1, l2))
      return false;
  if (config.use_angle && config.use_overlap && config.use_smartangle)
    if (!check_connection_smartangle(l1, l2))
      return false;
  if (config.use_perp)
    if (!check_connection_perp(l1, l2))
      return false;
  if (config.use_innerseg)
    if (!check_connection_innerseg(l1, l2))
      return false;
  return true;
}

double LineLinker2d::compute_score(const Line2d &l1, const Line2d &l2) const {
  double score = 1.0;
  if (config.use_angle)
    score = std::min(score, compute_score_angle(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_overlap)
    score = std::min(score, compute_score_overlap(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_angle && config.use_overlap && config.use_smartangle)
    score = std::min(score, compute_score_smartangle(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_perp)
    score = std::min(score, compute_score_perp(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_innerseg)
    score = std::min(score, compute_score_innerseg(l1, l2));
  return score;
}

double LineLinker3d::compute_score_angle(const Line3d &l1,
                                         const Line3d &l2) const {
  double angle = compute_angle<Line3d>(l1, l2);
  double score = expscore(angle, config.th_angle * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

double LineLinker3d::compute_score_smartangle(const Line3d &l1,
                                              const Line3d &l2) const {
  double angle = compute_angle<Line3d>(l1, l2);
  double th_angle = config.th_angle;
  double overlap = compute_bioverlap<Line3d>(l1, l2);
  if (overlap < config.th_smartoverlap) {
    double ratio = (config.th_smartoverlap - overlap) /
                   (config.th_smartoverlap - config.th_overlap);
    ratio = std::min(ratio, 1.0);
    th_angle =
        config.th_angle - ratio * (config.th_angle - config.th_smartangle);
  }
  double score = expscore(angle, th_angle * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker3d::check_connection_angle(const Line3d &l1,
                                          const Line3d &l2) const {
  double angle = compute_angle<Line3d>(l1, l2);
  return angle <= config.th_angle;
}

bool LineLinker3d::check_connection_smartangle(const Line3d &l1,
                                               const Line3d &l2) const {
  return compute_score_smartangle(l1, l2) >= config.score_th;
}

double LineLinker3d::compute_score_overlap(const Line3d &l1,
                                           const Line3d &l2) const {
  double overlap = compute_bioverlap<Line3d>(l1, l2);
  if (overlap > config.th_overlap)
    return 1.0;
  else
    return 0.0;
}

bool LineLinker3d::check_connection_overlap(const Line3d &l1,
                                            const Line3d &l2) const {
  return compute_score_overlap(l1, l2) == 1.0;
}

double LineLinker3d::compute_score_perp(const Line3d &l1,
                                        const Line3d &l2) const {
  double dist = compute_distance<Line3d>(l1, l2, LineDistType::PERPENDICULAR);
  double uncertainty = std::min(l1.uncertainty, l2.uncertainty);
  double score =
      expscore(dist, config.th_perp * uncertainty * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker3d::check_connection_perp(const Line3d &l1,
                                         const Line3d &l2) const {
  return compute_score_perp(l1, l2) >= config.score_th;
}

double LineLinker3d::compute_score_innerseg(const Line3d &l1,
                                            const Line3d &l2) const {
  double dist = compute_distance<Line3d>(l1, l2, LineDistType::INNERSEG);
  double uncertainty = std::min(l1.uncertainty, l2.uncertainty);
  double score =
      expscore(dist, config.th_innerseg * uncertainty * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker3d::check_connection_innerseg(const Line3d &l1,
                                             const Line3d &l2) const {
  return compute_score_innerseg(l1, l2) >= config.score_th;
}

double LineLinker3d::compute_score_scaleinv(const Line3d &l1,
                                            const Line3d &l2) const {
  double dist =
      compute_distance<Line3d>(l1, l2, LineDistType::ENDPOINTS_SCALEINV_ONEWAY);
  double score = expscore(dist, config.th_scaleinv * config.multiplier());
  if (score < config.score_th)
    score = 0.0;
  return score;
}

bool LineLinker3d::check_connection_scaleinv(const Line3d &l1,
                                             const Line3d &l2) const {
  return compute_score_scaleinv(l1, l2) >= config.score_th;
}

bool LineLinker3d::check_connection(const Line3d &l1, const Line3d &l2) const {
  if (config.use_angle)
    if (!check_connection_angle(l1, l2))
      return false;
  if (config.use_overlap)
    if (!check_connection_overlap(l1, l2))
      return false;
  if (config.use_angle && config.use_overlap && config.use_smartangle)
    if (!check_connection_smartangle(l1, l2))
      return false;
  if (config.use_perp)
    if (!check_connection_perp(l1, l2))
      return false;
  if (config.use_innerseg)
    if (!check_connection_innerseg(l1, l2))
      return false;
  if (config.use_scaleinv)
    if (!check_connection_scaleinv(l1, l2))
      return false;
  return true;
}

double LineLinker3d::compute_score(const Line3d &l1, const Line3d &l2) const {
  double score = 1.0;
  if (config.use_angle)
    score = std::min(score, compute_score_angle(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_overlap)
    score = std::min(score, compute_score_overlap(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_angle && config.use_overlap && config.use_smartangle)
    score = std::min(score, compute_score_smartangle(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_perp)
    score = std::min(score, compute_score_perp(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_innerseg)
    score = std::min(score, compute_score_innerseg(l1, l2));
  if (score < config.score_th)
    return score;
  if (config.use_scaleinv)
    score = std::min(score, compute_score_scaleinv(l1, l2));
  return score;
}

} // namespace limap
