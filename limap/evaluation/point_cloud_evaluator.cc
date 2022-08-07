#include "evaluation/point_cloud_evaluator.h"

#include <third-party/progressbar.hpp>
#include <queue>
#include <iostream>
#include <numeric>

namespace limap {

namespace evaluation {

double PointCloudEvaluator::ComputeDistPoint(const V3D& point) {
    V3D p = tree_.query_nearest(point);
    return (point - p).norm();
}

std::vector<double> PointCloudEvaluator::ComputeDistsforEachPoint(const std::vector<Line3d>& lines) const {
    size_t n_points = tree_.cloud.pts.size();
    std::vector<double> dists(n_points);
    progressbar bar(n_points);
#pragma omp parallel for
    for (size_t i = 0; i < n_points; ++i) {
        bar.update();
        V3D p = tree_.point(i);
        double min_dist = std::numeric_limits<double>::max();
        for (auto it = lines.begin(); it != lines.end(); ++it) {
            double dist = it->point_distance(p);
            if (dist < min_dist)
                min_dist = dist;
        }
        dists[i] = min_dist;
    }
    return dists;
}

} // namespace evaluation 

} // namespace limap

