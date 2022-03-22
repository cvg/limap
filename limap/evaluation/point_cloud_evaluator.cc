#include "evaluation/point_cloud_evaluator.h"

#include <queue>
#include <iostream>
#include <numeric>

namespace limap {

namespace evaluation {

double PointCloudEvaluator::ComputeDistPoint(const V3D& point) {
    V3D p = tree_.query_nearest(point);
    return (point - p).norm();
}

} // namespace evaluation 

} // namespace limap

