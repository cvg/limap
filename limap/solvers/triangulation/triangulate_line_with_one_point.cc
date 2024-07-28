#include "solvers/triangulation/triangulate_line_with_one_point.h"
#include <PoseLib/misc/univariate.h>
#include <iostream>

namespace limap {

namespace solvers {

namespace triangulation {

std::pair<double, double> triangulate_line_with_one_point(
    const Eigen::Vector4d &plane, const Eigen::Vector2d &p,
    const Eigen::Vector2d &v1, const Eigen::Vector2d &v2) {
  const double lx = plane[0];
  const double ly = plane[1];
  const double lz = plane[3];
  const double px = p[0];
  const double py = p[1];
  const double p1x = v1[0];
  const double p1y = v1[1];
  const double p2x = v2[0];
  const double p2y = v2[1];

  const double c4 =
      std::pow(p1x, 3) * p1y * std::pow(p2y, 4) * std::pow(px, 2) -
      std::pow(p2x, 3) * std::pow(p1y, 4) * p2y * std::pow(px, 2) -
      p1x * std::pow(p2x, 4) * std::pow(p1y, 3) * std::pow(py, 2) +
      std::pow(p1x, 4) * p2x * std::pow(p2y, 3) * std::pow(py, 2) -
      std::pow(p1x, 4) * std::pow(p2y, 4) * px * py +
      std::pow(p2x, 4) * std::pow(p1y, 4) * px * py +
      3 * p1x * std::pow(p2x, 2) * std::pow(p1y, 3) * std::pow(p2y, 2) *
          std::pow(px, 2) -
      3 * std::pow(p1x, 2) * p2x * std::pow(p1y, 2) * std::pow(p2y, 3) *
          std::pow(px, 2) +
      3 * std::pow(p1x, 2) * std::pow(p2x, 3) * std::pow(p1y, 2) * p2y *
          std::pow(py, 2) -
      3 * std::pow(p1x, 3) * std::pow(p2x, 2) * p1y * std::pow(p2y, 2) *
          std::pow(py, 2) -
      2 * p1x * std::pow(p2x, 3) * std::pow(p1y, 3) * p2y * px * py +
      2 * std::pow(p1x, 3) * p2x * p1y * std::pow(p2y, 3) * px * py;
  const double c3 = 0.0;
  const double c2 =
      4 * std::pow(lx, 2) * std::pow(lz, 2) * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) -
      4 * std::pow(lx, 2) * std::pow(lz, 2) * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) +
      4 * std::pow(ly, 2) * std::pow(lz, 2) * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) -
      4 * std::pow(ly, 2) * std::pow(lz, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y +
      12 * std::pow(lx, 4) * std::pow(p1x, 3) * std::pow(p2x, 4) * p1y *
          std::pow(py, 2) -
      12 * std::pow(lx, 4) * std::pow(p1x, 4) * std::pow(p2x, 3) * p2y *
          std::pow(py, 2) -
      12 * std::pow(ly, 4) * p1x * std::pow(p1y, 3) * std::pow(p2y, 4) *
          std::pow(px, 2) +
      12 * std::pow(ly, 4) * p2x * std::pow(p1y, 4) * std::pow(p2y, 3) *
          std::pow(px, 2) +
      4 * lx * ly * std::pow(lz, 2) * std::pow(p1x, 4) * std::pow(p2y, 4) -
      4 * lx * ly * std::pow(lz, 2) * std::pow(p2x, 4) * std::pow(p1y, 4) +
      4 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 4) * std::pow(p2y, 4) * px -
      4 * std::pow(lx, 2) * ly * lz * std::pow(p2x, 4) * std::pow(p1y, 4) * px +
      4 * lx * std::pow(ly, 2) * lz * std::pow(p1x, 4) * std::pow(p2y, 4) * py -
      4 * lx * std::pow(ly, 2) * lz * std::pow(p2x, 4) * std::pow(p1y, 4) * py -
      4 * std::pow(lx, 3) * lz * p1x * std::pow(p2x, 4) * std::pow(p1y, 3) *
          px +
      4 * std::pow(lx, 3) * lz * std::pow(p1x, 4) * p2x * std::pow(p2y, 3) *
          px +
      4 * std::pow(ly, 3) * lz * std::pow(p1x, 3) * p1y * std::pow(p2y, 4) *
          py -
      4 * std::pow(ly, 3) * lz * std::pow(p2x, 3) * std::pow(p1y, 4) * p2y *
          py +
      12 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 4) *
          std::pow(p2y, 4) * px * py -
      12 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 4) * px * py -
      12 * std::pow(lx, 4) * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * px * py +
      12 * std::pow(lx, 4) * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * px * py +
      12 * std::pow(ly, 4) * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * px * py -
      12 * std::pow(ly, 4) * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * px * py +
      12 * std::pow(lx, 2) * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y -
      12 * std::pow(lx, 2) * std::pow(lz, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) +
      12 * std::pow(ly, 2) * std::pow(lz, 2) * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) -
      12 * std::pow(ly, 2) * std::pow(lz, 2) * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) -
      24 * lx * std::pow(ly, 3) * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * std::pow(px, 2) -
      12 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * std::pow(px, 2) +
      24 * lx * std::pow(ly, 3) * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * std::pow(px, 2) +
      12 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * std::pow(px, 2) +
      12 * std::pow(lx, 2) * std::pow(ly, 2) * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * std::pow(py, 2) +
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * std::pow(py, 2) -
      12 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * std::pow(py, 2) -
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * std::pow(py, 2) +
      12 * std::pow(lx, 4) * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * std::pow(px, 2) -
      12 * std::pow(lx, 4) * std::pow(p1x, 3) * std::pow(p2x, 2) * p1y *
          std::pow(p2y, 2) * std::pow(px, 2) +
      12 * std::pow(ly, 4) * p1x * std::pow(p2x, 2) * std::pow(p1y, 3) *
          std::pow(p2y, 2) * std::pow(py, 2) -
      12 * std::pow(ly, 4) * std::pow(p1x, 2) * p2x * std::pow(p1y, 2) *
          std::pow(p2y, 3) * std::pow(py, 2) +
      8 * lx * ly * std::pow(lz, 2) * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y -
      8 * lx * ly * std::pow(lz, 2) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) +
      4 * lx * std::pow(ly, 2) * lz * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * px -
      4 * lx * std::pow(ly, 2) * lz * std::pow(p2x, 3) * std::pow(p1y, 4) *
          p2y * px -
      4 * std::pow(lx, 2) * ly * lz * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * py +
      4 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * py -
      24 * std::pow(lx, 3) * ly * p1x * std::pow(p2x, 4) * std::pow(p1y, 3) *
          px * py +
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 4) * p2x * std::pow(p2y, 3) *
          px * py +
      24 * lx * std::pow(ly, 3) * std::pow(p1x, 3) * p1y * std::pow(p2y, 4) *
          px * py -
      24 * lx * std::pow(ly, 3) * std::pow(p2x, 3) * std::pow(p1y, 4) * p2y *
          px * py +
      36 * std::pow(lx, 2) * std::pow(ly, 2) * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * std::pow(px, 2) -
      36 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * std::pow(px, 2) +
      36 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * std::pow(py, 2) -
      36 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * std::pow(py, 2) +
      24 * std::pow(lx, 3) * ly * p1x * std::pow(p2x, 3) * std::pow(p1y, 3) *
          p2y * std::pow(px, 2) -
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * std::pow(px, 2) +
      12 * std::pow(lx, 3) * lz * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * px -
      12 * std::pow(lx, 3) * lz * std::pow(p1x, 3) * std::pow(p2x, 2) * p1y *
          std::pow(p2y, 2) * px +
      24 * lx * std::pow(ly, 3) * p1x * std::pow(p2x, 3) * std::pow(p1y, 3) *
          p2y * std::pow(py, 2) -
      24 * lx * std::pow(ly, 3) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * std::pow(py, 2) +
      12 * std::pow(ly, 3) * lz * p1x * std::pow(p2x, 2) * std::pow(p1y, 3) *
          std::pow(p2y, 2) * py -
      12 * std::pow(ly, 3) * lz * std::pow(p1x, 2) * p2x * std::pow(p1y, 2) *
          std::pow(p2y, 3) * py +
      12 * lx * std::pow(ly, 2) * lz * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * px -
      12 * lx * std::pow(ly, 2) * lz * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * px +
      12 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * py -
      12 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 3) * std::pow(p2x, 2) *
          p1y * std::pow(p2y, 2) * py -
      24 * lx * std::pow(ly, 3) * p1x * std::pow(p2x, 2) * std::pow(p1y, 3) *
          std::pow(p2y, 2) * px * py +
      24 * lx * std::pow(ly, 3) * std::pow(p1x, 2) * p2x * std::pow(p1y, 2) *
          std::pow(p2y, 3) * px * py -
      48 * std::pow(lx, 2) * std::pow(ly, 2) * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * px * py +
      48 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * px * py -
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * px * py +
      24 * std::pow(lx, 3) * ly * std::pow(p1x, 3) * std::pow(p2x, 2) * p1y *
          std::pow(p2y, 2) * px * py +
      8 * std::pow(lx, 2) * ly * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * px -
      8 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * px +
      8 * lx * std::pow(ly, 2) * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * py -
      8 * lx * std::pow(ly, 2) * lz * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * py;
  const double c1 =
      16 * std::pow(lx, 6) * std::pow(p1x, 3) * std::pow(p2x, 4) * p1y * px *
          py -
      16 * std::pow(ly, 6) * std::pow(p1y, 4) * std::pow(p2y, 4) *
          std::pow(px, 2) -
      16 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 4) * std::pow(p2y, 4) -
      16 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 4) -
      16 * std::pow(lx, 4) * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 2) -
      16 * std::pow(lx, 4) * std::pow(lz, 2) * std::pow(p1x, 4) *
          std::pow(p2x, 2) * std::pow(p2y, 2) -
      16 * std::pow(ly, 4) * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p1y, 2) * std::pow(p2y, 4) -
      16 * std::pow(ly, 4) * std::pow(lz, 2) * std::pow(p2x, 2) *
          std::pow(p1y, 4) * std::pow(p2y, 2) -
      8 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 4) *
          std::pow(p2y, 4) * std::pow(px, 2) -
      8 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 4) * std::pow(px, 2) -
      8 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 4) *
          std::pow(p2y, 4) * std::pow(py, 2) -
      8 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p2x, 4) *
          std::pow(p1y, 4) * std::pow(py, 2) -
      8 * std::pow(lx, 6) * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * std::pow(px, 2) -
      8 * std::pow(lx, 6) * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * std::pow(px, 2) -
      8 * std::pow(ly, 6) * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * std::pow(py, 2) -
      8 * std::pow(ly, 6) * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * std::pow(py, 2) -
      16 * std::pow(lx, 6) * std::pow(p1x, 4) * std::pow(p2x, 4) *
          std::pow(py, 2) +
      16 * std::pow(lx, 6) * std::pow(p1x, 4) * std::pow(p2x, 3) * p2y * px *
          py +
      16 * std::pow(ly, 6) * p1x * std::pow(p1y, 3) * std::pow(p2y, 4) * px *
          py +
      16 * std::pow(ly, 6) * p2x * std::pow(p1y, 4) * std::pow(p2y, 3) * px *
          py -
      56 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 2) *
          std::pow(p1y, 2) * std::pow(p2y, 4) * std::pow(px, 2) -
      56 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p2x, 2) *
          std::pow(p1y, 4) * std::pow(p2y, 2) * std::pow(px, 2) -
      56 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 2) * std::pow(py, 2) -
      56 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 4) *
          std::pow(p2x, 2) * std::pow(p2y, 2) * std::pow(py, 2) -
      32 * std::pow(lx, 3) * ly * std::pow(lz, 2) * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) -
      32 * std::pow(lx, 3) * ly * std::pow(lz, 2) * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) -
      32 * lx * std::pow(ly, 3) * std::pow(lz, 2) * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) -
      32 * lx * std::pow(ly, 3) * std::pow(lz, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y -
      16 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 4) *
          std::pow(p2y, 4) * px -
      16 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p2x, 4) *
          std::pow(p1y, 4) * px -
      16 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p1x, 4) *
          std::pow(p2y, 4) * py -
      16 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p2x, 4) *
          std::pow(p1y, 4) * py +
      32 * std::pow(lx, 4) * std::pow(lz, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 3) * p1y * p2y +
      32 * std::pow(ly, 4) * std::pow(lz, 2) * p1x * p2x * std::pow(p1y, 3) *
          std::pow(p2y, 3) -
      16 * std::pow(lx, 5) * ly * p1x * std::pow(p2x, 4) * std::pow(p1y, 3) *
          std::pow(px, 2) -
      16 * std::pow(lx, 5) * ly * std::pow(p1x, 4) * p2x * std::pow(p2y, 3) *
          std::pow(px, 2) -
      48 * lx * std::pow(ly, 5) * p1x * std::pow(p1y, 3) * std::pow(p2y, 4) *
          std::pow(px, 2) -
      48 * lx * std::pow(ly, 5) * p2x * std::pow(p1y, 4) * std::pow(p2y, 3) *
          std::pow(px, 2) -
      16 * std::pow(lx, 5) * lz * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * px -
      48 * std::pow(lx, 5) * ly * std::pow(p1x, 3) * std::pow(p2x, 4) * p1y *
          std::pow(py, 2) -
      16 * std::pow(lx, 5) * lz * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * px -
      48 * std::pow(lx, 5) * ly * std::pow(p1x, 4) * std::pow(p2x, 3) * p2y *
          std::pow(py, 2) -
      16 * lx * std::pow(ly, 5) * std::pow(p1x, 3) * p1y * std::pow(p2y, 4) *
          std::pow(py, 2) -
      16 * lx * std::pow(ly, 5) * std::pow(p2x, 3) * std::pow(p1y, 4) * p2y *
          std::pow(py, 2) -
      16 * std::pow(ly, 5) * lz * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * py -
      16 * std::pow(ly, 5) * lz * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * py -
      32 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * std::pow(px, 2) -
      32 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * std::pow(px, 2) -
      32 * std::pow(lx, 3) * std::pow(ly, 3) * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * std::pow(py, 2) -
      32 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * std::pow(py, 2) +
      96 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 2) * std::pow(p2x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 2) -
      96 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 2) * std::pow(p1y, 2) * std::pow(p2y, 2) *
          std::pow(px, 2) -
      96 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 2) *
          std::pow(p2x, 2) * std::pow(p1y, 2) * std::pow(p2y, 2) *
          std::pow(py, 2) +
      32 * lx * std::pow(ly, 3) * std::pow(lz, 2) * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) +
      32 * lx * std::pow(ly, 3) * std::pow(lz, 2) * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) -
      32 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(lz, 2) * p1x *
          std::pow(p2x, 3) * std::pow(p1y, 3) * p2y -
      32 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 3) * p2x * p1y * std::pow(p2y, 3) +
      32 * std::pow(lx, 3) * ly * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y +
      32 * std::pow(lx, 3) * ly * std::pow(lz, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) -
      128 * std::pow(lx, 2) * std::pow(ly, 4) * p1x * p2x * std::pow(p1y, 3) *
          std::pow(p2y, 3) * std::pow(px, 2) -
      64 * std::pow(lx, 4) * std::pow(ly, 2) * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * std::pow(px, 2) -
      64 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * std::pow(px, 2) -
      32 * std::pow(lx, 5) * ly * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * std::pow(px, 2) -
      32 * std::pow(lx, 5) * ly * std::pow(p1x, 3) * std::pow(p2x, 2) * p1y *
          std::pow(p2y, 2) * std::pow(px, 2) -
      32 * lx * std::pow(ly, 5) * p1x * std::pow(p2x, 2) * std::pow(p1y, 3) *
          std::pow(p2y, 2) * std::pow(py, 2) -
      32 * lx * std::pow(ly, 5) * std::pow(p1x, 2) * p2x * std::pow(p1y, 2) *
          std::pow(p2y, 3) * std::pow(py, 2) -
      64 * std::pow(lx, 2) * std::pow(ly, 4) * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * std::pow(py, 2) -
      64 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * std::pow(py, 2) -
      128 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 3) * p1y * p2y * std::pow(py, 2) -
      32 * std::pow(lx, 4) * ly * lz * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * px -
      32 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * px -
      32 * lx * std::pow(ly, 4) * lz * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * py -
      32 * lx * std::pow(ly, 4) * lz * std::pow(p2x, 3) * std::pow(p1y, 4) *
          p2y * py +
      32 * std::pow(lx, 5) * lz * std::pow(p1x, 3) * std::pow(p2x, 3) * p1y *
          p2y * px +
      32 * std::pow(ly, 5) * lz * p1x * p2x * std::pow(p1y, 3) *
          std::pow(p2y, 3) * py -
      128 * std::pow(lx, 3) * std::pow(ly, 3) * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * std::pow(px, 2) -
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * std::pow(px, 2) -
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * std::pow(py, 2) -
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * std::pow(py, 2) -
      16 * lx * std::pow(ly, 4) * lz * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * px -
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * px -
      16 * lx * std::pow(ly, 4) * lz * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * px -
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * px -
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * py -
      16 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * py -
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * py -
      16 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * py +
      16 * std::pow(lx, 4) * std::pow(ly, 2) * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * px * py +
      32 * std::pow(lx, 5) * ly * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * px * py +
      16 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * px * py +
      32 * std::pow(lx, 5) * ly * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * px * py +
      32 * lx * std::pow(ly, 5) * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * px * py +
      16 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * px * py +
      32 * lx * std::pow(ly, 5) * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * px * py +
      16 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * px * py +
      96 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 2) * std::pow(p1y, 2) * std::pow(p2y, 2) * px +
      96 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 2) * std::pow(p1y, 2) * std::pow(p2y, 2) * py +
      384 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 2) *
          std::pow(p2x, 2) * std::pow(p1y, 2) * std::pow(p2y, 2) * px * py -
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * px -
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 3) * p2x *
          p1y * std::pow(p2y, 3) * px +
      32 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 2) * std::pow(p2x, 3) *
          std::pow(p1y, 2) * p2y * px +
      32 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 3) * std::pow(p2x, 2) *
          p1y * std::pow(p2y, 2) * px +
      32 * lx * std::pow(ly, 4) * lz * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * py +
      32 * lx * std::pow(ly, 4) * lz * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * py -
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * py -
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p1x, 3) * p2x *
          p1y * std::pow(p2y, 3) * py +
      128 * std::pow(lx, 3) * std::pow(ly, 3) * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * px * py +
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(p1x, 3) * p2x * p1y *
          std::pow(p2y, 3) * px * py +
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * px +
      32 * std::pow(lx, 2) * std::pow(ly, 3) * lz * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * px +
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * py +
      32 * std::pow(lx, 3) * std::pow(ly, 2) * lz * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * py +
      224 * std::pow(lx, 2) * std::pow(ly, 4) * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * px * py +
      224 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * px * py +
      224 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * px * py +
      224 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * px * py +
      32 * lx * std::pow(ly, 4) * lz * p1x * p2x * std::pow(p1y, 3) *
          std::pow(p2y, 3) * px +
      32 * std::pow(lx, 4) * ly * lz * std::pow(p1x, 3) * std::pow(p2x, 3) *
          p1y * p2y * py +
      128 * lx * std::pow(ly, 5) * p1x * p2x * std::pow(p1y, 3) *
          std::pow(p2y, 3) * px * py +
      128 * std::pow(lx, 5) * ly * std::pow(p1x, 3) * std::pow(p2x, 3) * p1y *
          p2y * px * py;
  const double c0 =
      16 * std::pow(lx, 6) * std::pow(lz, 2) * std::pow(p1x, 4) *
          std::pow(p2x, 3) * p2y -
      16 * std::pow(lx, 6) * std::pow(lz, 2) * std::pow(p1x, 3) *
          std::pow(p2x, 4) * p1y +
      16 * std::pow(ly, 6) * std::pow(lz, 2) * p1x * std::pow(p1y, 3) *
          std::pow(p2y, 4) -
      16 * std::pow(ly, 6) * std::pow(lz, 2) * p2x * std::pow(p1y, 4) *
          std::pow(p2y, 3) +
      16 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(lz, 2) *
          std::pow(p1x, 4) * std::pow(p2y, 4) -
      16 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(lz, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 4) -
      16 * std::pow(lx, 7) * lz * std::pow(p1x, 3) * std::pow(p2x, 4) * p1y *
          px +
      16 * std::pow(lx, 7) * lz * std::pow(p1x, 4) * std::pow(p2x, 3) * p2y *
          px +
      16 * std::pow(ly, 7) * lz * p1x * std::pow(p1y, 3) * std::pow(p2y, 4) *
          py -
      16 * std::pow(ly, 7) * lz * p2x * std::pow(p1y, 4) * std::pow(p2y, 3) *
          py +
      16 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p1x, 4) *
          std::pow(p2y, 4) * px -
      16 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p2x, 4) *
          std::pow(p1y, 4) * px +
      16 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p1x, 4) *
          std::pow(p2y, 4) * py -
      16 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p2x, 4) *
          std::pow(p1y, 4) * py -
      48 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(lz, 2) * p1x *
          std::pow(p2x, 4) * std::pow(p1y, 3) -
      48 * std::pow(lx, 5) * ly * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 2) +
      48 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 4) * p2x * std::pow(p2y, 3) +
      48 * std::pow(lx, 5) * ly * std::pow(lz, 2) * std::pow(p1x, 4) *
          std::pow(p2x, 2) * std::pow(p2y, 2) +
      48 * lx * std::pow(ly, 5) * std::pow(lz, 2) * std::pow(p1x, 2) *
          std::pow(p1y, 2) * std::pow(p2y, 4) +
      48 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(lz, 2) *
          std::pow(p1x, 3) * p1y * std::pow(p2y, 4) -
      48 * lx * std::pow(ly, 5) * std::pow(lz, 2) * std::pow(p2x, 2) *
          std::pow(p1y, 4) * std::pow(p2y, 2) -
      48 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(lz, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 4) * p2y -
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(lz, 2) * p1x *
          std::pow(p2x, 3) * std::pow(p1y, 3) * p2y +
      128 * std::pow(lx, 3) * std::pow(ly, 3) * std::pow(lz, 2) *
          std::pow(p1x, 3) * p2x * p1y * std::pow(p2y, 3) +
      48 * std::pow(lx, 2) * std::pow(ly, 5) * lz * std::pow(p1x, 2) *
          std::pow(p1y, 2) * std::pow(p2y, 4) * px -
      48 * std::pow(lx, 2) * std::pow(ly, 5) * lz * std::pow(p2x, 2) *
          std::pow(p1y, 4) * std::pow(p2y, 2) * px -
      48 * std::pow(lx, 5) * std::pow(ly, 2) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 4) * std::pow(p1y, 2) * py +
      48 * std::pow(lx, 5) * std::pow(ly, 2) * lz * std::pow(p1x, 4) *
          std::pow(p2x, 2) * std::pow(p2y, 2) * py +
      16 * lx * std::pow(ly, 6) * lz * p1x * std::pow(p1y, 3) *
          std::pow(p2y, 4) * px -
      16 * lx * std::pow(ly, 6) * lz * p2x * std::pow(p1y, 4) *
          std::pow(p2y, 3) * px -
      16 * std::pow(lx, 6) * ly * lz * std::pow(p1x, 3) * std::pow(p2x, 4) *
          p1y * py +
      16 * std::pow(lx, 6) * ly * lz * std::pow(p1x, 4) * std::pow(p2x, 3) *
          p2y * py -
      96 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(lz, 2) * p1x *
          std::pow(p2x, 2) * std::pow(p1y, 3) * std::pow(p2y, 2) +
      96 * std::pow(lx, 2) * std::pow(ly, 4) * std::pow(lz, 2) *
          std::pow(p1x, 2) * p2x * std::pow(p1y, 2) * std::pow(p2y, 3) -
      96 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 2) * std::pow(p2x, 3) * std::pow(p1y, 2) * p2y +
      96 * std::pow(lx, 4) * std::pow(ly, 2) * std::pow(lz, 2) *
          std::pow(p1x, 3) * std::pow(p2x, 2) * p1y * std::pow(p2y, 2) -
      48 * std::pow(lx, 5) * std::pow(ly, 2) * lz * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * px -
      48 * std::pow(lx, 6) * ly * lz * std::pow(p1x, 2) * std::pow(p2x, 4) *
          std::pow(p1y, 2) * px +
      48 * std::pow(lx, 5) * std::pow(ly, 2) * lz * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * px +
      48 * std::pow(lx, 6) * ly * lz * std::pow(p1x, 4) * std::pow(p2x, 2) *
          std::pow(p2y, 2) * px +
      48 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * px -
      48 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * px -
      48 * std::pow(lx, 4) * std::pow(ly, 3) * lz * p1x * std::pow(p2x, 4) *
          std::pow(p1y, 3) * py +
      48 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p1x, 4) * p2x *
          std::pow(p2y, 3) * py +
      48 * lx * std::pow(ly, 6) * lz * std::pow(p1x, 2) * std::pow(p1y, 2) *
          std::pow(p2y, 4) * py +
      48 * std::pow(lx, 2) * std::pow(ly, 5) * lz * std::pow(p1x, 3) * p1y *
          std::pow(p2y, 4) * py -
      48 * lx * std::pow(ly, 6) * lz * std::pow(p2x, 2) * std::pow(p1y, 4) *
          std::pow(p2y, 2) * py -
      48 * std::pow(lx, 2) * std::pow(ly, 5) * lz * std::pow(p2x, 3) *
          std::pow(p1y, 4) * p2y * py -
      128 * std::pow(lx, 4) * std::pow(ly, 3) * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * px +
      128 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p1x, 3) * p2x *
          p1y * std::pow(p2y, 3) * px -
      128 * std::pow(lx, 3) * std::pow(ly, 4) * lz * p1x * std::pow(p2x, 3) *
          std::pow(p1y, 3) * p2y * py +
      128 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p1x, 3) * p2x *
          p1y * std::pow(p2y, 3) * py -
      96 * std::pow(lx, 3) * std::pow(ly, 4) * lz * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * px +
      96 * std::pow(lx, 3) * std::pow(ly, 4) * lz * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * px -
      96 * std::pow(lx, 5) * std::pow(ly, 2) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * px +
      96 * std::pow(lx, 5) * std::pow(ly, 2) * lz * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * px -
      96 * std::pow(lx, 2) * std::pow(ly, 5) * lz * p1x * std::pow(p2x, 2) *
          std::pow(p1y, 3) * std::pow(p2y, 2) * py +
      96 * std::pow(lx, 2) * std::pow(ly, 5) * lz * std::pow(p1x, 2) * p2x *
          std::pow(p1y, 2) * std::pow(p2y, 3) * py -
      96 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p1x, 2) *
          std::pow(p2x, 3) * std::pow(p1y, 2) * p2y * py +
      96 * std::pow(lx, 4) * std::pow(ly, 3) * lz * std::pow(p1x, 3) *
          std::pow(p2x, 2) * p1y * std::pow(p2y, 2) * py;

  // Solve the quartic (note that the degree 3 term has zero coefficient)
  double mu_sols[4];
  int sols = poselib::univariate::solve_quartic_real(c3 / c4, c2 / c4, c1 / c4,
                                                     c0 / c4, mu_sols);

  std::pair<double, double> best_solution =
      std::make_pair<double, double>(-1, -1);
  double best_err = std::numeric_limits<double>::max();
  for (int k = 0; k < sols; ++k) {
    const double mu = mu_sols[k];
    // for each mu compute
    const double lambda_denom =
        4 * std::pow(lx, 4) * std::pow(p1x, 2) * std::pow(p2x, 2) +
        4 * std::pow(ly, 4) * std::pow(p1y, 2) * std::pow(p2y, 2) -
        std::pow(mu, 2) * std::pow(p1x, 2) * std::pow(p2y, 2) -
        std::pow(mu, 2) * std::pow(p2x, 2) * std::pow(p1y, 2) +
        4 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p1x, 2) *
            std::pow(p2y, 2) +
        4 * std::pow(lx, 2) * std::pow(ly, 2) * std::pow(p2x, 2) *
            std::pow(p1y, 2) +
        8 * std::pow(lx, 3) * ly * p1x * std::pow(p2x, 2) * p1y +
        8 * std::pow(lx, 3) * ly * std::pow(p1x, 2) * p2x * p2y +
        8 * lx * std::pow(ly, 3) * p1x * p1y * std::pow(p2y, 2) +
        8 * lx * std::pow(ly, 3) * p2x * std::pow(p1y, 2) * p2y +
        2 * std::pow(mu, 2) * p1x * p2x * p1y * p2y +
        16 * std::pow(lx, 2) * std::pow(ly, 2) * p1x * p2x * p1y * p2y;
    const double lambda_1 =
        (2 * std::pow(lx, 2) * mu * p1x * std::pow(p2x, 2) * py -
         4 * std::pow(ly, 3) * lz * p1y * std::pow(p2y, 2) -
         std::pow(mu, 2) * p1x * std::pow(p2y, 2) * px -
         std::pow(mu, 2) * std::pow(p2x, 2) * p1y * py -
         4 * lx * std::pow(ly, 2) * lz * p1x * std::pow(p2y, 2) -
         4 * std::pow(lx, 2) * ly * lz * std::pow(p2x, 2) * p1y -
         2 * std::pow(lx, 2) * mu * std::pow(p2x, 2) * p1y * px -
         4 * std::pow(lx, 3) * lz * p1x * std::pow(p2x, 2) -
         2 * std::pow(ly, 2) * mu * p1y * std::pow(p2y, 2) * px +
         2 * std::pow(ly, 2) * mu * p1x * std::pow(p2y, 2) * py -
         2 * lx * lz * mu * std::pow(p2x, 2) * p1y +
         2 * ly * lz * mu * p1x * std::pow(p2y, 2) +
         std::pow(mu, 2) * p2x * p1y * p2y * px +
         std::pow(mu, 2) * p1x * p2x * p2y * py -
         8 * std::pow(lx, 2) * ly * lz * p1x * p2x * p2y -
         8 * lx * std::pow(ly, 2) * lz * p2x * p1y * p2y +
         2 * lx * lz * mu * p1x * p2x * p2y -
         2 * ly * lz * mu * p2x * p1y * p2y -
         4 * lx * ly * mu * p2x * p1y * p2y * px +
         4 * lx * ly * mu * p1x * p2x * p2y * py) /
        lambda_denom;
    const double lambda_2 =
        (2 * std::pow(lx, 2) * mu * std::pow(p1x, 2) * p2y * px -
         4 * std::pow(ly, 3) * lz * std::pow(p1y, 2) * p2y -
         std::pow(mu, 2) * p2x * std::pow(p1y, 2) * px -
         std::pow(mu, 2) * std::pow(p1x, 2) * p2y * py -
         4 * lx * std::pow(ly, 2) * lz * p2x * std::pow(p1y, 2) -
         4 * std::pow(lx, 2) * ly * lz * std::pow(p1x, 2) * p2y -
         4 * std::pow(lx, 3) * lz * std::pow(p1x, 2) * p2x -
         2 * std::pow(lx, 2) * mu * std::pow(p1x, 2) * p2x * py +
         2 * std::pow(ly, 2) * mu * std::pow(p1y, 2) * p2y * px -
         2 * std::pow(ly, 2) * mu * p2x * std::pow(p1y, 2) * py +
         2 * lx * lz * mu * std::pow(p1x, 2) * p2y -
         2 * ly * lz * mu * p2x * std::pow(p1y, 2) +
         std::pow(mu, 2) * p1x * p1y * p2y * px +
         std::pow(mu, 2) * p1x * p2x * p1y * py -
         8 * std::pow(lx, 2) * ly * lz * p1x * p2x * p1y -
         8 * lx * std::pow(ly, 2) * lz * p1x * p1y * p2y -
         2 * lx * lz * mu * p1x * p2x * p1y +
         2 * ly * lz * mu * p1x * p1y * p2y +
         4 * lx * ly * mu * p1x * p1y * p2y * px -
         4 * lx * ly * mu * p1x * p2x * p1y * py) /
        lambda_denom;

    // cheirality check
    if (lambda_1 <= 0 || lambda_2 <= 0)
      continue;

    // solutions are now
    // x1 = lambda_1*[p1x; p1y]  and   x2 = lambda_2*[p2x;p2y]
    // These two points are co-linear with [px;py]
    // Compute
    const double err1 = lx * (lambda_1 * p1x) + ly * (lambda_1 * p1y) + lz;
    const double err2 = lx * (lambda_2 * p2x) + ly * (lambda_2 * p2y) + lz;
    const double err = err1 * err1 + err2 * err2;

    if (err < best_err) {
      best_err = err;
      best_solution.first = lambda_1;
      best_solution.second = lambda_2;
    }
  }
  return best_solution;
}

} // namespace triangulation

} // namespace solvers

} // namespace limap
