#include "base/line2d_parameterization.h"
#include <cmath>

namespace limap {

Line2d Line2d_2DOF::GetLine_fixedprojection() const {
    V2D dir_perp = line.perp_direction();
    Line2d newline;
    newline.start = line.start + dir_perp * vars[0];
    newline.end = line.end + dir_perp * vars[1];
    return newline;
}

Line2d Line2d_2DOF::GetLine_fixedlength() const {
    M2D R;
    R(0, 0) = cos(vars[0]); R(0, 1) = -sin(vars[0]);
    R(1, 0) = sin(vars[0]); R(1, 1) = cos(vars[0]);
    V2D new_direc = R * line.direction();
    V2D new_midpoint = line.midpoint() + line.perp_direction() * vars[1];

    double length = line.length();
    Line2d newline;
    newline.start = new_midpoint - new_direc * length / 2.0;
    newline.end = new_midpoint + new_direc * length / 2.0;
    return newline;
}

} // namespace limap

