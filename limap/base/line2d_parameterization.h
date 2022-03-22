#ifndef LIMAP_BASE_LINE2D_PARAMETERIZATION_H_
#define LIMAP_BASE_LINE2D_PARAMETERIZATION_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/linebase.h"
#include "base/infinite_line.h"
#include "util/types.h"
#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

class Line2d_2DOF {
public:  
    Line2d_2DOF() {}
    Line2d_2DOF(const Line2d& line2d) { line = line2d; vars = V2D(0.0, 0.0); }
    Line2d_2DOF(const Line2d& line2d, const std::string& method_): Line2d_2DOF(line2d) { 
        if (method_ == "fixedprojection")
            method = method_;
        else if (method_ == "fixedlength")
            method = method_;
        else
            throw std::runtime_error("method not supported");
    }

    Line2d line;
    V2D vars;
    std::string method = "fixedprojection";

    Line2d GetOriginalLine() const {return line; }
    V2D GetPoint(const double val) const {return line.start + val * (line.end - line.start); }
    V2D& GetVars() {return vars; }

    Line2d GetLine() const {
        if (method == "fixedprojection")
            return GetLine_fixedprojection();
        else if (method == "fixedlength")
            return GetLine_fixedlength();
        else
            throw std::runtime_error("method not supported");
    }

    // get a point on the line
    template <typename T>
    void GetPoint(const T input[2], const double val, T point[2]) const {
        T start[2], end[2];
        Ceres_GetLine(input, start, end);
        point[0] = start[0] + (end[0] - start[0]) * val;
        point[1] = start[1] + (end[1] - start[1]) * val;
    }

    template <typename T> 
    void GetInfiniteLine2d(const T input[2], T p[2], T dir[2]) const {
        T start[2], end[2];
        Ceres_GetLine(input, start, end);
        p[0] = start[0]; p[1] = start[1];
        dir[0] = end[0] - start[0]; dir[1] = end[1] - start[1];
        T length = ceres::sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
        dir[0] /= length;
        dir[1] /= length;
    }

    // TODO: Need to figure out how to do polymorphism with template method
    // virtual Line2d GetLine() const = 0;
    // template <typename T>
    // virtual void Ceres_GetLine(const T input[2], T start[2], T end[2]) const = 0; 

    template <typename T>
    void Ceres_GetLine(const T input[2], T start[2], T end[2]) const {
        if (method == "fixedprojection")
            Ceres_GetLine_fixedprojection<T>(input, start, end);
        else if (method == "fixedlength")
            Ceres_GetLine_fixedlength<T>(input, start, end);
        else
            throw std::runtime_error("method not supported");
    }

private:
    Line2d GetLine_fixedprojection() const;
    Line2d GetLine_fixedlength() const;

    template <typename T>
    void Ceres_GetLine_fixedprojection(const T input[2], T start[2], T end[2]) const {
        V2D dir_perp = line.perp_direction();
        T dir_perp_vec[2] = {T(dir_perp(0)), T(dir_perp(1))};
        T start_vec[2] = {T(line.start(0)), T(line.start(1))};
        T end_vec[2] = {T(line.end(0)), T(line.end(1))};

        start[0] = start_vec[0] + dir_perp_vec[0] * input[0];
        start[1] = start_vec[1] + dir_perp_vec[1] * input[0];

        end[0] = end_vec[0] + dir_perp_vec[0] * input[1];
        end[1] = end_vec[1] + dir_perp_vec[1] * input[1];
    }
    
    template <typename T>
    void Ceres_GetLine_fixedlength(const T input[2], T start[2], T end[2]) const {
        V2D dir = line.direction();
        T dir_vec[2] = {T(dir(0)), T(dir(1))};
        V2D dir_perp = line.perp_direction();
        T dir_perp_vec[2] = {T(dir_perp(0)), T(dir_perp(1))};
        V2D midpoint = line.midpoint();
        T midpoint_vec[2] = {T(midpoint(0)), T(midpoint(1))};

        T new_direc[2];
        T cosval = ceres::cos(input[0]);
        T sinval = ceres::sin(input[0]);
        new_direc[0] = cosval * dir_vec[0] - sinval * dir_vec[1];
        new_direc[1] = sinval * dir_vec[0] + cosval * dir_vec[1];

        T new_midpoint[2];
        new_midpoint[0] = midpoint_vec[0] + dir_perp_vec[0] * input[1];
        new_midpoint[1] = midpoint_vec[1] + dir_perp_vec[1] * input[1];

        T halflength = T(line.length() / 2.0);
        start[0] = new_midpoint[0] - new_direc[0] * halflength;
        start[1] = new_midpoint[1] - new_direc[1] * halflength;

        end[0] = new_midpoint[0] + new_direc[0] * halflength;
        end[1] = new_midpoint[1] + new_direc[1] * halflength;
    }
};

} // namespace limap

#endif

