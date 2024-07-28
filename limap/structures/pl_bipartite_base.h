#ifndef LIMAP_STRUCTURES_PL_BIPARTITE_BASE_H
#define LIMAP_STRUCTURES_PL_BIPARTITE_BASE_H

#include <map>
#include <set>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "_limap/helpers.h"

namespace limap {

namespace structures {

template <typename PTYPE> class Junction {
public:
  Junction() {}
  Junction(const PTYPE &p_,
           const std::vector<int> &line_ids_ = std::vector<int>())
      : p(p_), line_ids(line_ids_) {}

  PTYPE p;
  std::vector<int> line_ids; // the lines that are associated with it
  size_t degree() const { return line_ids.size(); }
};

template <typename PTYPE, typename LTYPE> class PL_Bipartite {
public:
  PL_Bipartite() {}
  ~PL_Bipartite() {}
  PL_Bipartite(const PL_Bipartite &obj)
      : points_(obj.points_), lines_(obj.lines_), np2l_(obj.np2l_),
        nl2p_(obj.nl2p_) {}
  PL_Bipartite(py::dict dict);
  py::dict as_dict() const;

  // insertion and deletion
  void add_edge(const int &point_id, const int &line_id);
  void delete_edge(const int &point_id, const int &line_id);
  void clear_edges();
  int add_point(const PTYPE &p, int point_id = -1,
                const std::vector<int> &neighbors = std::vector<int>());
  int add_line(const LTYPE &line, int line_id = -1,
               const std::vector<int> &neighbors = std::vector<int>());
  void delete_point(const int &point_id);
  void delete_line(const int &line_id);
  void update_point(const int &point_id, const PTYPE &p);
  void update_line(const int &line_id, const LTYPE &line);
  void clear_points();
  void clear_lines();
  void init_points(const std::vector<PTYPE> &points,
                   const std::vector<int> &ids = std::vector<int>());
  void init_lines(const std::vector<LTYPE> &lines,
                  const std::vector<int> &ids = std::vector<int>());
  void reset();

  // const operation
  size_t count_lines() const { return lines_.size(); }
  size_t count_points() const { return points_.size(); }
  int count_edges() const;
  bool exist_point(const int point_id) const {
    return points_.count(point_id) == 1;
  }
  bool exist_line(const int line_id) const {
    return lines_.count(line_id) == 1;
  }
  std::map<int, PTYPE> get_dict_points() const { return points_; }
  std::map<int, LTYPE> get_dict_lines() const { return lines_; }
  std::vector<PTYPE> get_all_points() const;
  std::vector<LTYPE> get_all_lines() const;
  std::vector<int> get_point_ids() const;
  std::vector<int> get_line_ids() const;
  size_t pdegree(const int point_id) const;
  size_t ldegree(const int line_id) const;
  std::vector<int> neighbor_lines(const int point_id) const;
  std::vector<int> neighbor_points(const int line_id) const;
  PTYPE point(const int point_id) const;
  LTYPE line(const int line_id) const;

  // junction
  Junction<PTYPE> junc(const int point_id) const;
  std::vector<Junction<PTYPE>> get_all_junctions() const;
  void add_junction(const Junction<PTYPE> &junction, int point_id = -1);

protected:
  std::map<int, PTYPE> points_;
  std::map<int, LTYPE> lines_;
  std::map<int, std::set<int>> np2l_; // neighboring lines for each point
  std::map<int, std::set<int>> nl2p_; // neighboring points for each line

  int get_default_new_point_id() const;
  int get_default_new_line_id() const;
};

} // namespace structures

} // namespace limap

#endif
