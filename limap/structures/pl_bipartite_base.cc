#include "structures/pl_bipartite_base.h"

#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"
#include "util/types.h"
#include "vplib/vptrack.h"

namespace limap {

namespace structures {

template <typename PTYPE, typename LTYPE>
py::dict PL_Bipartite<PTYPE, LTYPE>::as_dict() const {
  py::dict output;
  output["points_"] = points_;
  output["lines_"] = lines_;
  output["np2l_"] = np2l_;
  output["nl2p_"] = nl2p_;
  return output;
}

template <typename PTYPE, typename LTYPE>
PL_Bipartite<PTYPE, LTYPE>::PL_Bipartite(py::dict dict) {
#define TMPMAPTYPE std::map<int, PTYPE>
  ASSIGN_PYDICT_ITEM(dict, points_, TMPMAPTYPE)
#undef TMPMAPTYPE
#define TMPMAPTYPE std::map<int, LTYPE>
  ASSIGN_PYDICT_ITEM(dict, lines_, TMPMAPTYPE)
#undef TMPMAPTYPE
#define TMPMAPTYPE std::map<int, std::set<int>>
  ASSIGN_PYDICT_ITEM(dict, np2l_, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, nl2p_, TMPMAPTYPE)
#undef TMPMAPTYPE
}

template <typename PTYPE, typename LTYPE>
int PL_Bipartite<PTYPE, LTYPE>::get_default_new_point_id() const {
  if (points_.empty())
    return 0;
  else
    return points_.rbegin()->first + 1;
}

template <typename PTYPE, typename LTYPE>
int PL_Bipartite<PTYPE, LTYPE>::get_default_new_line_id() const {
  if (lines_.empty())
    return 0;
  else
    return lines_.rbegin()->first + 1;
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::add_edge(const int &point_id,
                                          const int &line_id) {
  THROW_CHECK_EQ(exist_point(point_id), true);
  THROW_CHECK_EQ(exist_line(line_id), true);
  np2l_.at(point_id).insert(line_id);
  nl2p_.at(line_id).insert(point_id);
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::delete_edge(const int &point_id,
                                             const int &line_id) {
  THROW_CHECK_EQ(exist_point(point_id), true);
  THROW_CHECK_EQ(exist_line(line_id), true);
  np2l_.at(point_id).erase(line_id);
  nl2p_.at(line_id).erase(point_id);
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::clear_edges() {
  for (auto it = np2l_.begin(); it != np2l_.end(); ++it) {
    it->second.clear();
  }
  for (auto it = nl2p_.begin(); it != nl2p_.end(); ++it) {
    it->second.clear();
  }
}

template <typename PTYPE, typename LTYPE>
int PL_Bipartite<PTYPE, LTYPE>::add_point(const PTYPE &p, int point_id,
                                          const std::vector<int> &neighbors) {
  if (point_id == -1)
    point_id = get_default_new_point_id();
  THROW_CHECK_EQ(exist_point(point_id), false);
  points_.insert(std::make_pair(point_id, p));
  np2l_.insert(std::make_pair(point_id, std::set<int>()));
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    int line_id = *it;
    THROW_CHECK_EQ(exist_line(line_id), true);
    np2l_.at(point_id).insert(line_id);
    nl2p_.at(line_id).insert(point_id);
  }
  return point_id;
}

template <typename PTYPE, typename LTYPE>
int PL_Bipartite<PTYPE, LTYPE>::add_line(const LTYPE &line, int line_id,
                                         const std::vector<int> &neighbors) {
  if (line_id == -1)
    line_id = get_default_new_line_id();
  THROW_CHECK_EQ(exist_line(line_id), false);
  lines_.insert(std::make_pair(line_id, line));
  nl2p_.insert(std::make_pair(line_id, std::set<int>()));
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    int point_id = *it;
    THROW_CHECK_EQ(exist_point(point_id), true);
    nl2p_.at(line_id).insert(point_id);
    np2l_.at(point_id).insert(line_id);
  }
  return line_id;
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::delete_point(const int &point_id) {
  THROW_CHECK_EQ(exist_point(point_id), true);
  const auto &neighbors = np2l_.at(point_id);
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    int line_id = *it;
    nl2p_.at(line_id).erase(point_id);
  }
  np2l_.at(point_id).clear();
  np2l_.erase(point_id);
  points_.erase(point_id);
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::delete_line(const int &line_id) {
  THROW_CHECK_EQ(exist_line(line_id), true);
  const auto &neighbors = nl2p_.at(line_id);
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    int point_id = *it;
    np2l_.at(point_id).erase(line_id);
  }
  nl2p_.at(line_id).clear();
  nl2p_.erase(line_id);
  lines_.erase(line_id);
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::update_point(const int &point_id,
                                              const PTYPE &p) {
  THROW_CHECK_EQ(exist_point(point_id), true);
  points_.at(point_id) = p;
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::update_line(const int &line_id,
                                             const LTYPE &line) {
  THROW_CHECK_EQ(exist_line(line_id), true);
  lines_.at(line_id) = line;
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::clear_points() {
  points_.clear();
  clear_edges();
  np2l_.clear();
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::clear_lines() {
  lines_.clear();
  clear_edges();
  nl2p_.clear();
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::init_points(const std::vector<PTYPE> &points,
                                             const std::vector<int> &ids) {
  if (!ids.empty())
    THROW_CHECK_EQ(points.size(), ids.size());
  size_t n_points = points.size();
  for (size_t idx = 0; idx < n_points; ++idx) {
    int point_id = idx;
    if (!ids.empty())
      point_id = ids[idx];
    add_point(points[idx], point_id);
  }
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::init_lines(const std::vector<LTYPE> &lines,
                                            const std::vector<int> &ids) {
  if (!ids.empty())
    THROW_CHECK_EQ(lines.size(), ids.size());
  size_t n_lines = lines.size();
  for (size_t idx = 0; idx < n_lines; ++idx) {
    int line_id = idx;
    if (!ids.empty())
      line_id = ids[idx];
    add_line(lines[idx], line_id);
  }
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::reset() {
  for (auto it = nl2p_.begin(); it != nl2p_.end(); ++it) {
    it->second.clear();
  }
  nl2p_.clear();
  for (auto it = np2l_.begin(); it != np2l_.end(); ++it) {
    it->second.clear();
  }
  np2l_.clear();
  points_.clear();
  lines_.clear();
}

template <typename PTYPE, typename LTYPE>
int PL_Bipartite<PTYPE, LTYPE>::count_edges() const {
  int counter = 0;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    counter += ldegree(it->first);
  }
  return counter;
}

template <typename PTYPE, typename LTYPE>
std::vector<PTYPE> PL_Bipartite<PTYPE, LTYPE>::get_all_points() const {
  std::vector<PTYPE> output;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    output.push_back(it->second);
  }
  return output;
}

template <typename PTYPE, typename LTYPE>
std::vector<LTYPE> PL_Bipartite<PTYPE, LTYPE>::get_all_lines() const {
  std::vector<LTYPE> output;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    output.push_back(it->second);
  }
  return output;
}

template <typename PTYPE, typename LTYPE>
std::vector<int> PL_Bipartite<PTYPE, LTYPE>::get_point_ids() const {
  std::vector<int> output;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    output.push_back(it->first);
  }
  return output;
}

template <typename PTYPE, typename LTYPE>
std::vector<int> PL_Bipartite<PTYPE, LTYPE>::get_line_ids() const {
  std::vector<int> output;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    output.push_back(it->first);
  }
  return output;
}

template <typename PTYPE, typename LTYPE>
size_t PL_Bipartite<PTYPE, LTYPE>::pdegree(const int point_id) const {
  THROW_CHECK_EQ(exist_point(point_id), true);
  return np2l_.at(point_id).size();
}

template <typename PTYPE, typename LTYPE>
size_t PL_Bipartite<PTYPE, LTYPE>::ldegree(const int line_id) const {
  THROW_CHECK_EQ(exist_line(line_id), true);
  return nl2p_.at(line_id).size();
}

template <typename PTYPE, typename LTYPE>
std::vector<int>
PL_Bipartite<PTYPE, LTYPE>::neighbor_lines(const int point_id) const {
  THROW_CHECK_EQ(exist_point(point_id), true);
  std::set<int> neighbors = np2l_.at(point_id);
  std::vector<int> output(neighbors.begin(), neighbors.end());
  return output;
}

template <typename PTYPE, typename LTYPE>
std::vector<int>
PL_Bipartite<PTYPE, LTYPE>::neighbor_points(const int line_id) const {
  THROW_CHECK_EQ(exist_line(line_id), true);
  std::set<int> neighbors = nl2p_.at(line_id);
  std::vector<int> output(neighbors.begin(), neighbors.end());
  return output;
}

template <typename PTYPE, typename LTYPE>
PTYPE PL_Bipartite<PTYPE, LTYPE>::point(const int point_id) const {
  THROW_CHECK_EQ(exist_point(point_id), true);
  return points_.at(point_id);
}

template <typename PTYPE, typename LTYPE>
LTYPE PL_Bipartite<PTYPE, LTYPE>::line(const int line_id) const {
  THROW_CHECK_EQ(exist_line(line_id), true);
  return lines_.at(line_id);
}

template <typename PTYPE, typename LTYPE>
Junction<PTYPE> PL_Bipartite<PTYPE, LTYPE>::junc(const int point_id) const {
  THROW_CHECK_EQ(exist_point(point_id), true);
  return Junction<PTYPE>(point(point_id), neighbor_lines(point_id));
}

template <typename PTYPE, typename LTYPE>
std::vector<Junction<PTYPE>>
PL_Bipartite<PTYPE, LTYPE>::get_all_junctions() const {
  std::vector<Junction<PTYPE>> junctions;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    junctions.push_back(junc(it->first));
  }
  return junctions;
}

template <typename PTYPE, typename LTYPE>
void PL_Bipartite<PTYPE, LTYPE>::add_junction(const Junction<PTYPE> &junction,
                                              int point_id) {
  if (point_id == -1)
    point_id = get_default_new_point_id();
  add_point(junction.p, point_id, junction.line_ids);
}

template class PL_Bipartite<Point2d, Line2d>;
template class PL_Bipartite<PointTrack, LineTrack>;
template class PL_Bipartite<vplib::VP2d, Line2d>; // vp-line bipartite on 2d
template class PL_Bipartite<vplib::VPTrack,
                            LineTrack>; // vp-line bipartite on 3d

} // namespace structures

} // namespace limap
