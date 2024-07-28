#include "structures/vpline_bipartite.h"

namespace limap {

namespace structures {

py::dict VPLine_Bipartite2d::as_dict() const {
  py::dict output;
  std::map<int, py::dict> dict_points;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    dict_points.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["points_"] = dict_points;
  std::map<int, Eigen::MatrixXd> dict_lines;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    dict_lines.insert(std::make_pair(it->first, it->second.as_array()));
  }
  output["lines_"] = dict_lines;
  output["np2l_"] = np2l_;
  output["nl2p_"] = nl2p_;
  return output;
}

VPLine_Bipartite2d::VPLine_Bipartite2d(py::dict dict) {
  // load points
  std::map<int, py::dict> dict_points;
  if (dict.contains("points_"))
    dict_points = dict["points_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"points_\" does not exist!");
  for (auto it = dict_points.begin(); it != dict_points.end(); ++it) {
    points_.insert(std::make_pair(it->first, vplib::VP2d(it->second)));
  }

  // load lines
  std::map<int, Eigen::MatrixXd> dict_lines;
  if (dict.contains("lines_"))
    dict_lines = dict["lines_"].cast<std::map<int, Eigen::MatrixXd>>();
  else
    throw std::runtime_error("Error! Key \"lines_\" does not exist!");
  for (auto it = dict_lines.begin(); it != dict_lines.end(); ++it) {
    lines_.insert(std::make_pair(it->first, Line2d(it->second)));
  }

  // load connections
#define TMPMAPTYPE std::map<int, std::set<int>>
  ASSIGN_PYDICT_ITEM(dict, np2l_, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, nl2p_, TMPMAPTYPE)
#undef TMPMAPTYPE
}

py::dict VPLine_Bipartite3d::as_dict() const {
  py::dict output;
  std::map<int, py::dict> dict_points;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    dict_points.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["points_"] = dict_points;
  std::map<int, py::dict> dict_lines;
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    dict_lines.insert(std::make_pair(it->first, it->second.as_dict()));
  }
  output["lines_"] = dict_lines;
  output["np2l_"] = np2l_;
  output["nl2p_"] = nl2p_;
  return output;
}

VPLine_Bipartite3d::VPLine_Bipartite3d(py::dict dict) {
  // load points
  std::map<int, py::dict> dict_points;
  if (dict.contains("points_"))
    dict_points = dict["points_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"points_\" does not exist!");
  for (auto it = dict_points.begin(); it != dict_points.end(); ++it) {
    points_.insert(std::make_pair(it->first, vplib::VPTrack(it->second)));
  }

  // load lines
  std::map<int, py::dict> dict_lines;
  if (dict.contains("lines_"))
    dict_lines = dict["lines_"].cast<std::map<int, py::dict>>();
  else
    throw std::runtime_error("Error! Key \"lines_\" does not exist!");
  for (auto it = dict_lines.begin(); it != dict_lines.end(); ++it) {
    lines_.insert(std::make_pair(it->first, LineTrack(it->second)));
  }

  // load connections
#define TMPMAPTYPE std::map<int, std::set<int>>
  ASSIGN_PYDICT_ITEM(dict, np2l_, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, nl2p_, TMPMAPTYPE)
#undef TMPMAPTYPE
}

std::map<int, VPLine_Bipartite2d> GetAllBipartites_VPLine2d(
    const std::map<int, std::vector<Line2d>> &all_2d_lines,
    const std::map<int, vplib::VPResult> &vpresults,
    const std::vector<vplib::VPTrack> &vptracks) {
  // assertion
  THROW_CHECK_EQ(all_2d_lines.size(), vpresults.size());
  std::vector<int> image_ids;
  for (auto it = all_2d_lines.begin(); it != all_2d_lines.end(); ++it) {
    image_ids.push_back(it->first);
  }

  // build invert id map for each image
  std::map<int, std::vector<int>> m_2d_to_3d; // default: -1
  for (auto it = vpresults.begin(); it != vpresults.end(); ++it) {
    int img_id = it->first;
    m_2d_to_3d.insert(std::make_pair(img_id, std::vector<int>()));
    m_2d_to_3d.at(img_id).resize(it->second.count_vps());
    std::fill(m_2d_to_3d.at(img_id).begin(), m_2d_to_3d.at(img_id).end(), -1);
  }
  for (size_t track_id = 0; track_id < vptracks.size(); ++track_id) {
    const auto &vptrack = vptracks[track_id];
    for (auto it = vptrack.supports.begin(); it != vptrack.supports.end();
         ++it) {
      int img_id = it->first;
      int vp2d_id = it->second;
      m_2d_to_3d.at(img_id)[vp2d_id] = track_id;
    }
  }

  // init each bipartite
  std::map<int, VPLine_Bipartite2d> all_bipartites;
  for (const int &img_id : image_ids) {
    VPLine_Bipartite2d bpt;

    // init lines
    bpt.init_lines(all_2d_lines.at(img_id));

    // init vps
    const auto &vpres = vpresults.at(img_id);
    for (int vp2d_id = 0; vp2d_id < vpres.count_vps(); ++vp2d_id) {
      int vp3d_id = m_2d_to_3d.at(img_id)[vp2d_id];
      if (vp3d_id < 0)
        continue;
      vplib::VP2d vp2d = vplib::VP2d(vpres.vps[vp2d_id], vp3d_id);
      bpt.add_point(vp2d, vp2d_id);
    }

    // add edges
    for (size_t line_id = 0; line_id < all_2d_lines.at(img_id).size();
         ++line_id) {
      if (!vpres.HasVP(line_id))
        continue;
      int vp2d_id = vpres.GetVPLabel(line_id);
      int vp3d_id = m_2d_to_3d.at(img_id)[vp2d_id];
      if (vp3d_id < 0)
        continue;
      bpt.add_edge(vp2d_id, line_id);
    }
    all_bipartites.insert(std::make_pair(img_id, bpt));
  }
  return all_bipartites;
}

} // namespace structures

} // namespace limap
