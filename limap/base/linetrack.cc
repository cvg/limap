#include "base/linetrack.h"
#include "base/line_dists.h"

#include <colmap/util/logging.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>

namespace limap {

LineTrack::LineTrack(const LineTrack &track) {
  size_t n_lines = count_lines();
  line = track.line;
  std::copy(track.image_id_list.begin(), track.image_id_list.end(),
            std::back_inserter(image_id_list));
  std::copy(track.line_id_list.begin(), track.line_id_list.end(),
            std::back_inserter(line_id_list));
  std::copy(track.line2d_list.begin(), track.line2d_list.end(),
            std::back_inserter(line2d_list));

  std::copy(track.node_id_list.begin(), track.node_id_list.end(),
            std::back_inserter(node_id_list));
  std::copy(track.score_list.begin(), track.score_list.end(),
            std::back_inserter(score_list));
  std::copy(track.line3d_list.begin(), track.line3d_list.end(),
            std::back_inserter(line3d_list));
  active = track.active;
}

py::dict LineTrack::as_dict() const {
  py::dict output;
  output["line"] = line.as_array();
  output["image_id_list"] = image_id_list;
  output["line_id_list"] = line_id_list;
  output["node_id_list"] = node_id_list;
  output["score_list"] = score_list;
  std::vector<Eigen::MatrixXd> py_line2d_list, py_line3d_list;
  for (auto it = line2d_list.begin(); it != line2d_list.end(); ++it)
    py_line2d_list.push_back(it->as_array());
  for (auto it = line3d_list.begin(); it != line3d_list.end(); ++it)
    py_line3d_list.push_back(it->as_array());
  output["line2d_list"] = py_line2d_list;
  output["line3d_list"] = py_line3d_list;
  output["active"] = active;
  return output;
}

LineTrack::LineTrack(py::dict dict) {
  Eigen::MatrixXd py_line;
  if (dict.contains("line"))
    py_line = dict["line"].cast<Eigen::MatrixXd>();
  else
    throw std::runtime_error("Error! Key \"line\" does not exist!");
  line = Line3d(py_line);
  ASSIGN_PYDICT_ITEM(dict, image_id_list, std::vector<int>)
  ASSIGN_PYDICT_ITEM(dict, line_id_list, std::vector<int>)
  ASSIGN_PYDICT_ITEM(dict, node_id_list, std::vector<int>)
  ASSIGN_PYDICT_ITEM(dict, score_list, std::vector<double>)
  std::vector<Eigen::MatrixXd> py_line2d_list, py_line3d_list;
  if (dict.contains("line2d_list")) {
    py_line2d_list = dict["line2d_list"].cast<std::vector<Eigen::MatrixXd>>();
    for (auto it = py_line2d_list.begin(); it != py_line2d_list.end(); ++it)
      line2d_list.push_back(Line2d(*it));
  }
  if (dict.contains("line3d_list")) {
    py_line3d_list = dict["line3d_list"].cast<std::vector<Eigen::MatrixXd>>();
    for (auto it = py_line3d_list.begin(); it != py_line3d_list.end(); ++it)
      line3d_list.push_back(Line3d(*it));
  }
  ASSIGN_PYDICT_ITEM(dict, active, bool)
}

std::vector<int> LineTrack::GetSortedImageIds() const {
  std::set<int> image_ids;
  for (int i = 0; i < image_id_list.size(); ++i) {
    image_ids.insert(image_id_list[i]);
  }
  std::vector<int> image_ids_vec;
  for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
    image_ids_vec.push_back(*it);
  }
  return image_ids_vec;
}

std::map<int, int> LineTrack::GetIndexMapforSorted() const {
  std::vector<int> image_ids = GetSortedImageIds();
  std::map<int, int> idmap;
  int n_images = count_images();
  for (int i = 0; i < n_images; ++i) {
    idmap.insert(std::make_pair(image_ids[i], i));
  }
  return idmap;
}

std::vector<int> LineTrack::GetIndexesforSorted() const {
  auto idmap = GetIndexMapforSorted();
  std::vector<int> p_image_ids;
  int n_lines = count_lines();
  for (int i = 0; i < n_lines; ++i) {
    p_image_ids.push_back(idmap[image_id_list[i]]);
  }
  return p_image_ids;
}

std::vector<Line2d>
LineTrack::projection(const std::vector<CameraView> &views) const {
  std::vector<Line2d> line2ds;
  size_t num_lines = node_id_list.size();
  for (size_t line_id = 0; line_id < num_lines; ++line_id) {
    line2ds.push_back(line.projection(views[image_id_list[line_id]]));
  }
  return line2ds;
}

void LineTrack::Resize(const size_t &n_lines) {
  image_id_list.resize(n_lines);
  line_id_list.resize(n_lines);
  node_id_list.resize(n_lines);
  score_list.resize(n_lines);
  line2d_list.resize(n_lines);
  line3d_list.resize(n_lines);
}

bool LineTrack::HasImage(const int &image_id) const {
  for (auto it = image_id_list.begin(); it != image_id_list.end(); ++it) {
    if (*it == image_id)
      return true;
  }
  return false;
}

void LineTrack::Write(const std::string &filename) const {
  std::ofstream file;
  file.open(filename.c_str());
  file << std::fixed << std::setprecision(10);
  // row1: line
  for (int i = 0; i < 3; ++i) {
    if (std::isnan(line.start[i])) {
      std::cout << "Warning! NaN values detected." << std::endl;
      ;
      file << 0.0 << " ";
    } else
      file << line.start[i] << " ";
  }
  for (int i = 0; i < 3; ++i) {
    if (std::isnan(line.end[i])) {
      std::cout << "Warning! NaN values detected." << std::endl;
      ;
      file << 0.0 << " ";
    } else
      file << line.end[i] << " ";
  }
  file << "\n";
  // row2: counts
  size_t n_lines = count_lines();
  size_t n_images = count_images();
  file << n_lines << " " << n_images << "\n";
  // row3: image id
  file << "image_id_list ";
  for (size_t i = 0; i < n_lines; ++i)
    file << image_id_list[i] << " ";
  file << "\n";
  // row4: line id
  file << "line_id_list ";
  for (size_t i = 0; i < n_lines; ++i)
    file << line_id_list[i] << " ";
  file << "\n";
  // row5: line2d_list
  file << "line2d_list\n";
  // row(5+i): line2d_list[i].start line2d_list[i].end
  for (size_t i = 0; i < n_lines; ++i) {
    const Line2d &line2d = line2d_list[i];
    file << line2d.start[0] << " " << line2d.start[1] << " ";
    file << line2d.end[0] << " " << line2d.end[1] << " ";
    file << "\n";
  }
  //////////////////////////////////////////////////////////////////////////////////////////
  // auxiliary information
  //////////////////////////////////////////////////////////////////////////////////////////
  // row6: node_id_list
  if (node_id_list.empty() == false) {
    file << "node_id_list ";
    for (size_t i = 0; i < n_lines; ++i)
      file << node_id_list[i] << " ";
    file << "\n";
  }
  // row7: score_list
  if (score_list.empty() == false) {
    file << "score_list ";
    for (size_t i = 0; i < n_lines; ++i)
      file << score_list[i] << " ";
    file << "\n";
  }
  // row8: line3d_list
  // row(8+i): line3d_list[i].start line3d_list[i].end
  if (line3d_list.empty() == false) {
    file << "line3d_list\n";
    for (size_t i = 0; i < n_lines; ++i) {
      const Line3d &line3d = line3d_list[i];
      file << line3d.start[0] << " " << line3d.start[1] << " "
           << line3d.start[2] << " ";
      file << line3d.end[0] << " " << line3d.end[1] << " " << line3d.end[2]
           << " ";
      file << "\n";
    }
  }
  file << "END\n";
}

void LineTrack::Read(const std::string &filename) {
  std::ifstream file;
  file.open(filename.c_str());
  // row1: line
  for (int i = 0; i < 3; ++i)
    file >> line.start[i];
  for (int i = 0; i < 3; ++i)
    file >> line.end[i];
  // row2: counts
  size_t n_lines, n_images;
  file >> n_lines >> n_images;
  Resize(n_lines);
  // row3: image id
  std::string str;
  file >> str;
  THROW_CHECK_EQ(str, "image_id_list");
  for (size_t i = 0; i < n_lines; ++i)
    file >> image_id_list[i];
  // row4: line id
  file >> str;
  THROW_CHECK_EQ(str, "line_id_list");
  for (size_t i = 0; i < n_lines; ++i)
    file >> line_id_list[i];
  // row5: line2d_list
  file >> str;
  // adapt to the previous version for visualization
  if (str != "line2d_list")
    return;
  // row(5+i): line2d_list[i].start line2d_list[i].end
  for (size_t i = 0; i < n_lines; ++i) {
    Line2d &line2d = line2d_list[i];
    file >> line2d.start[0] >> line2d.start[1];
    file >> line2d.end[0] >> line2d.end[1];
  }
  file >> str;
  if (str == "END") {
    return;
  }
  //////////////////////////////////////////////////////////////////////////////////////////
  // auxiliary information
  //////////////////////////////////////////////////////////////////////////////////////////
  // row6: node_id_list
  THROW_CHECK_EQ(str, "node_id_list");
  for (size_t i = 0; i < n_lines; ++i)
    file >> node_id_list[i];
  // row7: scores
  file >> str;
  THROW_CHECK_EQ(str, "score_list");
  for (size_t i = 0; i < n_lines; ++i)
    file >> score_list[i];
  // row8: line3d_list
  // row(8+i): line3d_list[i].start line3d_list[i].end
  file >> str;
  THROW_CHECK_EQ(str, "line3d_list");
  for (size_t i = 0; i < n_lines; ++i) {
    Line3d &line3d = line3d_list[i];
    file >> line3d.start[0] >> line3d.start[1] >> line3d.start[2];
    file >> line3d.end[0] >> line3d.end[1] >> line3d.end[2];
  }
}

std::map<int, std::vector<int>> LineTrack::GetIdMap() const {
  std::map<int, std::vector<int>> m;
  int n_lines = count_lines();
  for (int i = 0; i < n_lines; ++i) {
    int image_id = image_id_list[i];
    if (m.find(image_id) == m.end())
      m.insert(std::make_pair(image_id, std::vector<int>()));
    m[image_id].push_back(i);
  }
  return m;
}

void ComputeLineWeightsNormalized(const LineTrack &track,
                                  std::vector<double> &weights) {
  int n_lines = track.count_lines();
  int n_images = track.count_images();

  // compute sum of lengths for each image
  std::map<int, double> sum_lengths;
  for (int i = 0; i < n_lines; ++i) {
    int image_id = track.image_id_list[i];
    double length = track.line2d_list[i].length();
    if (sum_lengths.find(image_id) == sum_lengths.end())
      sum_lengths.insert(std::make_pair(image_id, length));
    else
      sum_lengths.at(image_id) += length;
  }

  double all_length = 0;
  for (auto it = sum_lengths.begin(); it != sum_lengths.end(); ++it)
    all_length += it->second;

  // compute weights
  weights.clear();
  for (int i = 0; i < n_lines; ++i) {
    int image_id = track.image_id_list[i];
    double length = track.line2d_list[i].length();
    double weight = 100.0 * length / sum_lengths.at(image_id);
    weight = weight * sum_lengths.at(image_id) / all_length;
    weights.push_back(weight);
  }
}

void ComputeLineWeights(const LineTrack &track, std::vector<double> &weights) {
  weights.clear();
  for (size_t i = 0; i < track.count_lines(); ++i) {
    double length = track.line2d_list[i].length();
    double weight = length / 30.0;
    weights.push_back(weight);
  }
}

void ComputeHeatmapSamples(
    const LineTrack &track,
    std::vector<std::vector<InfiniteLine2d>> &heatmap_samples,
    const std::pair<double, double> sample_range, const int n_samples) {
  // compute t_array
  double interval =
      (sample_range.second - sample_range.first) / (n_samples - 1);
  std::vector<double> t_array(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    double val = sample_range.first + interval * i;
    t_array[i] = val;
  }

  // collect sample infinite 2d lines for each supporting line2d
  int n_lines = track.count_lines();
  heatmap_samples.resize(n_lines);
  for (size_t line_id = 0; line_id < n_lines; ++line_id) {
    const Line2d &line2d = track.line2d_list[line_id];
    V2D dir_perp = line2d.perp_direction();

    for (int j = 0; j < n_samples; ++j) {
      double val = t_array[j];
      V2D point = line2d.start + val * (line2d.end - line2d.start);
      InfiniteLine2d inf_line(point, dir_perp);
      heatmap_samples.at(line_id).push_back(inf_line);
    }
  }
}

void ComputeFConsistencySamples(
    const LineTrack &track, const std::map<int, CameraView> &views,
    std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>>
        &fconsis_samples,
    const std::pair<double, double> sample_range, const int n_samples) {
  // initialize and check
  fconsis_samples.clear();
  std::vector<int> image_ids = track.GetSortedImageIds();
  for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
    if (views.count(*it) == 0)
      throw std::runtime_error("Error! some cameras are missing");
  }

  // sample on 3d
  std::vector<V3D> points;
  const auto &line = track.line;
  V3D interval = (line.end - line.start) / (n_samples - 1);
  for (int i = 0; i < n_samples; ++i) {
    V3D point = line.start + i * interval;
    points.push_back(point);
  }

  // get all line projections
  std::map<int, Line2d> line_projections;
  for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
    const auto &view = views.at(*it);
    Line2d line_projection = track.line.projection(view);
    line_projections.insert(std::make_pair(*it, line_projection));
  }

  for (const auto &point : points) {
    // get projections to all images
    std::map<int, V2D> projections;
    for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
      const auto &view = views.at(*it);
      V2D projection = view.projection(point);
      projections.insert(std::make_pair(*it, projection));
    }

    // check supporting line for each point
    std::vector<int> supports;
    for (int line_id = 0; line_id < track.count_lines(); ++line_id) {
      int img_id = track.image_id_list[line_id];
      Line2d line2d = track.line2d_list[line_id];
      const V2D &xy = projections.at(img_id);
      // project the 2d projection onto the line
      double length = line2d.length();
      V2D dir = line2d.direction();
      double proj = (xy - line2d.start).dot(dir) / length;
      if (proj >= sample_range.first && proj <= sample_range.second)
        supports.push_back(line_id);
    }

    // find the reference image with the longest supporting line within a
    // certain threshold in pixels
    double th_perp = 0.3;
    // transform it into 2d infinite line
    int n_supports = supports.size();
    if (n_supports < 2)
      continue;
    std::vector<int> good_supports;
    for (auto it = supports.begin(); it != supports.end(); ++it) {
      const Line2d &line2d = track.line2d_list[*it];
      int img_id = track.image_id_list[*it];
      const Line2d &line2d_proj = line_projections.at(img_id);
      double perp = compute_distance<Line2d>(
          line2d, line2d_proj, LineDistType::PERPENDICULAR_ONEWAY);
      if (perp < th_perp)
        good_supports.push_back(*it);
    }
    if (good_supports.empty())
      continue;
    double max_length = -1;
    int max_id = -1;
    for (auto it = good_supports.begin(); it != good_supports.end(); ++it) {
      double length = track.line2d_list[*it].length();
      if (length > max_length) {
        max_length = length;
        max_id = *it;
      }
    }
    int ref_image_id = track.image_id_list[max_id];
    const V2D &ref_point_projection = projections.at(ref_image_id);
    const auto &ref_line2d = track.line2d_list[max_id];
    V2D ref_dir_perp = ref_line2d.perp_direction();
    InfiniteLine2d inf_line(ref_point_projection, ref_dir_perp);

    // collect all supporting images
    std::set<int> image_ids;
    for (auto it = supports.begin(); it != supports.end(); ++it) {
      image_ids.insert(track.image_id_list[*it]);
    }
    std::vector<int> tgt_image_ids;
    for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
      if (*it == ref_image_id)
        continue;
      tgt_image_ids.push_back(*it);
    }
    fconsis_samples.push_back(
        std::make_tuple(ref_image_id, inf_line, tgt_image_ids));
  }
}

} // namespace limap
