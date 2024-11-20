#include "optimize/global_pl_association/global_associator.h"
#include "optimize/global_pl_association/cost_functions.h"

#include "base/line_linker.h"
#include "base/linetrack.h"
#include "ceresbase/parameterization.h"
#include "optimize/line_refinement/cost_functions.h"

#include <cmath>
#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/util/logging.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>

namespace limap {

namespace optimize {

namespace global_pl_association {

void GlobalAssociator::InitVPTracks(
    const std::map<int, vplib::VPTrack> &vptracks) {
  vps_.clear();
  vp_tracks_.clear();
  for (auto it = vptracks.begin(); it != vptracks.end(); ++it) {
    vps_.insert(std::make_pair(it->first, it->second.direction));
    vp_tracks_.insert(std::make_pair(it->first, it->second));
  }
}

void GlobalAssociator::InitVPTracks(
    const std::vector<vplib::VPTrack> &vptracks) {
  vps_.clear();
  vp_tracks_.clear();
  size_t n_tracks = vptracks.size();
  for (size_t track_id = 0; track_id < n_tracks; ++track_id) {
    vps_.insert(std::make_pair(track_id, vptracks[track_id].direction));
    vp_tracks_.insert(std::make_pair(track_id, vptracks[track_id]));
  }
}

void GlobalAssociator::ReassociateJunctions() {
  // build invert map from all_bpt2ds to the id of linetracks
  std::map<int, std::map<int, int>> m_lineid_2dto3d;
  for (auto it = all_bpt2ds_.begin(); it != all_bpt2ds_.end(); ++it) {
    m_lineid_2dto3d.insert(std::make_pair(it->first, std::map<int, int>()));
  }
  for (const auto &kv : line_tracks_) {
    int line3d_id = kv.first;
    const LineTrack &linetrack = line_tracks_.at(line3d_id);
    size_t n_supports = linetrack.count_lines();
    for (size_t support_id = 0; support_id < n_supports; ++support_id) {
      int img_id = linetrack.image_id_list[support_id];
      int line2d_id = linetrack.line_id_list[support_id];
      if (m_lineid_2dto3d.at(img_id).find(line2d_id) ==
          m_lineid_2dto3d.at(img_id).end())
        m_lineid_2dto3d.at(img_id).insert(std::make_pair(line2d_id, line3d_id));
      else
        throw std::runtime_error("Error!");
    }
  }

  // count connections by traversing 2d bipartites
  std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>
      track_edges_counter; // ((linetrack_id1, linetrack_id2), vector<img_id,
                           // point2d_id>)
  for (auto it_bpt2d = all_bpt2ds_.begin(); it_bpt2d != all_bpt2ds_.end();
       ++it_bpt2d) {
    int img_id = it_bpt2d->first;
    auto bpt2d = it_bpt2d->second;
    for (const int &point2d_id : bpt2d.get_point_ids()) {
      if (bpt2d.pdegree(point2d_id) <= 1)
        continue;
      std::vector<int> line2d_ids = bpt2d.neighbor_lines(point2d_id);
      size_t n_lines = line2d_ids.size();
      for (size_t i = 0; i < n_lines - 1; ++i) {
        int line2d_id1 = line2d_ids[i];
        if (m_lineid_2dto3d.at(img_id).find(line2d_id1) ==
            m_lineid_2dto3d.at(img_id).end())
          continue;
        int track_id1 = m_lineid_2dto3d.at(img_id).at(line2d_id1);
        Line2d line1 = all_bpt2ds_.at(img_id).line(line2d_id1);
        for (size_t j = i + 1; j < n_lines; ++j) {
          int line2d_id2 = line2d_ids[j];
          if (m_lineid_2dto3d.at(img_id).find(line2d_id2) ==
              m_lineid_2dto3d.at(img_id).end())
            continue;
          int track_id2 = m_lineid_2dto3d.at(img_id).at(line2d_id2);
          Line2d line2 = all_bpt2ds_.at(img_id).line(line2d_id2);
          if (track_id1 == track_id2)
            continue;
          double cosine = std::abs(line1.direction().dot(line2.direction()));
          if (cosine > 1.0)
            cosine = 1.0;
          double angle = acos(cosine) * 180.0 / M_PI;
          if (angle < config_.th_angle_lineline)
            continue;

          // add into counter
          std::pair<int, int> edge;
          if (track_id1 < track_id2)
            edge = std::make_pair(track_id1, track_id2);
          else
            edge = std::make_pair(track_id2, track_id1);
          if (track_edges_counter.find(edge) == track_edges_counter.end()) {
            track_edges_counter.insert(
                std::make_pair(edge, std::vector<std::pair<int, int>>()));
          }
          track_edges_counter.at(edge).push_back(
              std::make_pair(img_id, point2d_id));
        }
      }
    }
  }

  // build junctions
  for (auto it = track_edges_counter.begin(); it != track_edges_counter.end();
       ++it) {
    if (it->second.size() < config_.th_count_lineline)
      continue;

    std::vector<int> image_id_list;
    std::vector<int> p2d_id_list;
    std::vector<V2D> p2d_list;
    auto p2d_data = it->second;
    for (auto it = p2d_data.begin(); it != p2d_data.end(); ++it) {
      int img_id = it->first;
      int p2d_id = it->second;
      image_id_list.push_back(img_id);
      p2d_id_list.push_back(p2d_id);
      p2d_list.push_back(all_bpt2ds_.at(img_id).point(p2d_id).p);
    }
    // test angle betweem tracks, filter track pairs with small angles
    Line3d line1 = line_tracks_.at(it->first.first).line;
    Line3d line2 = line_tracks_.at(it->first.second).line;
    V3D direc1 = line1.direction();
    V3D direc2 = line2.direction();
    double inner_prod = direc1.dot(direc2);
    double cosine = std::abs(inner_prod);
    if (cosine > 1.0)
      cosine = 1.0;
    double angle = acos(cosine) * 180.0 / M_PI;
    if (angle < config_.th_angle_lineline)
      continue;
    // compute midpoint
    M2D A;
    A << 1.0, -inner_prod, -inner_prod, 1.0;
    V2D b;
    b(0) = direc1.dot(line2.start - line1.start);
    b(1) = direc2.dot(line1.start - line2.start);
    V2D res = A.inverse() * b;
    V3D point =
        0.5 * (line1.start + res[0] * direc1 + line2.start + res[1] * direc2);

    // add pointtrack
    PointTrack ptrack = PointTrack(point, image_id_list, p2d_id_list, p2d_list);
    int point3d_id = -1;
    if (point_tracks_.empty())
      point3d_id = 0;
    else
      point3d_id = point_tracks_.rbegin()->first + 1;
    point_tracks_.insert(std::make_pair(point3d_id, ptrack));
    points_.insert(std::make_pair(point3d_id, ptrack.p));

    // change the point3d_id for corresponding nodes in 2d bipartite
    for (auto it = p2d_data.begin(); it != p2d_data.end(); ++it) {
      int img_id = it->first;
      int p2d_id = it->second;
      Point2d point2d = all_bpt2ds_.at(img_id).point(p2d_id);
      point2d.point3D_id = point3d_id;
      all_bpt2ds_.at(img_id).update_point(p2d_id, point2d);
    }
  }
}

void GlobalAssociator::SetUp() {
  // reset problem
  problem_.reset(new ceres::Problem(config_.problem_options));

  // add residuals
  // R1.1: point geometric residual
  if (!config_.constant_point || !config_.constant_intrinsics ||
      !config_.constant_pose) {
    for (auto it = points_.begin(); it != points_.end(); ++it) {
      AddPointGeometricResiduals(it->first);
    }
  }
  // R1.2: line geometric residual
  if (!config_.constant_line || !config_.constant_intrinsics ||
      !config_.constant_pose) {
    for (auto it = lines_.begin(); it != lines_.end(); ++it) {
      AddLineGeometricResiduals(it->first);
    }
  }
  // R2.1: point line association residual
  if (enable_pointline_ &&
      (!config_.constant_point || !config_.constant_line)) {
    auto weights = construct_weights_pointline(config_.th_weight_pointline);
    AddPointLineAssociationResiduals(weights);
  }
  // R2.2: vp line association residual
  if (enable_vpline_ && (!config_.constant_vp || !config_.constant_line)) {
    auto weights = construct_weights_vpline(config_.th_count_vpline);
    AddVPLineAssociationResiduals(weights);
  }
  // R2.3: vp orthogonality residual
  if (enable_vpline_ && !config_.constant_vp) {
    AddVPOrthogonalityResiduals();
  }
  // R2.4: vp collinearity residual
  if (enable_vpline_ && !config_.constant_vp) {
    AddVPCollinearityResiduals();
  }

  // parameterize
  ParameterizeCameras();
  ParameterizePoints();
  ParameterizeLines();
  ParameterizeVPs();
}

void GlobalAssociator::AddPointLineAssociationResiduals(
    const std::map<std::pair<int, int>, double> &weights) {
  if (config_.lw_pointline_association <= 0)
    return;
  ceres::LossFunction *loss_function =
      config_.point_line_association_3d_loss_function.get();
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    double weight = it->second;

    ceres::CostFunction *cost_function =
        PointLineAssociation3dFunctor::Create();
    ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight * config_.lw_pointline_association,
        ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
        cost_function, scaled_loss_function, points_.at(point3d_id).data(),
        lines_.at(line3d_id).uvec.data(), lines_.at(line3d_id).wvec.data());
  }
}

void GlobalAssociator::AddVPLineAssociationResiduals(
    const std::map<std::pair<int, int>, int> &weights) {
  if (config_.lw_vpline_association <= 0)
    return;
  ceres::LossFunction *loss_function =
      config_.vp_line_association_3d_loss_function.get();
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int vp3d_id = it->first.first;
    int line3d_id = it->first.second;
    int weight = it->second;

    ceres::CostFunction *cost_function = VPLineAssociation3dFunctor::Create();
    ceres::LossFunction *scaled_loss_function = new ceres::ScaledLoss(
        loss_function, double(weight) * 1e2 * config_.lw_vpline_association,
        ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
        cost_function, scaled_loss_function, vps_.at(vp3d_id).data(),
        lines_.at(line3d_id).uvec.data(), lines_.at(line3d_id).wvec.data());
  }
}

void GlobalAssociator::AddVPOrthogonalityResiduals() {
  if (config_.lw_vp_orthogonality <= 0)
    return;
  ceres::LossFunction *loss_function =
      config_.vp_orthogonality_loss_function.get();
  auto pairs =
      construct_pairs_vp_orthogonality(vps_, config_.th_angle_orthogonality);
  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
    int id1 = it->first;
    int id2 = it->second;

    ceres::CostFunction *cost_function = VPOrthogonalityFunctor::Create();
    ceres::LossFunction *scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1e2 * config_.lw_vp_orthogonality,
                              ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id =
        problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                   vps_.at(id1).data(), vps_.at(id2).data());
  }
}

void GlobalAssociator::AddVPCollinearityResiduals() {
  if (config_.lw_vp_collinearity <= 0)
    return;
  ceres::LossFunction *loss_function =
      config_.vp_collinearity_loss_function.get();
  auto pairs =
      construct_pairs_vp_collinearity(vps_, config_.th_angle_collinearity);
  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
    int id1 = it->first;
    int id2 = it->second;

    ceres::CostFunction *cost_function = VPCollinearityFunctor::Create();
    ceres::LossFunction *scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1e2 * config_.lw_vp_collinearity,
                              ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id =
        problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                   vps_.at(id1).data(), vps_.at(id2).data());
  }
}

void GlobalAssociator::ParameterizeVPs() {
  for (auto it = vps_.begin(); it != vps_.end(); ++it) {
    double *vp_data = it->second.data();
    if (!problem_->HasParameterBlock(vp_data))
      continue;
    if (config_.constant_vp)
      problem_->SetParameterBlockConstant(vp_data);
    else
      SetSphereManifold<3>(problem_.get(), vp_data);
  }
}

double GlobalAssociator::dist2weight(const double &dist) const {
  // return expscore(dist, config_.th_pixel);
  if (dist <= config_.th_pixel)
    return 1.0;
  else
    return 0.0;
}

double GlobalAssociator::compute_overall_weight(
    const std::vector<std::pair<int, double>> &dists) const {
  // TODO: potential improvement here
  std::vector<double> values;
  for (auto it = dists.begin(); it != dists.end(); ++it) {
    double dist = it->second;
    values.push_back(dist2weight(dist));
  }
  std::sort(values.begin(), values.end());
  double weight = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    weight += values[i];
  }
  return weight;
}

std::map<std::pair<int, int>, double>
GlobalAssociator::construct_weights_pointline(const double th_weight) const {
  std::map<std::pair<int, int>, std::vector<std::pair<int, double>>>
      dists_collection; // (point3d_id, line3d_id), (img_id, dist)
  for (const auto &kv : line_tracks_) {
    int line3d_id = kv.first;
    const LineTrack &linetrack = line_tracks_.at(line3d_id);
    size_t n_supports = linetrack.count_lines();
    for (size_t support_id = 0; support_id < n_supports; ++support_id) {
      int img_id = linetrack.image_id_list[support_id];
      int line2d_id = linetrack.line_id_list[support_id];
      std::vector<int> point2d_ids =
          all_bpt2ds_.at(img_id).neighbor_points(line2d_id);
      for (const int &point2d_id : point2d_ids) {
        Point2d point = all_bpt2ds_.at(img_id).point(point2d_id);
        int point3d_id = point.point3D_id;
        if (point3d_id < 0)
          continue;

        // compute distance
        CameraView view = imagecols_.camview(img_id);
        V2D proj_point = point.p;
        Line2d proj_line = all_bpt2ds_.at(img_id).line(line2d_id);
        double dist = InfiniteLine2d(proj_line).point_distance(proj_point);

        // update dists collection
        std::pair<int, int> edge_index = std::make_pair(point3d_id, line3d_id);
        if (dists_collection.find(edge_index) == dists_collection.end())
          dists_collection.insert(std::make_pair(
              edge_index, std::vector<std::pair<int, double>>()));
        dists_collection[edge_index].push_back(std::make_pair(img_id, dist));
      }
    }
  }

  // construct weights
  std::map<std::pair<int, int>, double> weights;
  for (auto it = dists_collection.begin(); it != dists_collection.end(); ++it) {
    double weight = compute_overall_weight(it->second);
    if (weight < th_weight)
      continue;
    weights.insert(std::make_pair(it->first, weight));
  }
  return weights;
}

std::map<std::pair<int, int>, int>
GlobalAssociator::construct_weights_vpline(const int th_count) const {
  std::map<int, std::map<int, int>>
      edge_counter; // (line3d_id, (vp3d_id, count))
  for (const auto &kv : line_tracks_) {
    int line3d_id = kv.first;
    const LineTrack &linetrack = line_tracks_.at(line3d_id);
    size_t n_supports = linetrack.count_lines();
    for (size_t support_id = 0; support_id < n_supports; ++support_id) {
      int img_id = linetrack.image_id_list[support_id];
      int line2d_id = linetrack.line_id_list[support_id];
      std::vector<int> vp2d_ids =
          all_bpt2ds_vp_.at(img_id).neighbor_points(line2d_id);
      if (vp2d_ids.empty())
        continue;
      THROW_CHECK_EQ(vp2d_ids.size(), 1);
      int vp2d_id = vp2d_ids[0];
      vplib::VP2d vp2d = all_bpt2ds_vp_.at(img_id).point(vp2d_id);
      int vp3d_id = vp2d.point3D_id;
      if (vp3d_id < 0)
        continue;
      if (edge_counter.find(line3d_id) == edge_counter.end())
        edge_counter.insert(std::make_pair(line3d_id, std::map<int, int>()));
      if (edge_counter.at(line3d_id).find(vp3d_id) ==
          edge_counter.at(line3d_id).end())
        edge_counter.at(line3d_id).insert(std::make_pair(vp3d_id, 0));
      edge_counter.at(line3d_id).at(vp3d_id) += 1;
    }
  }

  // construct best assignments
  std::map<int, std::pair<int, int>>
      best_assignments; // (line3d_id, (best_vp3d_id, max_count))
  for (auto it = edge_counter.begin(); it != edge_counter.end(); ++it) {
    int line3d_id = it->first;
    int best_vp3d_id = -1;
    int best_count = 0;
    for (auto it2 = edge_counter.at(line3d_id).begin();
         it2 != edge_counter.at(line3d_id).end(); ++it2) {
      int vp3d_id = it2->first;
      int count = it2->second;
      if (count > best_count) {
        best_vp3d_id = vp3d_id;
        best_count = count;
      }
    }
    best_assignments.insert(
        std::make_pair(line3d_id, std::make_pair(best_vp3d_id, best_count)));
  }

  // construct weights
  std::map<std::pair<int, int>, int> weights;
  for (auto it = best_assignments.begin(); it != best_assignments.end(); ++it) {
    int line3d_id = it->first;
    int vp3d_id = it->second.first;
    int count = it->second.second;
    if (count < th_count)
      continue;
    weights.insert(std::make_pair(std::make_pair(vp3d_id, line3d_id), count));
  }
  return weights;
}

std::vector<std::pair<int, int>>
GlobalAssociator::construct_pairs_vp_orthogonality(
    const std::map<int, V3D> &vps, const double th_angle_orthogonality) const {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> vp3d_ids;
  for (auto it = vps.begin(); it != vps.end(); ++it) {
    vp3d_ids.push_back(it->first);
  }
  size_t n_vps = vp3d_ids.size();
  for (size_t i = 0; i < n_vps - 1; ++i) {
    int id1 = vp3d_ids[i];
    V3D vp1 = vps.at(id1);
    for (size_t j = i + 1; j < n_vps; ++j) {
      double id2 = vp3d_ids[j];
      V3D vp2 = vps.at(id2);
      double cosine = std::abs(vp1.dot(vp2));
      if (cosine > 1.0)
        cosine = 1.0;
      double angle = acos(cosine) * 180.0 / M_PI;
      if (angle < th_angle_orthogonality)
        continue;
      pairs.push_back(std::make_pair(id1, id2));
    }
  }
  return pairs;
}

std::vector<std::pair<int, int>>
GlobalAssociator::construct_pairs_vp_collinearity(
    const std::map<int, V3D> &vps, const double th_angle_collinearity) const {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> vp3d_ids;
  for (auto it = vps.begin(); it != vps.end(); ++it) {
    vp3d_ids.push_back(it->first);
  }
  size_t n_vps = vp3d_ids.size();
  for (size_t i = 0; i < n_vps - 1; ++i) {
    int id1 = vp3d_ids[i];
    V3D vp1 = vps.at(id1);
    for (size_t j = i + 1; j < n_vps; ++j) {
      double id2 = vp3d_ids[j];
      V3D vp2 = vps.at(id2);
      double cosine = std::abs(vp1.dot(vp2));
      if (cosine > 1.0)
        cosine = 1.0;
      double angle = acos(cosine) * 180.0 / M_PI;
      if (angle > th_angle_collinearity)
        continue;
      pairs.push_back(std::make_pair(id1, id2));
    }
  }
  return pairs;
}

bool GlobalAssociator::Solve() {
  if (problem_->NumParameterBlocks() == 0)
    return false;
  if (problem_->NumResiduals() == 0)
    return false;
  ceres::Solver::Options solver_options = config_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 900;
  const size_t num_images = imagecols_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else { // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (config_.print_summary) {
    colmap::PrintSolverSummary(summary_, "Optimization report");
  }
  return true;
}

std::map<int, V3D> GlobalAssociator::GetOutputVPs() const {
  std::map<int, V3D> outputs;
  for (auto it = vp_tracks_.begin(); it != vp_tracks_.end(); ++it) {
    int track_id = it->first;
    outputs.insert(std::make_pair(track_id, vps_.at(track_id)));
  }
  return outputs;
}

std::map<int, vplib::VPTrack> GlobalAssociator::GetOutputVPTracks() const {
  std::map<int, vplib::VPTrack> outputs;
  for (auto it = vp_tracks_.begin(); it != vp_tracks_.end(); ++it) {
    int track_id = it->first;
    vplib::VPTrack track = it->second;
    track.direction = vps_.at(track_id);
    outputs.insert(std::make_pair(track_id, track));
  }
  return outputs;
}

bool GlobalAssociator::test_linetrack_validity(const LineTrack &track) const {
  // set up line linker in 2d
  LineLinker2dConfig linker2d_config;
  LineLinker2d linker = LineLinker2d(linker2d_config);

  // check if at least half of the supports have inliers
  std::set<int> valid_images;
  for (size_t i = 0; i < track.count_lines(); ++i) {
    int img_id = track.image_id_list[i];
    Line2d line2d_proj = track.line.projection(imagecols_.camview(img_id));
    if (linker.check_connection(line2d_proj, track.line2d_list[i]))
      valid_images.insert(img_id);
  }
  if (valid_images.size() < 0.5 * track.count_images())
    return false;
  else
    return true;
}

structures::PL_Bipartite3d
GlobalAssociator::GetBipartite3d_PointLine_Constraints() const {
  // init bipartite
  structures::PL_Bipartite3d bpt;

  // get pointtracks
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    int point3d_id = it->first;
    PointTrack ptrack = PointTrack(point_tracks_.at(point3d_id));
    ptrack.p = points_.at(point3d_id);
    bpt.add_point(ptrack, point3d_id);
  }
  // get linetracks
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    int line3d_id = it->first;
    LineTrack ltrack = LineTrack(line_tracks_.at(line3d_id));
    InfiniteLine3d inf_line = lines_.at(line3d_id).GetInfiniteLine();
    std::vector<CameraView> views;
    for (size_t i = 0; i < ltrack.count_lines(); ++i) {
      views.push_back(imagecols_.camview(ltrack.image_id_list[i]));
    }
    Line3d line =
        GetLineSegmentFromInfiniteLine3d(inf_line, views, ltrack.line2d_list);
    ltrack.line = line;
    if (!test_linetrack_validity(ltrack))
      ltrack.line = line_tracks_.at(line3d_id).line;
    bpt.add_line(ltrack, line3d_id);
  }

  // build connections
  auto weights = construct_weights_pointline(config_.th_weight_pointline);
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    if (!bpt.exist_point(point3d_id) || !bpt.exist_line(line3d_id))
      continue;
    bpt.add_edge(point3d_id, line3d_id);
  }
  return bpt;
}

structures::PL_Bipartite3d GlobalAssociator::GetBipartite3d_PointLine() const {
  // init bipartite
  structures::PL_Bipartite3d bpt;

  // get pointtracks
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    int point3d_id = it->first;
    PointTrack ptrack = PointTrack(point_tracks_.at(point3d_id));
    ptrack.p = points_.at(point3d_id);
    bpt.add_point(ptrack, point3d_id);
  }
  // get linetracks
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    int line3d_id = it->first;
    LineTrack ltrack = LineTrack(line_tracks_.at(line3d_id));
    InfiniteLine3d inf_line = lines_.at(line3d_id).GetInfiniteLine();
    std::vector<CameraView> views;
    for (size_t i = 0; i < ltrack.count_lines(); ++i) {
      views.push_back(imagecols_.camview(ltrack.image_id_list[i]));
    }
    Line3d line =
        GetLineSegmentFromInfiniteLine3d(inf_line, views, ltrack.line2d_list);
    ltrack.line = line;
    if (!test_linetrack_validity(ltrack))
      ltrack.line = line_tracks_.at(line3d_id).line;
    bpt.add_line(ltrack, line3d_id);
  }

  // roughly estimate point uncertainty
  // TODO: use point covariance from jacobian
  std::map<int, double> point_uncertainties;
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    int point3d_id = it->first;
    const PointTrack &ptrack = point_tracks_.at(point3d_id);
    double min_uncertainty = std::numeric_limits<double>::max();
    for (size_t i = 0; i < ptrack.count_images(); ++i) {
      int img_id = ptrack.image_id_list[i];
      CameraView view = imagecols_.camview(img_id);
      double depth = view.pose.projdepth(ptrack.p);
      double uncertainty = view.cam.uncertainty(depth, 1.0);
      if (uncertainty < min_uncertainty)
        min_uncertainty = uncertainty;
    }
    point_uncertainties.insert(std::make_pair(point3d_id, min_uncertainty));
  }

  // build connections
  auto weights = construct_weights_pointline(config_.th_weight_pointline);
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    if (!bpt.exist_point(point3d_id) || !bpt.exist_line(line3d_id))
      continue;
    double dist =
        bpt.line(line3d_id).line.point_distance(bpt.point(point3d_id).p);
    // test on 3d
    if (dist > config_.th_hard_pl_dist3d * point_uncertainties.at(point3d_id))
      continue;
    bpt.add_edge(point3d_id, line3d_id);
  }
  return bpt;
}

structures::VPLine_Bipartite3d GlobalAssociator::GetBipartite3d_VPLine() const {
  // init bipartite
  structures::VPLine_Bipartite3d bpt;

  // get vptracks
  for (auto it = vps_.begin(); it != vps_.end(); ++it) {
    int vp3d_id = it->first;
    vplib::VPTrack track = vplib::VPTrack(vp_tracks_.at(vp3d_id));
    track.direction = vps_.at(vp3d_id);
    bpt.add_point(track, vp3d_id);
  }

  // get linetracks
  for (auto it = lines_.begin(); it != lines_.end(); ++it) {
    int line3d_id = it->first;
    LineTrack ltrack = LineTrack(line_tracks_.at(line3d_id));
    InfiniteLine3d inf_line = lines_.at(line3d_id).GetInfiniteLine();
    std::vector<CameraView> views;
    for (size_t i = 0; i < ltrack.count_lines(); ++i) {
      views.push_back(imagecols_.camview(ltrack.image_id_list[i]));
    }
    Line3d line =
        GetLineSegmentFromInfiniteLine3d(inf_line, views, ltrack.line2d_list);
    ltrack.line = line;
    if (!test_linetrack_validity(ltrack))
      ltrack.line = line_tracks_.at(line3d_id).line;
    bpt.add_line(ltrack, line3d_id);
  }

  // build connections
  auto weights = construct_weights_vpline(config_.th_count_vpline);
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int vp3d_id = it->first.first;
    int line3d_id = it->first.second;
    if (!bpt.exist_point(vp3d_id) || !bpt.exist_line(line3d_id))
      continue;
    // test on 3d
    double cosine = std::abs(
        bpt.point(vp3d_id).direction.dot(bpt.line(line3d_id).line.direction()));
    if (cosine > 1.0)
      cosine = 1.0;
    double angle = acos(cosine) * 180.0 / M_PI;
    if (angle > config_.th_hard_vpline_angle3d)
      continue;
    bpt.add_edge(vp3d_id, line3d_id);
  }

  // delete unassociated vp
  for (const int &vp3d_id : bpt.get_point_ids()) {
    if (bpt.pdegree(vp3d_id) == 0)
      bpt.delete_point(vp3d_id);
  }
  return bpt;
}

} // namespace global_pl_association

} // namespace optimize

} // namespace limap
