#include "sfm/sfm_model.h"

#include <colmap/util/math.h>

namespace limap {

namespace sfm {

colmap::mvs::Image CreateSfmImage(const std::string& filename, 
                                  const int width, const int height, 
                                  const std::vector<double>& K, 
                                  const std::vector<double>& R, 
                                  const std::vector<double>& T) 
{
    std::vector<float> K_float(K.begin(), K.end());
    std::vector<float> R_float(R.begin(), R.end());
    std::vector<float> T_float(T.begin(), T.end());
    return colmap::mvs::Image(filename, width, height, K_float.data(), R_float.data(), T_float.data());
}

void SfmModel::addPoint(double x, double y, double z, const std::vector<int>& image_ids) {
    Point p;
    p.x = x; p.y = y; p.z = z;
    p.track = image_ids;
    points.push_back(p);
}

void SfmModel::addImage(const colmap::mvs::Image& image) {
    images.push_back(image);
}

std::vector<int> SfmModel::ComputeNumPoints() const {
  std::vector<int> num_points(images.size(), 0);
  for (const auto& point : points) {
    for (size_t i = 0; i < point.track.size(); ++i) {
      const int image_idx = point.track[i];
      num_points.at(image_idx) += 1;
    }
  }
  return num_points;
}

std::vector<std::string> SfmModel::GetImageNames() const {
    std::vector<std::string> image_names;
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        image_names.push_back(GetImageName(image_idx));
    }
    return image_names;
}

// The original SfmModel::GetMaxOverlappingImages in COLMAP uses the number of intersection rather than IoU
std::vector<std::vector<int>> SfmModel::GetMaxIoUImages(
    const size_t num_images, const double min_triangulation_angle) const {
  std::vector<std::vector<int>> overlapping_images(images.size());

  const float min_triangulation_angle_rad = colmap::DegToRad(min_triangulation_angle);

  const float kTriangulationAnglePercentile = 75;
  const auto triangulation_angles =
      ComputeTriangulationAngles(kTriangulationAnglePercentile);

  const std::vector<std::map<int, int>> shared_num_points = ComputeSharedPoints();
  const std::vector<int> num_points = ComputeNumPoints();

#pragma omp parallel for
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const std::map<int, int>& shared_images = shared_num_points.at(image_idx);
    const int& num_point_1 = num_points.at(image_idx);
    const auto& overlapping_triangulation_angles =
        triangulation_angles.at(image_idx);

    std::vector<std::pair<int, double>> ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      if (overlapping_triangulation_angles.at(image.first) >=
          min_triangulation_angle_rad) {
        int v_intersection = image.second;
        const int& num_point_2 = num_points.at(image.first);
        int v_union = num_point_1 + num_point_2 - v_intersection;
        double IoU = double(v_intersection) / double(v_union);
        ordered_images.emplace_back(image.first, IoU);
      }
    }

    const size_t eff_num_images = std::min(ordered_images.size(), num_images);
    if (eff_num_images < shared_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, double> image1,
                           const std::pair<int, double> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, double> image1,
                   const std::pair<int, double> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx].push_back(ordered_images[i].first);
    }
  }

  return overlapping_images;
}

std::vector<std::vector<int>> SfmModel::GetMaxDiceCoeffImages(
    const size_t num_images, const double min_triangulation_angle) const {
  std::vector<std::vector<int>> overlapping_images(images.size());

  const float min_triangulation_angle_rad = colmap::DegToRad(min_triangulation_angle);

  const float kTriangulationAnglePercentile = 75;
  const auto triangulation_angles =
      ComputeTriangulationAngles(kTriangulationAnglePercentile);

  const std::vector<std::map<int, int>> shared_num_points = ComputeSharedPoints();
  const std::vector<int> num_points = ComputeNumPoints();

#pragma omp parallel for
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const std::map<int, int>& shared_images = shared_num_points.at(image_idx);
    const int& num_point_1 = num_points.at(image_idx);
    const auto& overlapping_triangulation_angles =
        triangulation_angles.at(image_idx);

    std::vector<std::pair<int, double>> ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      if (overlapping_triangulation_angles.at(image.first) >=
          min_triangulation_angle_rad) {
        int v_intersection = image.second;
        const int& num_point_2 = num_points.at(image.first);
        int v_union = num_point_1 + num_point_2 - v_intersection;
        double dicecoeff = double(2 * v_intersection) / double(v_union + v_intersection);
        ordered_images.emplace_back(image.first, dicecoeff);
      }
    }

    const size_t eff_num_images = std::min(ordered_images.size(), num_images);
    if (eff_num_images < shared_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, double> image1,
                           const std::pair<int, double> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, double> image1,
                   const std::pair<int, double> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx].push_back(ordered_images[i].first);
    }
  }

  return overlapping_images;
}

std::pair<float, float> SfmModel::get_robust_range(std::vector<float>& data, const std::pair<double, double>& range_robust, const double& kstretch) const {
    std::pair<float, float> range;

    std::sort(data.begin(), data.end());
    const float kMinPercentile = range_robust.first;
    const float kMaxPercentile = range_robust.second;
    range.first = data[data.size() * kMinPercentile];
    range.second = data[data.size() * kMaxPercentile];

    const float kStretchRatio = kstretch;
    float diff  = range.second - range.first;
    range.first -= kStretchRatio * diff;
    range.second += kStretchRatio * diff;
    return range;
}

std::pair<V3D, V3D> SfmModel::ComputeRanges(const std::pair<double, double>& range_robust, const double& kstretch) const
{
    std::vector<float> xdata, ydata, zdata;
    for (const auto& point: points) {
        xdata.push_back(point.x);
        ydata.push_back(point.y);
        zdata.push_back(point.z);
    }

    auto range_x = get_robust_range(xdata, range_robust, kstretch);
    auto range_y = get_robust_range(ydata, range_robust, kstretch);
    auto range_z = get_robust_range(zdata, range_robust, kstretch);
    return std::make_pair(V3D(range_x.first, range_y.first, range_z.first), V3D(range_x.second, range_y.second, range_z.second));
}

} // namespace sfm

} // namespace limap

