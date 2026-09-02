#pragma once

#include <filesystem>

#include <colmap/geometry/sim3.h>
#include <colmap/scene/reconstruction.h>
#include <unordered_map>

#include "limap/scene/structure_reconstruction.h"

namespace limap {

class HolisticReconstruction {
public:
  HolisticReconstruction();
  HolisticReconstruction(const colmap::Reconstruction &point_recon);

  // Read data from text or binary file. Prefer binary data if it exists.
  void Read(const std::filesystem::path &path);
  void Write(const std::filesystem::path &path) const;

  // Read data from binary/text file.
  void ReadText(const std::filesystem::path &path);
  void ReadBinary(const std::filesystem::path &path);

  // Write data from binary/text file.
  void WriteText(const std::filesystem::path &path) const;
  void WriteBinary(const std::filesystem::path &path) const;

  // Check if a complete holistic reconstruction exists at the given path.
  static bool ExistsModel(const std::filesystem::path &path);

  const colmap::Reconstruction &PointRecon() const;
  colmap::Reconstruction &PointRecon();

  const StructureReconstruction &StructureRecon() const;
  StructureReconstruction &StructureRecon();

  // Normalize the reconstruction (points, cameras, AND lines).
  // Delegates to COLMAP's Normalize for points/cameras, then applies
  // the same Sim3d transform to all line endpoints.
  colmap::Sim3d Normalize(bool fixed_scale = false, double extent = 10.0,
                          double min_percentile = 0.1,
                          double max_percentile = 0.9, bool use_images = true);

  // Check if the number of keypoints on each image matches across
  // reconstructions
  bool CheckValidity() const;

private:
  // Underlying colmap reconstruction
  colmap::Reconstruction point_recon_;

  // Structural reconstruction
  StructureReconstruction structure_recon_;
};

} // namespace limap
