#include "limap/scene/holistic_reconstruction.h"

#include <colmap/util/file.h>
#include <colmap/util/logging.h>

namespace limap {

HolisticReconstruction::HolisticReconstruction()
    : structure_recon_(StructureReconstruction(point_recon_)) {}

HolisticReconstruction::HolisticReconstruction(
    const colmap::Reconstruction &point_recon)
    : point_recon_(point_recon), structure_recon_(point_recon_) {}

namespace {

bool ExistsStructureBinary(const std::filesystem::path &path) {
  return colmap::ExistsFile(path / "structures/structures2d.bin") &&
         colmap::ExistsFile(path / "structures/lines3D.bin") &&
         colmap::ExistsFile(path / "structures/groups3D.bin") &&
         colmap::ExistsFile(path / "structures/wireframe3D.bin");
}

bool ExistsStructureText(const std::filesystem::path &path) {
  return colmap::ExistsFile(path / "structures/structures2d.txt") &&
         colmap::ExistsFile(path / "structures/lines3D.txt") &&
         colmap::ExistsFile(path / "structures/groups3D.txt") &&
         colmap::ExistsFile(path / "structures/wireframe3D.txt");
}

bool ExistsColmapBinary(const std::filesystem::path &path) {
  return colmap::ExistsFile(path / "cameras.bin") &&
         colmap::ExistsFile(path / "images.bin") &&
         colmap::ExistsFile(path / "points3D.bin");
}

bool ExistsColmapText(const std::filesystem::path &path) {
  return colmap::ExistsFile(path / "cameras.txt") &&
         colmap::ExistsFile(path / "images.txt") &&
         colmap::ExistsFile(path / "points3D.txt");
}

} // namespace

bool HolisticReconstruction::ExistsModel(const std::filesystem::path &path) {
  // Check for COLMAP reconstruction files (binary or text)
  bool colmap_exists = ExistsColmapBinary(path) || ExistsColmapText(path);

  // Check for structure files (binary or text)
  bool structure_exists =
      ExistsStructureBinary(path) || ExistsStructureText(path);

  return colmap_exists && structure_exists;
}

void HolisticReconstruction::Read(const std::filesystem::path &path) {
  point_recon_.Read(path);

  const std::filesystem::path struct_path = path / "structures";

  if (ExistsStructureBinary(path)) {
    structure_recon_.ReadBinary(struct_path);
  } else if (ExistsStructureText(path)) {
    structure_recon_.ReadText(struct_path);
  }
}

void HolisticReconstruction::Write(const std::filesystem::path &path) const {
  WriteBinary(path);
}

void HolisticReconstruction::ReadText(const std::filesystem::path &path) {
  point_recon_.ReadText(path);

  const std::filesystem::path struct_path = path / "structures";
  if (ExistsStructureText(path)) {
    structure_recon_.ReadText(struct_path);
  }
}

void HolisticReconstruction::ReadBinary(const std::filesystem::path &path) {
  point_recon_.ReadBinary(path);

  const std::filesystem::path struct_path = path / "structures";
  if (ExistsStructureBinary(path)) {
    structure_recon_.ReadBinary(struct_path);
  }
}

void HolisticReconstruction::WriteText(
    const std::filesystem::path &path) const {
  THROW_CHECK(colmap::ExistsDir(path))
      << "Directory " << path << " does not exist.";

  point_recon_.WriteText(path);

  const std::filesystem::path struct_path = path / "structures";
  colmap::CreateDirIfNotExists(struct_path);

  structure_recon_.WriteText(struct_path);
}

void HolisticReconstruction::WriteBinary(
    const std::filesystem::path &path) const {
  THROW_CHECK(colmap::ExistsDir(path))
      << "Directory " << path << " does not exist.";

  point_recon_.WriteBinary(path);

  const std::filesystem::path struct_path = path / "structures";
  colmap::CreateDirIfNotExists(struct_path);

  structure_recon_.WriteBinary(struct_path);
}

const colmap::Reconstruction &HolisticReconstruction::PointRecon() const {
  return point_recon_;
}
colmap::Reconstruction &HolisticReconstruction::PointRecon() {
  return point_recon_;
}

const StructureReconstruction &HolisticReconstruction::StructureRecon() const {
  return structure_recon_;
}
StructureReconstruction &HolisticReconstruction::StructureRecon() {
  return structure_recon_;
}

colmap::Sim3d HolisticReconstruction::Normalize(const bool fixed_scale,
                                                const double extent,
                                                const double min_percentile,
                                                const double max_percentile,
                                                const bool use_images) {
  // Normalize points and cameras (COLMAP)
  const colmap::Sim3d tform = point_recon_.Normalize(
      fixed_scale, extent, min_percentile, max_percentile, use_images);

  // Apply the same transform to all line endpoints
  // Sim3d convention: x_new = scale * R * x_old + t
  for (auto &[_, line3d] : structure_recon_.Lines3D()) {
    line3d.start = tform * line3d.start;
    line3d.end = tform * line3d.end;
  }

  return tform;
}

bool HolisticReconstruction::CheckValidity() const {
  for (const auto &[img_id, image] : PointRecon().Images()) {
    size_t n_keypoints_from_point = image.NumPoints2D();
    size_t n_keypoints_from_structure =
        StructureRecon().Structure2d(img_id).NumPoints();
    if (n_keypoints_from_point != n_keypoints_from_structure) {
      LOG(INFO) << "Error: The number of points are not matched at Image "
                << img_id << ": (" << n_keypoints_from_point
                << " != " << n_keypoints_from_structure << ")";
      return false;
    }
  }
  return true;
}

} // namespace limap
