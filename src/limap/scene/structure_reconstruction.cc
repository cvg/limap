#include "limap/scene/structure_reconstruction.h"
#include "limap/scene/structure_reconstruction_io_binary.h"
#include "limap/scene/structure_reconstruction_io_text.h"

#include <colmap/util/file.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <map>
#include <mutex>
#include <set>

namespace limap {

StructureReconstruction::StructureReconstruction(
    const colmap::Reconstruction &point_recon)
    : point_recon_(point_recon) {}

void StructureReconstruction::Load(const StructureDatabaseCache &cache) {
  structures2d_ = cache.Structures2d();
}

void StructureReconstruction::Read(const std::filesystem::path &path) {
  if (colmap::ExistsFile(path / "structures2d.bin") &&
      colmap::ExistsFile(path / "lines3D.bin") &&
      colmap::ExistsFile(path / "groups3D.bin") &&
      colmap::ExistsFile(path / "wireframe3D.bin")) {

    ReadBinary(path);

  } else if (colmap::ExistsFile(path / "structures2d.txt") &&
             colmap::ExistsFile(path / "lines3D.txt") &&
             colmap::ExistsFile(path / "groups3D.txt") &&
             colmap::ExistsFile(path / "wireframe3D.txt")) {

    ReadText(path);

  } else {
    LOG(FATAL_THROW)
        << "Expected structures2d, lines3D, groups3D, wireframe3D files "
        << "in either .bin or .txt at directory: " << path;
  }
}

void StructureReconstruction::Write(const std::filesystem::path &path) const {
  WriteBinary(path);
}

void StructureReconstruction::ReadText(const std::filesystem::path &path) {
  structures2d_.clear();
  lines3D_.clear();
  groups3D_.clear();
  wireframe_.Clear();

  ReadStructures2dText(*this, path / "structures2d.txt");
  ReadLines3DText(*this, path / "lines3D.txt");
  ReadGroups3DText(*this, path / "groups3D.txt");
  ReadWireframeText(*this, path / "wireframe3D.txt");
}

void StructureReconstruction::ReadBinary(const std::filesystem::path &path) {
  structures2d_.clear();
  lines3D_.clear();
  groups3D_.clear();
  wireframe_.Clear();

  ReadStructures2dBinary(*this, path / "structures2d.bin");
  ReadLines3DBinary(*this, path / "lines3D.bin");
  ReadGroups3DBinary(*this, path / "groups3D.bin");
  ReadWireframeBinary(*this, path / "wireframe3D.bin");
}

void StructureReconstruction::WriteText(
    const std::filesystem::path &path) const {
  THROW_CHECK(colmap::ExistsDir(path))
      << "Directory " << path << " does not exist.";
  WriteStructures2dText(*this, path / "structures2d.txt");
  WriteLines3DText(*this, path / "lines3D.txt");
  WriteGroups3DText(*this, path / "groups3D.txt");
  WriteWireframeText(*this, path / "wireframe3D.txt");
}

void StructureReconstruction::WriteBinary(
    const std::filesystem::path &path) const {
  THROW_CHECK(colmap::ExistsDir(path))
      << "Directory " << path << " does not exist.";
  WriteStructures2dBinary(*this, path / "structures2d.bin");
  WriteLines3DBinary(*this, path / "lines3D.bin");
  WriteGroups3DBinary(*this, path / "groups3D.bin");
  WriteWireframeBinary(*this, path / "wireframe3D.bin");
}

void StructureReconstruction::InitializeAllWireframes(double threshold) {
  LOG(INFO) << "Initializing wireframes for " << structures2d_.size()
            << " images (threshold=" << threshold << ")";

  // Collect all images that have both structure data and valid keypoints.
  // 2D wireframes only need keypoint positions and line segments — no 3D info
  // or registration required. We iterate over all images in the reconstruction
  // (not just registered ones) so wireframes are ready when images get
  // registered later during incremental SfM.
  struct ImageData {
    const colmap::Image *image;
    class Structure2d *structure2d;
  };
  std::vector<ImageData> tasks;
  tasks.reserve(structures2d_.size());
  int num_skipped = 0;
  for (auto &[img_id, structure2d] : structures2d_) {
    if (!point_recon_.ExistsImage(img_id))
      continue;
    const auto &image = point_recon_.Image(img_id);
    if (image.NumPoints2D() != structure2d.NumPoints()) {
      LOG(WARNING) << "Skipping image " << img_id << ": point_recon has "
                   << image.NumPoints2D() << " Point2D but structure_db has "
                   << structure2d.NumPoints();
      ++num_skipped;
      continue;
    }
    tasks.push_back({&image, &structure2d});
  }
  if (num_skipped > 0) {
    LOG(WARNING) << "Skipped " << num_skipped
                 << " images with mismatched point counts";
  }
  LOG(INFO) << "Creating 2D wireframes for " << tasks.size() << " images";

  // Parallel: compute wireframes per image (each writes to its own Structure2d)
  const int num_threads = colmap::GetEffectiveNumThreads(-1);
  colmap::ThreadPool pool(num_threads);
  for (auto &task : tasks) {
    pool.AddTask([task, threshold]() {
      std::vector<V2D> points;
      points.reserve(task.image->NumPoints2D());
      for (const auto &point : task.image->Points2D()) {
        points.push_back(point.xy);
      }
      const std::vector<Line2d> &lines = task.structure2d->Lines();
      std::shared_ptr<Wireframe2d> wf =
          CreateWireframe2d(points, lines, threshold);
      task.structure2d->SetWireframe(*wf);
    });
  }
  pool.Wait();
}

void StructureReconstruction::ConstructWireframe3dFrom2d(
    const WireframeVotingOptions &options) {
  // Vote accumulator: (point3D, line3D) -> (image_id -> max_weight from that
  // image)
  using VoteMap = std::map<std::pair<point3D_t, line3D_t>,
                           std::map<colmap::image_t, double>>;

  // Pre-collect pointers to avoid concurrent unordered_map::at() calls
  struct ImageData {
    colmap::image_t image_id;
    const colmap::Image *image;
    const class Structure2d *structure2d;
  };
  std::vector<ImageData> image_data;
  for (const auto &[image_id, structure2d] : structures2d_) {
    if (point_recon_.ExistsImage(image_id)) {
      image_data.push_back(
          {image_id, &point_recon_.Image(image_id), &structure2d});
    }
  }

  // Parallel: each thread builds a local vote map
  const int num_threads = colmap::GetEffectiveNumThreads(-1);
  std::mutex merge_mutex;
  VoteMap votes;

  colmap::ThreadPool pool(num_threads);
  for (const auto &data : image_data) {
    pool.AddTask([data, &votes, &merge_mutex]() {
      const colmap::image_t image_id = data.image_id;
      const class Structure2d &structure2d = *data.structure2d;
      const colmap::Image &image = *data.image;

      // Build thread-local votes for this image
      VoteMap local_votes;
      for (const auto &edge : structure2d.Wireframe().GetAllEdges()) {
        const point2D_t p2d = edge.point_idx;
        const line2D_t l2d = edge.line_idx;
        const double w = edge.w;

        if (p2d >= static_cast<point2D_t>(image.NumPoints2D())) {
          continue;
        }
        const colmap::Point2D &point2D = image.Point2D(p2d);
        if (!point2D.HasPoint3D()) {
          continue;
        }
        if (l2d >= static_cast<line2D_t>(structure2d.NumLines())) {
          continue;
        }
        const Line2d &line2D = structure2d.Line(l2d);
        if (!line2D.HasLine3D()) {
          continue;
        }

        const point3D_t p3d = point2D.point3D_id;
        const line3D_t l3d = line2D.line3D_id;

        auto &image_weights = local_votes[{p3d, l3d}];
        auto it = image_weights.find(image_id);
        if (it == image_weights.end()) {
          image_weights[image_id] = w;
        } else {
          it->second = std::max(it->second, w);
        }
      }

      // Merge into global votes
      std::lock_guard<std::mutex> lock(merge_mutex);
      for (auto &[edge_key, local_image_weights] : local_votes) {
        auto &global_image_weights = votes[edge_key];
        for (const auto &[img_id, w] : local_image_weights) {
          auto it = global_image_weights.find(img_id);
          if (it == global_image_weights.end()) {
            global_image_weights[img_id] = w;
          } else {
            it->second = std::max(it->second, w);
          }
        }
      }
    });
  }
  pool.Wait();

  // Clear existing 3D wireframe and add edges meeting thresholds
  wireframe_.Clear();
  for (const auto &[edge, image_weights] : votes) {
    const int num_votes = static_cast<int>(image_weights.size());
    // Sum the max weight from each image
    double weight_sum = 0.0;
    for (const auto &[img_id, w] : image_weights) {
      weight_sum += w;
    }
    if (num_votes >= options.min_num_votes &&
        weight_sum >= options.min_weight_sum) {
      wireframe_.AddEdge(edge.first, edge.second, weight_sum);
    }
  }

  LOG(INFO) << "Constructed Wireframe3d with " << wireframe_.CountEdges()
            << " edges from 2D associations";
}

} // namespace limap
