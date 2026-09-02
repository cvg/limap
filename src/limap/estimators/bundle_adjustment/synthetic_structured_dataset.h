#pragma once

#include "limap/scene/holistic_reconstruction.h"

namespace limap {
namespace estimators {

// Options controlling the synthetic structured scene geometry.
struct SyntheticStructuredDatasetOptions {
  // Camera setup
  int num_rigs = 1;            // number of sensor rigs
  int num_cameras_per_rig = 1; // camera models per rig (intrinsics)
  int num_frames_per_rig = 50; // frames per rig (poses)
  // num_images = num_rigs * num_cameras_per_rig * num_frames_per_rig
  int camera_width = 1024;
  int camera_height = 768;

  // Points
  int num_points3D = 1000;       // total 3D points
  int num_points_per_plane = 50; // points placed on each plane
  // num_lines_per_plane = num_points_per_plane / 5
  int point_track_length = 10;

  // Lines
  int num_lines3D = 250; // total 3D lines
  int line_track_length = 10;

  // Groups (planes only -- VPs omitted for simplicity)
  int num_planes = 6; // Manhattan-aligned planes with random offsets

  // Wireframe (junction points where lines meet)
  int num_wireframe_edges = 50;

  // Scene geometry
  double cube_half_extent = 3.0;     // geometry in [-3, 3]^3
  double camera_sphere_radius = 5.0; // cameras at radius 5
  double line_min_length = 0.5;
  double line_max_length = 3.0;
  int min_visible_cameras = 2; // discard features visible in < N cameras

  // Fraction of line_track_length shared by all lines on the same plane.
  // Junction points inherit these shared views, giving co-visibility of
  // (1 + plane_shared_track_ratio) / 2 with each connected line.
  double plane_shared_track_ratio = 0.5;
};

// Options controlling the noise added to the synthetic scene.
struct SyntheticStructuredNoiseOptions {
  double pose_translation_stddev = 0.01;
  double pose_rotation_stddev = 1.0; // degrees
  double point3D_stddev = 0.05;
  double point2D_stddev = 1.0;
  double line3D_endpoint_stddev = 0.05;
  double line2D_endpoint_stddev = 1.0;
  double plane_normal_stddev = 0.02; // radians
  double plane_offset_stddev = 0.05;
};

// Synthesize a structured scene with points, lines, planes, and wireframe.
// Fills |reconstruction| with cameras, images, point tracks, line tracks,
// group (plane) tracks, and wireframe edges.
void SynthesizeStructuredDataset(
    const SyntheticStructuredDatasetOptions &options,
    HolisticReconstruction *reconstruction);

// Add Gaussian noise to all parameters in |reconstruction|.
void SynthesizeStructuredNoise(const SyntheticStructuredNoiseOptions &options,
                               HolisticReconstruction *reconstruction);

} // namespace estimators
} // namespace limap
