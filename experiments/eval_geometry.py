"""Geometry evaluation for structure triangulation outputs.

Subcommands:
  eth3d        — Evaluate point clouds against ETH3D ground truth laser scans.
  inlier       — Evaluate 3D lines against ground truth meshes or point clouds.
  mesh_points  — Evaluate COLMAP points against ground truth meshes
                 (inlier ratio).

Examples:
  # ETH3D point evaluation
  python experiments/eval_geometry.py eth3d \
      --model_dir outputs/test_eth3d/courtyard/final_model \
      --gt_mlp_path /path/to/courtyard/scan_alignment.mlp

  # ETH3D with line evaluation
  python experiments/eval_geometry.py eth3d \
      --model_dir outputs/test_eth3d/courtyard/final_model \
      --gt_mlp_path /path/to/courtyard/scan_alignment.mlp \
      --eval_lines

  # Compare multiple models on ETH3D
  python experiments/eval_geometry.py eth3d \
      --model_dirs model_A model_B \
      --labels baseline ours \
      --gt_mlp_path /path/to/scan_alignment.mlp \
      --eval_lines

  # Inlier evaluation against mesh (Hypersim)
  python experiments/eval_geometry.py inlier \
      --model_dir outputs/test_hypersim/ai_001_001/final_model \
      --gt_mesh /path/to/mesh.obj --mpau 0.0254

  # Inlier evaluation against point cloud
  python experiments/eval_geometry.py inlier \
      --model_dir outputs/.../final_model \
      --gt_ply /path/to/pointcloud.ply

  # Point inlier evaluation against mesh (Hypersim)
  python experiments/eval_geometry.py mesh_points \
      --model_dirs model_A model_B --labels groups lines_only \
      --gt_mesh /path/to/mesh.obj --mpau 0.0254
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_holistic_reconstruction(model_dir):
    """Load HolisticReconstruction from a model directory.

    Returns (holistic_recon, is_holistic) where is_holistic indicates whether
    the model contains structure data (lines/groups) beyond COLMAP points.
    """
    from limap.scene import HolisticReconstruction

    recon = HolisticReconstruction()
    recon.read(model_dir)
    is_holistic = recon.structure_recon.num_lines3D() > 0
    return recon, is_holistic


def load_model(model_dir):
    """Load a HolisticReconstruction, falling back to pycolmap.

    Returns (recon, lines) where:
      - recon: pycolmap.Reconstruction (point_recon component)
      - lines: list of Line3d objects (empty if no structure data)
    """
    try:
        holistic, is_holistic = load_holistic_reconstruction(model_dir)
        point_recon = holistic.point_recon
        lines = []
        if is_holistic:
            for line_id in holistic.structure_recon.lines3D:
                lines.append(holistic.structure_recon.lines3D[line_id])
        return point_recon, lines
    except Exception:
        import pycolmap

        recon = pycolmap.Reconstruction(model_dir)
        return recon, []


# ---------------------------------------------------------------------------
# PLY export helpers
# ---------------------------------------------------------------------------


def export_points_ply(point_recon, path):
    """Export COLMAP 3D points to PLY file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    point_recon.export_PLY(path)


def sample_line_points(lines, samples_per_meter=100):
    """Uniformly sample points along line segments.

    Args:
        lines: list of Line3d objects
        samples_per_meter: sampling density (points per meter of line length)

    Returns:
        np.ndarray of shape (N, 3)
    """
    all_points = []
    for line in lines:
        arr = line.as_array()  # (2, 3)
        start, end = arr[0], arr[1]
        length = line.length()
        n_samples = max(2, int(np.ceil(length * samples_per_meter)))
        ts = np.linspace(0.0, 1.0, n_samples)
        points = start[None, :] + ts[:, None] * (end - start)[None, :]
        all_points.append(points)
    if not all_points:
        return np.zeros((0, 3))
    return np.vstack(all_points)


def export_lines_ply(lines, path, samples_per_meter=100):
    """Sample points from lines and save as PLY."""
    import limap.util.io as limapio

    points = sample_line_points(lines, samples_per_meter)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    limapio.save_ply(path, points)
    return points.shape[0]


# ---------------------------------------------------------------------------
# ETH3D binary evaluation
# ---------------------------------------------------------------------------

DEFAULT_ETH3D_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference",
    "multi-view-evaluation",
    "build",
    "ETH3DMultiViewEvaluation",
)

DEFAULT_TOLERANCES = "0.01,0.02,0.05,0.1,0.2,0.5"


def run_eth3d_eval(ply_path, mlp_path, binary_path, tolerances_str):
    """Run the ETH3D multi-view evaluation binary and parse results.

    Args:
        ply_path: path to reconstruction PLY file
        mlp_path: path to ground truth scan_alignment.mlp
        binary_path: path to ETH3DMultiViewEvaluation binary
        tolerances_str: comma-separated tolerance values in meters

    Returns:
        dict with keys 'tolerances', 'completeness', 'accuracy', 'f1'
        each mapping to a list of floats, or None on failure.
    """
    if not os.path.isfile(binary_path):
        print(
            f"Error: ETH3D binary not found at {binary_path}", file=sys.stderr
        )
        return None

    cmd = [
        binary_path,
        "--reconstruction_ply_path",
        ply_path,
        "--ground_truth_mlp_path",
        mlp_path,
        "--tolerances",
        tolerances_str,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print("Error: ETH3D evaluation timed out", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(
            f"Error: ETH3D binary returned {result.returncode}", file=sys.stderr
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return None

    return parse_eth3d_output(result.stdout)


def parse_eth3d_output(stdout):
    """Parse ETH3D evaluation binary stdout.

    Expected format:
        Tolerances: 0.01 0.02 ...
        Completenesses: 0.123 0.234 ...
        Accuracies: 0.456 0.567 ...
        F1-scores: 0.345 0.456 ...
    """
    results = {}
    key_map = {
        "Tolerances:": "tolerances",
        "Completenesses:": "completeness",
        "Accuracies:": "accuracy",
        "F1-scores:": "f1",
    }

    for line in stdout.strip().split("\n"):
        line = line.strip()
        for prefix, key in key_map.items():
            if line.startswith(prefix):
                values = line[len(prefix) :].strip().split()
                results[key] = [float(v) for v in values]
                break

    if not all(
        k in results for k in ["tolerances", "completeness", "accuracy", "f1"]
    ):
        print(
            "Warning: Could not parse all ETH3D output fields", file=sys.stderr
        )
        print("Stdout was:", file=sys.stderr)
        print(stdout, file=sys.stderr)
        return None

    return results


def print_eth3d_results(all_results, tolerances, labels=None):
    """Print formatted ETH3D results table.

    Args:
        all_results: list of result dicts (one per model)
        tolerances: list of tolerance values
        labels: list of model labels (optional)
    """
    if labels is None:
        labels = [f"model_{i}" for i in range(len(all_results))]

    # Build header
    tol_strs = []
    for t in tolerances:
        if t < 0.01:
            tol_strs.append(f"{t * 1000:.0f}mm")
        else:
            tol_strs.append(f"{t * 100:.0f}cm")

    header_parts = ["Scene".ljust(20)]
    for ts in tol_strs:
        header_parts.extend(
            [f"Acc@{ts}".rjust(9), f"Cmp@{ts}".rjust(9), f"F1@{ts}".rjust(9)]
        )
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, res in zip(labels, all_results, strict=False):
        if res is None:
            parts = [label.ljust(20)]
            parts.extend(["    FAIL".rjust(9)] * (len(tolerances) * 3))
            print(" | ".join(parts))
            continue
        parts = [label.ljust(20)]
        for i in range(len(tolerances)):
            acc = res["accuracy"][i] * 100
            cmp = res["completeness"][i] * 100
            f1 = res["f1"][i] * 100
            parts.extend(
                [
                    f"{acc:8.2f}".rjust(9),
                    f"{cmp:8.2f}".rjust(9),
                    f"{f1:8.2f}".rjust(9),
                ]
            )
        print(" | ".join(parts))

    print(sep)


# ---------------------------------------------------------------------------
# Inlier ratio evaluation (against mesh or point cloud)
# ---------------------------------------------------------------------------


def create_evaluator(gt_mesh=None, gt_ply=None, mpau=1.0):
    """Create a MeshEvaluator or PointCloudEvaluator.

    Args:
        gt_mesh: path to GT .obj mesh file
        gt_ply: path to GT .ply point cloud file
        mpau: meters per arbitrary unit (for mesh scaling)

    Returns:
        evaluator instance
    """
    import limap.evaluation as limap_eval

    if gt_mesh is not None:
        return limap_eval.MeshEvaluator(gt_mesh, mpau)
    elif gt_ply is not None:
        import limap.util.io as limapio

        points = limapio.read_ply(gt_ply)
        evaluator = limap_eval.PointCloudEvaluator(points)
        evaluator.Build()
        return evaluator
    else:
        raise ValueError("Must provide either --gt_mesh or --gt_ply")


def eval_lines(lines, evaluator, thresholds):
    """Evaluate lines against ground truth using inlier ratios.

    Args:
        lines: list of Line3d objects
        evaluator: MeshEvaluator or PointCloudEvaluator
        thresholds: list of distance thresholds in meters

    Returns:
        dict with:
          'thresholds': list of thresholds
          'recall': correct length in meters per threshold
          'precision': correct length / total length per threshold
          'n_lines': number of lines
          'total_length': total line length in meters
    """
    if not lines:
        return {
            "thresholds": thresholds,
            "recall": [0.0] * len(thresholds),
            "precision": [0.0] * len(thresholds),
            "n_lines": 0,
            "total_length": 0.0,
        }

    lengths = np.array([line.length() for line in lines])
    total_length = float(lengths.sum())
    n_lines = len(lines)

    recalls = []
    precisions = []

    for threshold in thresholds:
        ratios = np.array(
            [evaluator.ComputeInlierRatio(line, threshold) for line in lines]
        )
        # Recall: correct length in meters
        correct_length = float((lengths * ratios).sum())
        # Precision: correct length / total length
        precision = correct_length / total_length if total_length > 0 else 0.0
        recalls.append(correct_length)
        precisions.append(precision)

    return {
        "thresholds": thresholds,
        "recall": recalls,
        "precision": precisions,
        "n_lines": n_lines,
        "total_length": total_length,
    }


def eval_points(point_recon, evaluator, thresholds):
    """Evaluate COLMAP 3D points against ground truth using inlier ratios.

    Args:
        point_recon: pycolmap.Reconstruction with points3D
        evaluator: MeshEvaluator or PointCloudEvaluator
        thresholds: list of distance thresholds in meters

    Returns:
        dict with 'thresholds', 'inlier_ratio', 'n_points', 'median_dist'
    """
    points3D = point_recon.points3D
    if not points3D:
        return {
            "thresholds": thresholds,
            "inlier_ratio": [0.0] * len(thresholds),
            "n_points": 0,
            "median_dist": 0.0,
        }

    coords = np.array([p.xyz for p in points3D.values()])
    dists = np.array([evaluator.ComputeDistPoint(c) for c in coords])

    inlier_ratios = []
    for threshold in thresholds:
        ratio = float(100.0 * np.mean(dists < threshold))
        inlier_ratios.append(ratio)

    return {
        "thresholds": thresholds,
        "inlier_ratio": inlier_ratios,
        "n_points": len(coords),
        "median_dist": float(np.median(dists)),
    }


def print_point_inlier_results(all_results, thresholds, labels=None):
    """Print formatted point inlier evaluation results table."""
    if labels is None:
        labels = [f"model_{i}" for i in range(len(all_results))]

    tol_strs = [f"{t * 1000:.0f}mm" for t in thresholds]

    header_parts = ["Model".ljust(20)]
    for ts in tol_strs:
        header_parts.append(f"Inl@{ts}".rjust(10))
    header_parts.extend(["#Points".rjust(10), "Med(mm)".rjust(10)])
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, res in zip(labels, all_results, strict=False):
        if res is None:
            parts = [label.ljust(20)]
            parts.extend(["     FAIL".rjust(10)] * (len(thresholds) + 2))
            print(" | ".join(parts))
            continue
        parts = [label.ljust(20)]
        for i in range(len(thresholds)):
            parts.append(f"{res['inlier_ratio'][i]:9.2f}%".rjust(10))
        parts.append(f"{res['n_points']:>10d}")
        parts.append(f"{res['median_dist'] * 1000:9.2f}".rjust(10))
        print(" | ".join(parts))

    print(sep)


def print_inlier_results(all_results, thresholds, labels=None):
    """Print formatted inlier evaluation results table.

    Recall is correct length in meters, precision is correct/total ratio.

    Args:
        all_results: list of result dicts (one per model)
        thresholds: list of threshold values
        labels: list of model labels
    """
    if labels is None:
        labels = [f"model_{i}" for i in range(len(all_results))]

    tol_strs = []
    for t in thresholds:
        tol_strs.append(f"{t * 1000:.0f}mm")

    header_parts = ["Scene".ljust(20)]
    for ts in tol_strs:
        header_parts.extend([f"R@{ts}(m)".rjust(10), f"P@{ts}".rjust(8)])
    header_parts.append("TotLen(m)".rjust(10))
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, res in zip(labels, all_results, strict=False):
        if res is None:
            parts = [label.ljust(20)]
            parts.extend(["     FAIL".rjust(10)] * (len(thresholds) * 2 + 1))
            print(" | ".join(parts))
            continue
        parts = [label.ljust(20)]
        for i in range(len(thresholds)):
            r = res["recall"][i]
            p = res["precision"][i]
            parts.extend([f"{r:9.1f}".rjust(10), f"{p * 100:6.1f}%".rjust(8)])
        parts.append(f"{res['total_length']:9.1f}".rjust(10))
        print(" | ".join(parts))

    print(sep)


# ---------------------------------------------------------------------------
# CLI: eth3d subcommand
# ---------------------------------------------------------------------------


def add_eth3d_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", type=str, help="Single model directory")
    group.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        help="Multiple model directories for comparison",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Display labels for comparison"
    )
    parser.add_argument(
        "--gt_mlp_path",
        type=str,
        required=True,
        help="Path to scan_alignment.mlp",
    )
    parser.add_argument(
        "--eval_binary",
        type=str,
        default=DEFAULT_ETH3D_BINARY,
        help="Path to ETH3D evaluation binary",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default=DEFAULT_TOLERANCES,
        help="Comma-separated tolerance values in meters",
    )
    parser.add_argument(
        "--eval_lines",
        action="store_true",
        help="Also evaluate sampled line points",
    )
    parser.add_argument(
        "--samples_per_meter",
        type=int,
        default=100,
        help="Line sampling density",
    )
    parser.add_argument(
        "--output_json", type=str, help="Path to save results as JSON"
    )


def cmd_eth3d(args):
    model_dirs = args.model_dirs if args.model_dirs else [args.model_dir]
    labels = (
        args.labels
        if args.labels
        else [os.path.basename(d.rstrip("/")) for d in model_dirs]
    )
    tolerances_str = args.tolerances
    tolerances = [float(t) for t in tolerances_str.split(",")]

    if len(labels) != len(model_dirs):
        print(
            "Error: number of labels must match number of model directories",
            file=sys.stderr,
        )
        sys.exit(1)

    all_point_results = []
    all_line_results = []

    for model_dir, label in zip(model_dirs, labels, strict=False):
        print(f"\nEvaluating: {label} ({model_dir})")

        point_recon, lines = load_model(model_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Evaluate points
            points_ply = os.path.join(tmpdir, "points.ply")
            export_points_ply(point_recon, points_ply)
            n_points = len(point_recon.points3D)
            print(f"  Points: {n_points}")

            point_res = run_eth3d_eval(
                points_ply, args.gt_mlp_path, args.eval_binary, tolerances_str
            )
            all_point_results.append(point_res)

            # Evaluate lines (if requested and available)
            if args.eval_lines:
                if lines:
                    lines_ply = os.path.join(tmpdir, "lines.ply")
                    n_line_points = export_lines_ply(
                        lines, lines_ply, args.samples_per_meter
                    )
                    print(
                        f"  Lines: {len(lines)}, "
                        f"sampled points: {n_line_points}"
                    )

                    line_res = run_eth3d_eval(
                        lines_ply,
                        args.gt_mlp_path,
                        args.eval_binary,
                        tolerances_str,
                    )
                    all_line_results.append(line_res)
                else:
                    print("  No lines found in model")
                    all_line_results.append(None)

    # Print results
    print("\n\nETH3D Geometry Evaluation — Points")
    print_eth3d_results(all_point_results, tolerances, labels)

    if args.eval_lines:
        print("\nETH3D Geometry Evaluation — Lines")
        print_eth3d_results(all_line_results, tolerances, labels)

    # Save JSON
    if args.output_json:
        output = {
            "tolerances": tolerances,
            "points": {
                label: res
                for label, res in zip(labels, all_point_results, strict=False)
            },
        }
        if args.eval_lines:
            output["lines"] = {
                label: res
                for label, res in zip(labels, all_line_results, strict=False)
            }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI: inlier subcommand
# ---------------------------------------------------------------------------


def add_inlier_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", type=str, help="Single model directory")
    group.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        help="Multiple model directories for comparison",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Display labels for comparison"
    )
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--gt_mesh", type=str, help="Path to GT .obj mesh")
    gt_group.add_argument(
        "--gt_ply", type=str, help="Path to GT .ply point cloud"
    )
    parser.add_argument(
        "--mpau",
        type=float,
        default=1.0,
        help="Meters per arbitrary unit (Hypersim: 0.0254)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.001,0.005,0.01",
        help="Comma-separated distance thresholds in meters",
    )
    parser.add_argument(
        "--output_json", type=str, help="Path to save results as JSON"
    )


def cmd_inlier(args):
    model_dirs = args.model_dirs if args.model_dirs else [args.model_dir]
    labels = (
        args.labels
        if args.labels
        else [os.path.basename(d.rstrip("/")) for d in model_dirs]
    )
    thresholds = [float(t) for t in args.thresholds.split(",")]

    if len(labels) != len(model_dirs):
        print(
            "Error: number of labels must match number of model directories",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building evaluator...")
    evaluator = create_evaluator(
        gt_mesh=args.gt_mesh, gt_ply=args.gt_ply, mpau=args.mpau
    )

    all_results = []

    for model_dir, label in zip(model_dirs, labels, strict=False):
        print(f"\nEvaluating: {label} ({model_dir})")

        _, lines = load_model(model_dir)
        if not lines:
            print("  No lines found in model")
            all_results.append(None)
            continue

        print(f"  Lines: {len(lines)}")
        res = eval_lines(lines, evaluator, thresholds)
        all_results.append(res)

    # Print results
    print("\nLine Inlier Evaluation")
    print_inlier_results(all_results, thresholds, labels)

    # Save JSON
    if args.output_json:
        output = {
            "thresholds": thresholds,
            "results": {
                label: res
                for label, res in zip(labels, all_results, strict=False)
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI: mesh_points subcommand
# ---------------------------------------------------------------------------


def add_mesh_points_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", type=str, help="Single model directory")
    group.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        help="Multiple model directories for comparison",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Display labels for comparison"
    )
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--gt_mesh", type=str, help="Path to GT .obj mesh")
    gt_group.add_argument(
        "--gt_ply", type=str, help="Path to GT .ply point cloud"
    )
    parser.add_argument(
        "--mpau",
        type=float,
        default=1.0,
        help="Meters per arbitrary unit (Hypersim: 0.0254)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.001,0.005,0.01",
        help="Comma-separated distance thresholds in meters",
    )
    parser.add_argument(
        "--output_json", type=str, help="Path to save results as JSON"
    )


def cmd_mesh_points(args):
    model_dirs = args.model_dirs if args.model_dirs else [args.model_dir]
    labels = (
        args.labels
        if args.labels
        else [os.path.basename(d.rstrip("/")) for d in model_dirs]
    )
    thresholds = [float(t) for t in args.thresholds.split(",")]

    if len(labels) != len(model_dirs):
        print(
            "Error: number of labels must match number of model directories",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building evaluator...")
    evaluator = create_evaluator(
        gt_mesh=args.gt_mesh, gt_ply=args.gt_ply, mpau=args.mpau
    )

    all_results = []

    for model_dir, label in zip(model_dirs, labels, strict=False):
        print(f"\nEvaluating: {label} ({model_dir})")

        point_recon, _ = load_model(model_dir)
        n_points = len(point_recon.points3D)
        if n_points == 0:
            print("  No points found in model")
            all_results.append(None)
            continue

        print(f"  Points: {n_points}")
        res = eval_points(point_recon, evaluator, thresholds)
        all_results.append(res)

    # Print results
    print("\nPoint Inlier Evaluation (against mesh)")
    print_point_inlier_results(all_results, thresholds, labels)

    # Save JSON
    if args.output_json:
        output = {
            "thresholds": thresholds,
            "results": {
                label: res
                for label, res in zip(labels, all_results, strict=False)
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Geometry evaluation for structure triangulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eth3d_parser = subparsers.add_parser(
        "eth3d", help="ETH3D binary evaluation"
    )
    add_eth3d_args(eth3d_parser)

    inlier_parser = subparsers.add_parser(
        "inlier", help="Line inlier ratio evaluation"
    )
    add_inlier_args(inlier_parser)

    mesh_points_parser = subparsers.add_parser(
        "mesh_points", help="Point inlier ratio against mesh"
    )
    add_mesh_points_args(mesh_points_parser)

    args = parser.parse_args()

    if args.command == "eth3d":
        cmd_eth3d(args)
    elif args.command == "inlier":
        cmd_inlier(args)
    elif args.command == "mesh_points":
        cmd_mesh_points(args)


if __name__ == "__main__":
    main()
