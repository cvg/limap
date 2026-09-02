"""Assert that a written reconstruction is non-empty.

The pipelines exit 0 on an empty result, so CI needs this to catch it.
"""

import argparse
import sys
from pathlib import Path

from limap.scene import HolisticReconstruction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="model directory to check")
    parser.add_argument(
        "--min-points", type=int, default=1, help="minimum 3D points"
    )
    parser.add_argument(
        "--min-lines", type=int, default=1, help="minimum 3D lines"
    )
    parser.add_argument(
        "--min-groups", type=int, default=0, help="minimum 3D groups"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not HolisticReconstruction.exists_model(args.model_dir):
        sys.exit(f"No reconstruction found in {args.model_dir}")

    recon = HolisticReconstruction()
    recon.read(args.model_dir)
    counts = {
        "points": (recon.point_recon.num_points3D(), args.min_points),
        "lines": (recon.structure_recon.num_lines3D(), args.min_lines),
        "groups": (recon.structure_recon.num_groups3D(), args.min_groups),
    }

    print(f"{args.model_dir}:")
    for name, (actual, minimum) in counts.items():
        print(f"  {name}: {actual} (min {minimum})")

    failures = [
        f"{name}: got {actual}, expected >= {minimum}"
        for name, (actual, minimum) in counts.items()
        if actual < minimum
    ]
    if failures:
        sys.exit(
            "Reconstruction is under-populated:\n  " + "\n  ".join(failures)
        )


if __name__ == "__main__":
    main()
