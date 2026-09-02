"""Dataset loaders for the SfM benchmark.

Every loader turns a scene into a common :class:`Scene` record so that
``benchmark_sfm.py`` can treat all datasets identically: an image directory,
an unposed input reconstruction to seed the pipeline, the image ids to
evaluate over, and (where it exists) the ground-truth pose per image.

Every dataset is read straight from its own public release -- there is no
derived workspace to build first. Views that need undistorting or resizing
are prepared once and cached under the output tree.

Supported datasets, and the release each expects at --data_dir:
    hypersim   Hypersim, GT poses from the release
    scannetpp  DA3-BENCH (depth-anything/DA3-BENCH on HuggingFace)
    7scenes    DA3-BENCH, same root as scannetpp
    eth3d      ETH3D DSLR, undistorted release
    1dsfm      1DSfM internet photo collections; retrieval-based, no GT
"""

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np
import pycolmap
from PIL import Image
from pycolmap import logging

# Hypersim's loader lives with its runner, not in the limap package.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "runners" / "hypersim")
)
from Hypersim import Hypersim  # noqa: E402
from loader import read_scene_hypersim  # noqa: E402

# ---------------------------------------------------------------------------
# Scene lists
# ---------------------------------------------------------------------------
HYPERSIM_SCENES = [f"ai_001_{i:03d}" for i in range(1, 9)]

SCANNETPP_SCENES = [
    "09c1414f1b",
    "1ada7a0617",
    "21d970d8de",
    "286b55a2bf",
    "38d58a7a31",
    "3e8bba0176",
    "40aec5fffa",
    "578511c8a9",
    "5f99900f09",
    "7831862f02",
    "7bc286c1b6",
    "9071e139d9",
    "acd95847c5",
    "bcd2436daf",
    "bde1e479ad",
    "c4c04e6d6c",
    "c5439f4607",
    "cc5237fd77",
    "f3d64c30f8",
    "fb5a96b1a2",
]

ETH3D_SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrains",
]

SEVENSCENES_SCENES = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
]

# 1DSfM.
ONEDSFM_SCENES = [
    "Alamo",  # 2915 images
    "Ellis_Island",  # 2587 images
    "Gendarmenmarkt",  # 1463 images
    "Madrid_Metropolis",  # 1344 images
    "Montreal_Notre_Dame",  # 2298 images
    "NYC_Library",  # 2550 images
    "Piazza_del_Popolo",  # 2251 images
    "Roman_Forum",  # 2364 images
    "Tower_of_London",  # 1576 images
]

# 1DSfM is expensive; default to the three smallest scenes.
ONEDSFM_DEFAULT_SCENES = [
    "Madrid_Metropolis",
    "Gendarmenmarkt",
    "Tower_of_London",
]


@dataclass
class Scene:
    """One benchmark scene, in the form the runner needs."""

    dataset: str
    scene_id: str
    image_dir: Path
    # Unposed reconstruction (cameras + images, no poses) seeding the pipeline.
    recon: pycolmap.Reconstruction
    # Image ids the evaluation iterates over.
    index_list: list[int]
    # img_id -> (R, t) of cam_from_world. None for datasets without GT.
    gt_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
    # Called lazily, only when the frontend actually runs, since retrieval is
    # expensive and derived methods reuse the frontend database.
    # Returns (neighbors or None for exhaustive, seconds spent).
    neighbors_fn: Callable[[Path], tuple[dict | None, float]] | None = None

    @property
    def has_gt(self) -> bool:
        return self.gt_poses is not None


def _poses_from_recon(recon, index_list):
    """Extract cam_from_world (R, t) per image id."""
    poses = {}
    for img_id in index_list:
        mat = np.asarray(recon.images[img_id].cam_from_world().matrix())
        poses[img_id] = (mat[:3, :3], mat[:3, 3])
    return poses


# ---------------------------------------------------------------------------
# Hypersim
# ---------------------------------------------------------------------------
class HypersimLoader:
    """Native Hypersim trajectories with GT poses from the release."""

    name = "hypersim"
    default_scenes = HYPERSIM_SCENES

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.cam_id = cfg.get("cam_id", 0)
        self.dataset = Hypersim(cfg["data_dir"])

    def _index_list(self, scene_id):
        self.dataset.set_scene_id(scene_id)
        index_list = np.arange(
            0, self.cfg["input_n_views"], self.cfg["input_stride"]
        ).tolist()
        return self.dataset.filter_index_list(index_list, cam_id=self.cam_id)

    def _copy_images(self, scene_id, index_list):
        """Copy scene images into the output tree (once)."""
        dst_dir = self.output_dir / self.name / scene_id / "images"
        if dst_dir.exists():
            return dst_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_id in index_list:
            src = Path(self.dataset.load_imname(img_id, cam_id=self.cam_id))
            shutil.copy2(src, dst_dir / src.name)
        logging.info(f"Copied {len(index_list)} images to {dst_dir}")
        return dst_dir

    def load(self, scene_id, with_images=True):
        index_list = self._index_list(scene_id)
        image_dir = (
            self._copy_images(scene_id, index_list)
            if with_images
            else self.output_dir / self.name / scene_id / "images"
        )

        _, recon, _ = read_scene_hypersim(
            self.cfg,
            self.dataset,
            scene_id,
            cam_id=self.cam_id,
            load_depth=False,
            load_poses=False,
            uncalibrated=self.cfg.get("uncalibrated", False),
            per_image_cameras=self.cfg.get("per_image_cameras", False),
        )

        Ts, Rs = self.dataset.load_cameras(cam_id=self.cam_id)
        gt_poses = {i: (Rs[i], Ts[i]) for i in index_list}

        return Scene(
            dataset=self.name,
            scene_id=scene_id,
            image_dir=image_dir,
            recon=recon,
            index_list=index_list,
            gt_poses=gt_poses,
        )


# ---------------------------------------------------------------------------
# View manifests
# ---------------------------------------------------------------------------
# A prepared scene records exactly which source frames it used, in order, so a
# tree is self-describing: the selection rule can change without silently
# invalidating trees built under the old one. Image id i always refers to
# manifest entry i, which is the frame prepared as f"{i:06d}.png".
VIEWS_MANIFEST = "views.json"


def manifest_path(output_dir, dataset, scene_id):
    return Path(output_dir) / dataset / scene_id / VIEWS_MANIFEST


def read_manifest(output_dir, dataset, scene_id):
    """Source frame names this scene was prepared from, or None."""
    path = manifest_path(output_dir, dataset, scene_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())["views"]


def write_manifest(output_dir, dataset, scene_id, names):
    path = manifest_path(output_dir, dataset, scene_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"dataset": dataset, "scene": scene_id, "views": list(names)},
            indent=1,
        )
    )
    logging.info(f"[{dataset}/{scene_id}] wrote {path}")


def _select_views(
    output_dir, dataset, scene_id, ordered, target_views, name_of=lambda x: x
):
    """Apply the manifest if there is one, else subsample and record it.

    Warns rather than guessing silently when a tree already holds prepared
    images but no manifest: that tree was built by something else, and a
    re-derived selection may not be the one its reconstructions used.
    """
    names = read_manifest(output_dir, dataset, scene_id)
    if names is not None:
        by_name = {name_of(x): x for x in ordered}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(
                f"[{dataset}/{scene_id}] {len(missing)} frame(s) in "
                f"{VIEWS_MANIFEST} are absent from the source release, "
                f"e.g. {missing[:3]}"
            )
        return [by_name[n] for n in names]

    selected = _subsample(ordered, target_views)
    image_dir = Path(output_dir) / dataset / scene_id / "images"
    if image_dir.exists() and any(image_dir.iterdir()):
        logging.warning(
            f"[{dataset}/{scene_id}] prepared images exist but no "
            f"{VIEWS_MANIFEST}; re-deriving the selection. If this tree was "
            f"built elsewhere the frames may not correspond -- recover it "
            f"with: python experiments/sfm_datasets.py recover-views --help"
        )
    else:
        write_manifest(
            output_dir, dataset, scene_id, [name_of(x) for x in selected]
        )
    return selected


# ---------------------------------------------------------------------------
# Shared scene construction
# ---------------------------------------------------------------------------
def _build_scene(dataset, scene_id, image_dir, cameras, entries):
    """Assemble a Scene from prepared images.

    Args:
        cameras: {camera_id: pycolmap.Camera} for the prepared images.
        entries: ordered [(file_name, camera_id, cam_from_world)]. Image ids
            are assigned 1..N in this order, so sorted-by-id == this order.

    Returns a Scene whose `recon` is unposed and whose `gt_poses` carry the
    ground truth.
    """
    blank = pycolmap.Reconstruction()
    for cam in cameras.values():
        blank.add_camera(cam)
    for cam_id, cam in cameras.items():
        rig = pycolmap.Rig(rig_id=cam_id)
        rig.add_ref_sensor(cam.sensor_id)
        blank.add_rig(rig)

    gt_poses = {}
    index_list = []
    for idx, (name, cam_id, cam_from_world) in enumerate(entries):
        image_id = idx + 1
        img = pycolmap.Image(
            name=name,
            camera_id=cam_id,
            image_id=image_id,
            frame_id=image_id,
        )
        frame = pycolmap.Frame(frame_id=image_id, rig_id=cam_id)
        frame.add_data_id(img.data_id)
        blank.add_frame(frame)
        blank.add_image(img)

        mat = np.asarray(cam_from_world.matrix())
        gt_poses[image_id] = (mat[:3, :3], mat[:3, 3])
        index_list.append(image_id)

    return Scene(
        dataset=dataset,
        scene_id=scene_id,
        image_dir=Path(image_dir),
        recon=blank,
        index_list=index_list,
        gt_poses=gt_poses,
    )


def _subsample(items, target_views):
    """Uniformly sample `target_views` items across the sequence, in order."""
    if not target_views or len(items) <= target_views:
        return items
    idx = np.linspace(0, len(items) - 1, target_views).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def _mark_prior_focal(cam, uncalibrated):
    cam.has_prior_focal_length = not uncalibrated
    return cam


# ---------------------------------------------------------------------------
# ScanNet++ (DA3-BENCH)
# ---------------------------------------------------------------------------
class ScanNetppLoader:
    """ScanNet++ scenes from the DA3-BENCH release.

    Reads the merged GT model directly out of
    `<root>/scannetpp/<scene>/merge_dslr_iphone/`, keeps one camera's frames
    (iphone by default -- its principal point is exactly centred, unlike the
    fitted render_rgb one), subsamples them temporally, and undistorts with
    COLMAP's own `undistort_camera`, capped at `max_dim`. The prepared views
    are cached under the output tree, so this is paid once per scene.

    Note this does not reproduce the exact view set of the March 2026 runs,
    which were not a uniform temporal subsample.
    """

    name = "scannetpp"
    default_scenes = SCANNETPP_SCENES

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.data_dir = Path(cfg["data_dir"]).expanduser()
        self.image_set = cfg.get("scannetpp_image_set", "iphone")
        self.target_views = cfg.get("target_views", 100)
        self.max_dim = cfg.get("benchmark_max_dim", 800)

    def load(self, scene_id, with_images=True):
        scene_dir = self.data_dir / self.name / scene_id / "merge_dslr_iphone"
        gt_dir = scene_dir / "colmap" / "sparse_render_rgb"
        if not gt_dir.exists():
            raise FileNotFoundError(f"GT reconstruction not found: {gt_dir}")

        gt_recon = pycolmap.Reconstruction(gt_dir)

        # Keep one image set. Its frames all share a single source camera.
        selected = [
            im
            for im in gt_recon.images.values()
            if im.name.startswith(f"{self.image_set}/")
        ]
        if not selected:
            raise FileNotFoundError(
                f"No '{self.image_set}/' images in {gt_dir}. "
                f"Try --scannetpp_image_set render_rgb."
            )
        # frame_NNNNNN.jpg sorts temporally.
        selected.sort(key=lambda im: im.name)
        selected = _select_views(
            self.output_dir,
            self.name,
            scene_id,
            selected,
            self.target_views,
            name_of=lambda im: im.name,
        )

        src_cam_ids = {im.camera_id for im in selected}
        if len(src_cam_ids) != 1:
            raise ValueError(
                f"Expected one camera for '{self.image_set}', "
                f"got {sorted(src_cam_ids)}"
            )
        src_cam = gt_recon.cameras[next(iter(src_cam_ids))]

        opts = pycolmap.UndistortCameraOptions()
        opts.max_image_size = self.max_dim
        und_cam = pycolmap.undistort_camera(opts, src_cam)
        und_cam = pycolmap.Camera(
            model="PINHOLE",
            width=und_cam.width,
            height=und_cam.height,
            params=list(und_cam.params),
            camera_id=1,
        )
        _mark_prior_focal(und_cam, self.cfg.get("uncalibrated", False))

        image_dir = self.output_dir / self.name / scene_id / "images"
        entries = []
        for idx, im in enumerate(selected):
            out_name = f"{idx:06d}.png"
            entries.append((out_name, 1, im.cam_from_world()))
            out_path = image_dir / out_name
            if not with_images or out_path.exists():
                continue
            image_dir.mkdir(parents=True, exist_ok=True)
            src_path = scene_dir / "images" / im.name
            bitmap = pycolmap.Bitmap.read(src_path, True)
            if bitmap is None:
                raise FileNotFoundError(f"Cannot read image: {src_path}")
            und_bitmap, _ = pycolmap.undistort_image(opts, bitmap, src_cam)
            und_bitmap.write(out_path)

        logging.info(
            f"[{self.name}/{scene_id}] {len(entries)} views "
            f"({self.image_set}) at {und_cam.width}x{und_cam.height}"
        )
        return _build_scene(
            self.name, scene_id, image_dir, {1: und_cam}, entries
        )


# ---------------------------------------------------------------------------
# 7Scenes (DA3-BENCH)
# ---------------------------------------------------------------------------
class SevenScenesLoader:
    """7Scenes from the DA3-BENCH release.

    Frames live in `<root>/7scenes/7Scenes/<scene>/seq-NN/` as
    `frame-NNNNNN.{color.png,pose.txt}`. 7Scenes ships a single calibrated
    SIMPLE_PINHOLE camera (f=525, 640x480) with no distortion, so the frames
    are used as-is and only the pose file needs converting: it stores
    camera-to-world, while COLMAP wants world-to-camera.
    """

    name = "7scenes"
    default_scenes = SEVENSCENES_SCENES
    FOCAL = 525.0

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.data_dir = Path(cfg["data_dir"]).expanduser()
        self.target_views = cfg.get("target_views", 100)

    def load(self, scene_id, with_images=True):
        scene_dir = self.data_dir / self.name / "7Scenes" / scene_id
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {scene_dir}")

        frames = sorted(scene_dir.glob("seq-*/frame-*.color.png"))
        if not frames:
            raise FileNotFoundError(f"No color frames under {scene_dir}")
        frames = _select_views(
            self.output_dir,
            self.name,
            scene_id,
            frames,
            self.target_views,
            name_of=lambda p: str(p.relative_to(scene_dir)),
        )

        with Image.open(frames[0]) as im:
            width, height = im.size
        cam = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[self.FOCAL, width / 2.0, height / 2.0],
            camera_id=1,
        )
        _mark_prior_focal(cam, self.cfg.get("uncalibrated", False))

        image_dir = self.output_dir / self.name / scene_id / "images"
        entries = []
        for idx, frame in enumerate(frames):
            pose_path = frame.with_name(
                frame.name.replace(".color.png", ".pose.txt")
            )
            cam_to_world = np.loadtxt(pose_path)
            world_to_cam = np.linalg.inv(cam_to_world)
            cam_from_world = pycolmap.Rigid3d(world_to_cam[:3, :4])

            out_name = f"{idx:06d}.png"
            entries.append((out_name, 1, cam_from_world))
            out_path = image_dir / out_name
            if not with_images or out_path.exists():
                continue
            image_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame, out_path)

        logging.info(
            f"[{self.name}/{scene_id}] {len(entries)} views at {width}x{height}"
        )
        return _build_scene(self.name, scene_id, image_dir, {1: cam}, entries)


# ---------------------------------------------------------------------------
# ETH3D
# ---------------------------------------------------------------------------
class Eth3dLoader:
    """ETH3D DSLR scenes, from the undistorted release.

    `<root>/<scene>/dslr_calibration_undistorted/` is already a COLMAP model
    with PINHOLE cameras, and the matching images are under
    `<root>/<scene>/images/`. The only preparation is downscaling to
    `max_dim`, with intrinsics scaled to match.
    """

    name = "eth3d"
    default_scenes = ETH3D_SCENES

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.data_dir = Path(cfg["data_dir"]).expanduser()
        self.target_views = cfg.get("target_views", 0)  # 0 == keep all
        self.max_dim = cfg.get("benchmark_max_dim", 800)

    def load(self, scene_id, with_images=True):
        scene_dir = self.data_dir / scene_id
        gt_dir = scene_dir / "dslr_calibration_undistorted"
        if not gt_dir.exists():
            raise FileNotFoundError(f"GT reconstruction not found: {gt_dir}")

        gt_recon = pycolmap.Reconstruction(gt_dir)
        selected = sorted(gt_recon.images.values(), key=lambda im: im.name)
        selected = _select_views(
            self.output_dir,
            self.name,
            scene_id,
            selected,
            self.target_views,
            name_of=lambda im: im.name,
        )

        # Scale every camera the selected images use.
        cameras, scales = {}, {}
        for cam_id in sorted({im.camera_id for im in selected}):
            src = gt_recon.cameras[cam_id]
            scale = min(1.0, self.max_dim / max(src.width, src.height))
            width = int(round(src.width * scale))
            height = int(round(src.height * scale))
            fx, fy, cx, cy = src.params[:4]
            cam = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=[fx * scale, fy * scale, cx * scale, cy * scale],
                camera_id=cam_id,
            )
            _mark_prior_focal(cam, self.cfg.get("uncalibrated", False))
            cameras[cam_id] = cam
            scales[cam_id] = (width, height)

        image_dir = self.output_dir / self.name / scene_id / "images"
        entries = []
        for idx, im in enumerate(selected):
            out_name = f"{idx:06d}.png"
            entries.append((out_name, im.camera_id, im.cam_from_world()))
            out_path = image_dir / out_name
            if not with_images or out_path.exists():
                continue
            image_dir.mkdir(parents=True, exist_ok=True)
            src_path = scene_dir / "images" / im.name
            img = cv2.imread(str(src_path))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {src_path}")
            width, height = scales[im.camera_id]
            cv2.imwrite(
                str(out_path),
                cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA),
            )

        logging.info(f"[{self.name}/{scene_id}] {len(entries)} views")
        return _build_scene(self.name, scene_id, image_dir, cameras, entries)


# ---------------------------------------------------------------------------
# 1DSfM
# ---------------------------------------------------------------------------
def _parse_pairs_to_neighbors(pairs_path, recon):
    """Convert an hloc pairs file into a neighbors dict keyed by image id."""
    name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
    neighbors = {}
    with open(pairs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            id_a = name_to_id.get(parts[0])
            id_b = name_to_id.get(parts[1])
            if id_a is None or id_b is None:
                continue
            neighbors.setdefault(id_a, []).append(id_b)
    for img_id in recon.images:
        neighbors.setdefault(img_id, [])
    return neighbors


class OneDSfMLoader:
    """1DSfM internet photo collections.

    Scenes run up to ~3000 images, so pairs come from NetVLAD retrieval
    rather than exhaustive matching. There are no reliable GT poses, so
    these scenes report runtime and reconstruction size only.
    """

    name = "1dsfm"
    default_scenes = ONEDSFM_DEFAULT_SCENES
    all_scenes = ONEDSFM_SCENES

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.data_dir = Path(cfg["data_dir"])

    def _copy_images(self, scene_id):
        src_dir = self.data_dir / scene_id / "images"
        dst_dir = self.output_dir / self.name / scene_id / "images"
        if dst_dir.exists():
            return dst_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in src_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)
                n += 1
        logging.info(f"Copied {n} images to {dst_dir}")
        return dst_dir

    def load(self, scene_id, with_images=True):
        image_dir = (
            self._copy_images(scene_id)
            if with_images
            else self.output_dir / self.name / scene_id / "images"
        )

        scene_dir = self.data_dir / scene_id
        list_path = scene_dir / "list.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"list.txt not found: {list_path}")

        # list.txt lines are "<image_name> 0 <focal_length>".
        entries = []
        with open(list_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                entries.append((Path(parts[0]).name, float(parts[2])))
        logging.info(f"Parsed {len(entries)} images from {list_path}")

        # One camera per image: internet photos have varying intrinsics.
        recon = pycolmap.Reconstruction()
        uncalibrated = self.cfg.get("uncalibrated", False)
        for idx, (img_name, focal) in enumerate(entries):
            img_path = image_dir / img_name
            if not img_path.exists():
                logging.warning(f"Image not found, skipping: {img_path}")
                continue
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size

            camera_id = image_id = idx + 1
            cam = pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=w,
                height=h,
                params=[focal, w / 2.0, h / 2.0],
                camera_id=camera_id,
            )
            cam.has_prior_focal_length = not uncalibrated
            recon.add_camera(cam)

            rig = pycolmap.Rig(rig_id=camera_id)
            rig.add_ref_sensor(cam.sensor_id)
            recon.add_rig(rig)

            img = pycolmap.Image(
                name=img_name,
                camera_id=camera_id,
                image_id=image_id,
                frame_id=image_id,
            )
            frame = pycolmap.Frame(frame_id=image_id, rig_id=camera_id)
            frame.add_data_id(img.data_id)
            recon.add_frame(frame)
            recon.add_image(img)

        logging.info(
            f"Built reconstruction: {recon.num_images()} images, "
            f"{recon.num_cameras()} cameras"
        )

        num_matched = self.cfg.get("n_neighbors", 30)

        def neighbors_fn(structure_dir):
            import hloc.extract_features
            import hloc.pairs_from_retrieval

            workspace_dir = Path(structure_dir) / "retrieval"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            pairs_path = workspace_dir / f"pairs-netvlad{num_matched}.txt"

            t0 = time.time()
            logging.info("Extracting NetVLAD global descriptors...")
            descriptor_path = hloc.extract_features.main(
                hloc.extract_features.confs["netvlad"], image_dir, workspace_dir
            )
            logging.info(f"Computing top-{num_matched} retrieval pairs...")
            hloc.pairs_from_retrieval.main(
                descriptor_path, pairs_path, num_matched
            )
            elapsed = time.time() - t0
            return _parse_pairs_to_neighbors(pairs_path, recon), elapsed

        return Scene(
            dataset=self.name,
            scene_id=scene_id,
            image_dir=image_dir,
            recon=recon,
            index_list=sorted(recon.images.keys()),
            gt_poses=None,
            neighbors_fn=neighbors_fn,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_LOADERS = {
    "hypersim": HypersimLoader,
    "scannetpp": ScanNetppLoader,
    "eth3d": Eth3dLoader,
    "7scenes": SevenScenesLoader,
    "1dsfm": OneDSfMLoader,
}

DATASET_NAMES = list(_LOADERS)

# Where each dataset's source data lives, unless --data_dir overrides it.
# scannetpp and 7scenes both come from the DA3-BENCH release.
DEFAULT_DATA_ROOTS = {
    "hypersim": "~/data/Hypersim/data",
    "scannetpp": "~/data/benchmark_dataset",
    "7scenes": "~/data/benchmark_dataset",
    "eth3d": (
        "~/myProjects/benchmark/colmap/benchmark/reconstruction/data/eth3d/dslr"
    ),
    "1dsfm": "data/1dsfm",
}


def get_loader(name, cfg, output_dir):
    """Build the loader for a dataset name."""
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {' '.join(DATASET_NAMES)}"
        )
    return _LOADERS[name](cfg, output_dir)
