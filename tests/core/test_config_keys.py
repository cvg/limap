"""Guard against config keys that silently do nothing.

``dacite`` runs in non-strict mode, so a YAML key matching no dataclass field
is dropped without a warning. Walks every shipped config against each option
class that loads it. Stops at pybind option objects, which are not dataclasses.
"""

import dataclasses
import typing
from pathlib import Path

import pytest
import yaml

import limap.runners

REPO_ROOT = Path(__file__).resolve().parents[2]
CFGS_DIR = REPO_ROOT / "cfgs"

# Read straight off the config dict by the runners, not via from_dict.
RUNNER_KEYS = frozenset(
    {
        "base_config_file",
        "output_dir",
        "data_dir",
        "scene_id",
        "cam_id",
        "input_n_views",
        "input_stride",
        "model_path",
        "image_path",
    }
)

# Option classes each config directory is loaded into. Directories with more
# than one consumer are the case worth guarding.
CONFIG_CONSUMERS = {
    "structure_triangulation": (
        limap.runners.AutomaticStructureTriangulationOptions,
        limap.runners.IncrementalStructureTriangulationOptions,
    ),
    "structure_incremental_reconstruction": (
        limap.runners.AutomaticStructureIncrementalReconstructionOptions,
    ),
    "geometry_guided_line_reconstruction": (
        limap.runners.GeometryGuidedLineReconstructionOptions,
    ),
    "localization": (limap.runners.PointLineLocalizationOptions,),
}

# Files loaded by only some of their directory's consumers.
FILE_CONSUMERS = {
    "cfgs/structure_triangulation/default_cpu.yaml": (
        limap.runners.AutomaticStructureTriangulationOptions,
    ),
}


def _field_types(data_class):
    """Map field name -> resolved type, or None if not a dataclass."""
    try:
        fields = dataclasses.fields(data_class)
    except TypeError:
        return None
    hints = None
    resolved = {}
    for f in fields:
        annotation = f.type
        if isinstance(annotation, str):
            if hints is None:
                hints = typing.get_type_hints(data_class)
            annotation = hints.get(f.name)
        # Unwrap Optional[X] / X | None to X.
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args:
            annotation = args[0]
        resolved[f.name] = annotation
    return resolved


def _dead_keys(cfg, data_class, prefix=""):
    field_types = _field_types(data_class)
    if field_types is None:
        return []
    dead = []
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if key not in field_types:
            if not (prefix == "" and key in RUNNER_KEYS):
                dead.append(path)
            continue
        if isinstance(value, dict):
            dead += _dead_keys(value, field_types[key], f"{path}.")
    return dead


def _cases():
    """Every (config file, consuming option class) pair to check."""
    for directory, consumers in sorted(CONFIG_CONSUMERS.items()):
        for cfg_path in sorted((CFGS_DIR / directory).glob("*.yaml")):
            rel = str(cfg_path.relative_to(REPO_ROOT))
            for data_class in FILE_CONSUMERS.get(rel, consumers):
                yield cfg_path, data_class


# Keys a secondary consumer does not declare, but that are live for the
# directory's primary consumer. Wiring them in would change behaviour, so each
# is recorded with why. test_known_undeclared_is_current keeps this honest.
KNOWN_UNDECLARED = {
    (
        "cfgs/structure_triangulation/default.yaml",
        "IncrementalStructureTriangulationOptions",
    ): {
        "skip_exists": (
            "Declaring it here would flip line_detection/line_matcher "
            "skip_exists from True to False. Needs a deliberate call."
        ),
        "n_visible_views": (
            "Never read by the primary consumer either; the live filter is "
            "line_triangulation.min_visible_views. Wire it or drop it."
        ),
    },
}


def _allowed(cfg_path, data_class):
    key = (str(cfg_path.relative_to(REPO_ROOT)), data_class.__name__)
    return set(KNOWN_UNDECLARED.get(key, {}))


@pytest.mark.ci_workflow
def test_configs_have_no_dead_keys():
    violations = []
    for cfg_path, data_class in _cases():
        cfg = yaml.safe_load(cfg_path.read_text())
        dead = sorted(
            set(_dead_keys(cfg, data_class)) - _allowed(cfg_path, data_class)
        )
        if dead:
            violations.append(
                f"  {cfg_path.relative_to(REPO_ROOT)} -> "
                f"{data_class.__name__}: {dead}"
            )
    assert not violations, (
        "These configs set keys their consuming option class does not "
        "declare, so dacite drops them silently:\n"
        + "\n".join(violations)
        + "\nEither point them at the field that actually holds the setting, "
        "or remove them."
    )


@pytest.mark.ci_workflow
def test_known_undeclared_is_current():
    """Every recorded asymmetry must still be one, so the list cannot rot."""
    stale = []
    for (rel, class_name), entries in KNOWN_UNDECLARED.items():
        cfg_path = REPO_ROOT / rel
        assert cfg_path.exists(), f"{rel} no longer exists; drop its entry"
        data_class = next(
            (
                c
                for consumers in CONFIG_CONSUMERS.values()
                for c in consumers
                if c.__name__ == class_name
            ),
            None,
        )
        assert data_class is not None, (
            f"{class_name} is no longer a registered consumer; drop its entry"
        )
        cfg = yaml.safe_load(cfg_path.read_text())
        still_dead = set(_dead_keys(cfg, data_class))
        for key in entries:
            if key not in still_dead:
                stale.append(f"{rel} -> {class_name}: {key}")
    assert not stale, (
        "These keys are now declared, so their KNOWN_UNDECLARED entries are "
        f"stale and should be removed: {stale}"
    )


@pytest.mark.ci_workflow
def test_every_config_directory_has_a_consumer():
    """A new config directory must be registered above, or it goes unchecked."""
    on_disk = {p.name for p in CFGS_DIR.iterdir() if p.is_dir()}
    assert on_disk == set(CONFIG_CONSUMERS), (
        "cfgs/ subdirectories and CONFIG_CONSUMERS disagree: "
        f"unregistered={sorted(on_disk - set(CONFIG_CONSUMERS))}, "
        f"stale={sorted(set(CONFIG_CONSUMERS) - on_disk)}"
    )
