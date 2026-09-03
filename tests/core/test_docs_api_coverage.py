"""Guard against an API reference that drifts behind the code.

Most pages under ``docs/api`` list their entries one by one, so a newly
exported class or function is not picked up on its own -- it silently misses
the reference, which is how ``limap.geometry`` once ended up documenting five
of its thirty-one names. Walks the public API of every documented module and
checks each name is reachable from a directive in ``docs/api/*.rst``.

``automodule`` directives are expanded the way autodoc does: the members
defined in that module, restricted to the documented ones unless the directive
also carries ``:undoc-members:``.
"""

import ast
import importlib
import importlib.util
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import limap

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "docs" / "api"

# Every module the API reference covers. A package listed here is checked name
# by name; add new ones alongside their page.
DOCUMENTED_MODULES = (
    "limap.geometry",
    "limap.image",
    "limap.image.line",
    "limap.image.point",
    "limap.image.groups",
    "limap.image.groups.vplib",
    "limap.image.groups.planelib",
    "limap.image.dense_matcher",
    "limap.scene",
    "limap.sfm",
    "limap.estimators",
    "limap.estimators.absolute_pose",
    "limap.estimators.bundle_adjustment",
    "limap.estimators.group3d",
    "limap.estimators.line3d",
    "limap.estimators.triangulation",
    "limap.evaluation",
    "limap.runners",
    "limap.visualize",
    "limap.util",
)

# Top-level packages the reference is expected to have a page for. `_limap` is
# the compiled extension, re-exported through the packages above.
_SKIPPED_PACKAGES = frozenset({"_limap"})

# Entry points documented on the limap.cli page, but excluded there as
# argparse plumbing rather than API.
_CLI_INTERNALS = frozenset({"parse_args", "parse_config"})

_DIRECTIVE = re.compile(
    r"^\.\.\s+(?P<directive>currentmodule|module|automodule|autoclass|"
    r"autofunction|autodata|autoexception|py:class|py:function|py:data)::"
    r"\s+(?P<target>[\w\.]+)\s*$"
)
_OPTION = re.compile(r"^\s+:(?P<name>[\w-]+):\s*(?P<value>.*)$")


def _module_source(module_name):
    """Path of a module's .py file, or None for compiled or missing ones."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _members_of_source(path, include_undocumented):
    """Top-level names autodoc would document for a module source file."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if node.name.startswith("_"):
            continue
        if include_undocumented or ast.get_docstring(node) is not None:
            names.add(node.name)
    return names


def _members_of_module(module_name, include_undocumented):
    """Same, for a module that has to be imported (compiled submodules)."""
    module = importlib.import_module(module_name)
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    resolved = set()
    for name in names:
        member = getattr(module, name, None)
        if member is None or inspect.ismodule(member):
            continue
        if include_undocumented or getattr(member, "__doc__", None):
            resolved.add(name)
    return resolved


def _automodule_members(module_name, options):
    """Names an ``automodule`` directive puts on the page."""
    include_undocumented = "undoc-members" in options
    try:
        names = _members_of_module(module_name, include_undocumented)
    except ImportError:
        # An optional dependency is missing here but may well be installed
        # where the docs are built. Fall back to reading the source, and give
        # up on the directive if even that is out of reach.
        source = _module_source(module_name)
        if source is None:
            return None
        names = _members_of_source(source, include_undocumented)
    excluded = {
        part.strip() for part in options.get("exclude-members", "").split(",")
    }
    return names - excluded


def _documented_names():
    """Leaf names reachable from the directives in docs/api/*.rst."""
    documented = set()
    unresolved = set()
    for page in sorted(API_DIR.glob("*.rst")):
        lines = page.read_text().split("\n")
        for index, line in enumerate(lines):
            match = _DIRECTIVE.match(line)
            if match is None:
                continue
            directive = match.group("directive")
            target = match.group("target")
            if directive in ("currentmodule", "module"):
                continue
            if directive != "automodule":
                documented.add(target.rsplit(".", 1)[-1])
                continue
            options = {}
            for option_line in lines[index + 1 :]:
                option = _OPTION.match(option_line)
                if option is None:
                    break
                options[option.group("name")] = option.group("value")
            members = _automodule_members(target, options)
            if members is None:
                unresolved.add(target)
            else:
                documented |= members
    return documented, unresolved


def _public_names(module):
    """The public API of a module, submodules aside."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return {
        name
        for name in names
        if not inspect.ismodule(getattr(module, name, None))
    }


@pytest.fixture(scope="module")
def documented():
    if not API_DIR.is_dir():
        pytest.skip("docs/api is not part of this checkout")
    return _documented_names()


@pytest.mark.ci_workflow
def test_public_api_is_documented(documented):
    """Every public name of a documented module appears in the reference."""
    documented_names, unresolved = documented
    missing = []
    skipped = []
    for module_name in DOCUMENTED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # open3d, torch and the git-sourced detectors are optional; check
            # the modules that are importable here rather than giving up.
            skipped.append(module_name)
            continue
        undocumented = sorted(_public_names(module) - documented_names)
        if undocumented:
            missing.append(f"  {module_name}: {undocumented}")
    if len(skipped) == len(DOCUMENTED_MODULES):
        pytest.skip(f"none of the documented modules import here: {skipped}")
    assert not missing, (
        "These names are exported but appear on no page under docs/api:\n"
        + "\n".join(missing)
        + "\nAdd them to the page of their module, or drop them from "
        "__all__ if they are not public."
        + (
            f"\n(unresolved automodule targets: {sorted(unresolved)})"
            if unresolved
            else ""
        )
        + (f"\n(modules not importable here: {skipped})" if skipped else "")
    )


@pytest.mark.ci_workflow
def test_cli_entry_points_are_documented(documented):
    """The CLI page keeps up with the modules under limap/cli."""
    documented_names, _ = documented
    missing = []
    for info in pkgutil.iter_modules(
        importlib.import_module("limap.cli").__path__
    ):
        module = importlib.import_module(f"limap.cli.{info.name}")
        entry_points = {
            name
            for name, member in vars(module).items()
            if not name.startswith("_")
            and inspect.isfunction(member)
            and member.__module__ == module.__name__
            and name not in _CLI_INTERNALS
        }
        undocumented = sorted(entry_points - documented_names)
        if undocumented:
            missing.append(f"  limap.cli.{info.name}: {undocumented}")
    assert not missing, (
        "These CLI entry points appear on no page under docs/api:\n"
        + "\n".join(missing)
    )


@pytest.mark.ci_workflow
def test_every_package_is_documented():
    """A new limap package must be registered above, or it goes unchecked."""
    on_disk = {
        info.name
        for info in pkgutil.iter_modules(limap.__path__)
        if info.ispkg and not info.name.startswith("_")
    } - _SKIPPED_PACKAGES
    # limap.cli has a page of its own, guarded by the test above.
    registered = {name.split(".")[1] for name in DOCUMENTED_MODULES} | {"cli"}
    assert on_disk <= registered, (
        "These packages under src/limap have no entry in "
        f"DOCUMENTED_MODULES: {sorted(on_disk - registered)}. Give them a "
        "page under docs/api and register them here."
    )
