# Everything here needs matplotlib, seaborn or open3d, which live in the `viz`
# extra rather than the core dependencies -- and open3d ships no wheels for
# Python 3.13+. Resolve names on first access so that importing limap does not
# require any of them.
_LAZY_ATTRS = {
    "draw_2d_points": "viz2d",
    "draw_2d_lines": "viz2d",
    "draw_2d_vpresult": "viz2d",
    "open3d_get_3d_lines": "viz3d",
    "open3d_get_3d_points": "viz3d",
    "open3d_get_camera_frustums": "viz3d",
    "open3d_get_plane_mesh": "viz3d",
    "open3d_get_sphere_mesh": "viz3d",
    "open3d_get_cylinder_mesh": "viz3d",
    "open3d_visualize_3d_lines": "viz3d",
    "PlaneMesh": "viz3d",
    "SphereMesh": "viz3d",
    "CylinderMesh": "viz3d",
    "make_big_image": "viz_utils",
    "compute_robust_range_lines": "viz_utils",
    "compute_robust_range_points": "viz_utils",
    "get_group_binary_mask": "texture",
    "project_points_to_image": "texture",
    "create_textured_plane_mesh": "texture",
    "create_textured_cylinder_mesh": "texture",
    "create_textured_sphere_mesh": "texture",
}

__all__ = [
    "draw_2d_points",
    "draw_2d_lines",
    "draw_2d_vpresult",
    "open3d_get_3d_lines",
    "open3d_get_3d_points",
    "open3d_get_camera_frustums",
    "open3d_get_plane_mesh",
    "open3d_get_sphere_mesh",
    "open3d_get_cylinder_mesh",
    "open3d_visualize_3d_lines",
    "PlaneMesh",
    "SphereMesh",
    "CylinderMesh",
    "make_big_image",
    "compute_robust_range_lines",
    "compute_robust_range_points",
    "get_group_binary_mask",
    "project_points_to_image",
    "create_textured_plane_mesh",
    "create_textured_cylinder_mesh",
    "create_textured_sphere_mesh",
]


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing not in {"matplotlib", "seaborn", "open3d"}:
            raise
        hint = f"{name} needs {missing}, which is part of the `viz` extra:\n"
        hint += '    python -m pip install "limap[viz]"'
        if missing == "open3d":
            hint += (
                "\nopen3d publishes no wheels for Python 3.13+, so it is "
                "skipped there and the 3D helpers are unavailable."
            )
        raise ImportError(hint) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
