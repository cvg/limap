Output format and COLMAP compatibility
=======================================

**Starting from LIMAP 2.0.0, the toolbox is fully compatible with the COLMAP ecosystem**. Every mapping and SfM pipeline writes a plain COLMAP model, with the 3D structures stored alongside it under ``structures/``. There is no conversion or export step, and no LIMAP-specific container wrapped around the COLMAP data.

.. code-block:: text

    final_model/
        cameras.bin       # COLMAP: intrinsics
        images.bin        # COLMAP: poses and 2D-3D point observations
        points3D.bin      # COLMAP: 3D points
        rigs.bin
        frames.bin
        structures/
            lines3D.bin       # 3D line tracks
            groups3D.bin      # vanishing points, planes, primitives
            wireframe3D.bin   # 3D wireframe edges
            structures2d.bin  # per-image 2D structures and associations

Because the COLMAP files sit at the root of the model directory, anything that reads a COLMAP model reads a LIMAP reconstruction directly and simply ignores ``structures/``. Nothing needs to be converted or exported:

.. code-block:: python

    import pycolmap

    recon = pycolmap.Reconstruction("outputs/.../final_model")
    print(recon.num_images(), recon.num_points3D())

The same directory opens in the COLMAP GUI, and can be fed back into COLMAP's own tooling (``model_converter``, dense reconstruction, and so on).

------------------------------------------
Inheriting COLMAP advances
------------------------------------------

Compatibility is not only about the file format. The point side of the pipeline comes directly from COLMAP: LIMAP consolidates with its scene types (``Image``, ``Camera``, ``Reconstruction``, ``Track``, ``Rig``, ``Frame``), its estimators, its correspondence graph and various incremental mapper logic, and adds the structures on top rather than maintaining a parallel implementation. Improvements on the COLMAP side are therefore inherited rather than reimplemented:

* **Multi-camera rigs.** Reconstructions use COLMAP's rig and frame model - hence ``rigs.bin`` and ``frames.bin`` above, and the bundle adjustment can refine the rig extrinsics and the per-frame poses (``refine_sensor_from_rig``, ``refine_rig_from_world``), alongside the line, group and wireframe residuals.
* **Two-view geometry.** Image pairs are verified and initialised with COLMAP's own two-view geometry estimation, and the line and group correspondences are carried in the same ``TwoViewGeometry`` structures as the point matches.

Upgrading the pinned COLMAP version therefore benefits the structure pipelines too, without changes on the LIMAP side.

------------------------------------------
Reading the structures
------------------------------------------

To read the structures as well, use :py:class:`limap.scene.HolisticReconstruction`, which holds both halves:

.. code-block:: python

    from limap.scene import HolisticReconstruction

    recon = HolisticReconstruction()
    recon.read("outputs/.../final_model")

    print(recon.point_recon.num_points3D())     # pycolmap.Reconstruction
    print(recon.structure_recon.num_lines3D())
    print(recon.structure_recon.num_groups3D())

``point_recon`` is an ordinary ``pycolmap.Reconstruction``, so the point and pose side of the API is unchanged. ``HolisticReconstruction.exists_model(path)`` reports whether a directory holds both parts.

Text output is available through ``write_text`` / ``read_text``, with the binary format used by default.
