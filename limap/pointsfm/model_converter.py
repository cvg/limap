import os
import sys

from limap.util.geometry import rotation_from_quaternion

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hloc.utils.read_write_model as colmap_utils
from colmap_reader import PyReadCOLMAP


def convert_colmap_to_visualsfm(colmap_model_path, output_nvm_file):
    reconstruction = PyReadCOLMAP(colmap_model_path)
    colmap_cameras, colmap_images, colmap_points = (
        reconstruction["cameras"],
        reconstruction["images"],
        reconstruction["points"],
    )
    with open(output_nvm_file, "w") as f:
        f.write("NVM_V3\n\n")

        # write images
        f.write(f"{len(colmap_images)}\n")
        map_image_id = dict()
        for cnt, item in enumerate(colmap_images.items()):
            img_id, colmap_image = item
            map_image_id[img_id] = cnt
            img_name = colmap_image.name
            cam_id = colmap_image.camera_id
            cam = colmap_cameras[cam_id]
            if cam.model == "SIMPLE_PINHOLE":
                assert cam.params[1] == 0.5 * cam.width
                assert cam.params[2] == 0.5 * cam.height
                focal = cam.params[0]
                k1 = 0.0
            elif cam.model == "PINHOLE":
                assert cam.params[0] == cam.params[1]
                assert cam.params[2] == 0.5 * cam.width
                assert cam.params[3] == 0.5 * cam.height
                focal = cam.params[0]
                k1 = 0.0
            elif cam.model == "SIMPLE_RADIAL":
                assert cam.params[1] == 0.5 * cam.width
                assert cam.params[2] == 0.5 * cam.height
                focal = cam.params[0]
                k1 = cam.params[3]
            else:
                raise ValueError("Camera model not supported in VisualSfM.")
            f.write(f"{img_name}\t")
            f.write(f" {focal}")
            qvec, tvec = colmap_image.qvec, colmap_image.tvec
            R = rotation_from_quaternion(qvec)
            center = -R.transpose() @ tvec
            f.write(f" {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]}")
            f.write(f" {center[0]} {center[1]} {center[2]}")
            f.write(f" {k1} 0\n")
        f.write("\n")

        # write points
        f.write(f"{len(colmap_points)}\n")
        for _, point in colmap_points.items():
            xyz = point.xyz
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}")
            f.write(" 128 128 128")  # dummy color
            n_supports = len(point.image_ids)
            f.write(f" {n_supports}")
            for idx in range(n_supports):
                img_id = point.image_ids[idx]
                xy_id = point.point2D_idxs[idx]
                img_index = map_image_id[img_id]
                f.write(f" {img_index} {xy_id}")
                xy = colmap_images[img_id].xys[xy_id]
                f.write(f" {xy[0]} {xy[1]}")
            f.write("\n")


def convert_imagecols_to_colmap(imagecols, colmap_output_path):
    ### make folders
    if not os.path.exists(colmap_output_path):
        os.makedirs(colmap_output_path)

    ### write cameras.txt
    colmap_cameras = {}
    for cam_id in imagecols.get_cam_ids():
        cam = imagecols.cam(cam_id)
        model_id = int(cam.model)
        model_name = None
        if model_id == 0:
            model_name = "SIMPLE_PINHOLE"
        elif model_id == 1:
            model_name = "PINHOLE"
        elif model_id == 2:
            model_name = "SIMPLE_RADIAL"
        elif model_id == 3:
            model_name = "RADIAL"
        elif model_id == 4:
            model_name = "OPENCV"
        elif model_id == 5:
            model_name = "OPENCV_FISHEYE"
        elif model_id == 6:
            model_name = "FULL_OPENCV"
        else:
            raise ValueError("Camera model not supported.")
        colmap_cameras[cam_id] = colmap_utils.Camera(
            id=cam_id,
            model=model_name,
            width=cam.w(),
            height=cam.h(),
            params=cam.params,
        )
    fname = os.path.join(colmap_output_path, "cameras.txt")
    colmap_utils.write_cameras_text(colmap_cameras, fname)
    colmap_utils.write_cameras_binary(colmap_cameras, fname[:-4] + ".bin")

    ### write images.txt
    colmap_images = {}
    for img_id in imagecols.get_img_ids():
        imname = imagecols.image_name(img_id)
        camimage = imagecols.camimage(img_id)
        cam_id = camimage.cam_id
        qvec = camimage.pose.qvec
        tvec = camimage.pose.tvec
        colmap_images[img_id] = colmap_utils.Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=cam_id,
            name=imname,
            xys=[],
            point3D_ids=[],
        )
    fname = os.path.join(colmap_output_path, "images.txt")
    colmap_utils.write_images_text(colmap_images, fname)
    colmap_utils.write_images_binary(colmap_images, fname[:-4] + ".bin")

    ### write empty points3D.txt
    fname = os.path.join(colmap_output_path, "points3D.txt")
    colmap_utils.write_points3D_text({}, fname)
    colmap_utils.write_points3D_binary({}, fname[:-4] + ".bin")
