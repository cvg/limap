import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from limap.util.geometry import rotation_from_quaternion

from .colmap_reader import PyReadCOLMAP

def convert_colmap_to_visualsfm(colmap_model_path, output_nvm_file):
    colmap_cameras, colmap_images, colmap_points = PyReadCOLMAP(colmap_model_path)
    with open(output_nvm_file, "w") as f:
        f.write("NVM_V3\n\n")

        # write images
        f.write("{0}\n".format(len(colmap_images)))
        map_image_id = dict()
        counter = 0
        for img_id, colmap_image in colmap_images.items():
            map_image_id[img_id] = counter
            counter += 1
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
            f.write("{0}\t".format(img_name))
            f.write(" {0}".format(focal))
            qvec, tvec = colmap_image.qvec, colmap_image.tvec
            R = rotation_from_quaternion(qvec)
            center = - R.transpose() @ tvec
            f.write(" {0} {1} {2} {3}".format(qvec[0], qvec[1], qvec[2], qvec[3]))
            f.write(" {0} {1} {2}".format(center[0], center[1], center[2]))
            f.write(" {0} 0\n".format(k1))
        f.write("\n")

        # write points
        f.write("{0}\n".format(len(colmap_points)))
        for pid, point in colmap_points.items():
            xyz = point.xyz
            f.write("{0} {1} {2}".format(xyz[0], xyz[1], xyz[2]))
            f.write(" 128 128 128") # dummy color
            n_supports = len(point.image_ids)
            f.write(" {0}".format(n_supports))
            for idx in range(n_supports):
                img_id = point.image_ids[idx]
                xy_id = point.point2D_idxs[idx]
                img_index = map_image_id[img_id]
                f.write(" {0} {1}".format(img_index, xy_id))
                xy = colmap_images[img_id].xys[xy_id]
                f.write(" {0} {1}".format(xy[0], xy[1]))
            f.write("\n")

