import numpy as np


def rotation_from_euler_angles(rot_x, rot_y, rot_z):
    # Calculate rotation about y axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)],
        ]
    )
    # Calculate rotation about y axis
    R_y = np.array(
        [
            [np.cos(rot_y), 0.0, np.sin(rot_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(rot_y), 0.0, np.cos(rot_y)],
        ]
    )
    # Calculate rotation about z axis
    R_z = np.array(
        [
            [
                np.cos(rot_z),
                -np.sin(rot_z),
                0.0,
            ],
            [
                np.sin(rot_z),
                np.cos(rot_z),
                0.0,
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    return R_z @ R_y @ R_x


def rotation_from_quaternion(quad):
    norm = np.linalg.norm(quad)
    if norm < 1e-10:
        raise ValueError(
            f"Error! the quaternion is not robust. quad.norm() = {norm}"
        )
    quad = quad / norm
    qr, qi, qj, qk = quad[0], quad[1], quad[2], quad[3]
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = 1 - 2 * (qj**2 + qk**2)
    rot_mat[0, 1] = 2 * (qi * qj - qk * qr)
    rot_mat[0, 2] = 2 * (qi * qk + qj * qr)
    rot_mat[1, 0] = 2 * (qi * qj + qk * qr)
    rot_mat[1, 1] = 1 - 2 * (qi**2 + qk**2)
    rot_mat[1, 2] = 2 * (qj * qk - qi * qr)
    rot_mat[2, 0] = 2 * (qi * qk - qj * qr)
    rot_mat[2, 1] = 2 * (qj * qk + qi * qr)
    rot_mat[2, 2] = 1 - 2 * (qi**2 + qj**2)
    return rot_mat
