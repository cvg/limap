import numpy as np

def to_homogeneous(arr):
    # Adds a new column with ones
    return np.hstack([arr, np.ones((len(arr), 1))])

def to_homogeneous_t(arr):
    # Adds a new row with ones
    return np.vstack([arr, np.ones((1, arr.shape[1]))])

def to_cartesian(arr):
    return arr[..., :-1] / arr[..., -1].reshape((-1,) + (1,) * (arr.ndim - 1))

def to_cartesian_t(arr):
    return arr[:-1] / arr[-1]

def euler_angles_to_rot_mat(rot_x, rot_y, rot_z):
    # Calculate rotation about y axis
    R_x = np.array([[1., 0., 0.],
                    [0, np.cos(rot_x), -np.sin(rot_x)],
                    [0, np.sin(rot_x), np.cos(rot_x)]])
    # Calculate rotation about y axis
    R_y = np.array([[np.cos(rot_y), 0., np.sin(rot_y)],
                    [0., 1., 0.],
                    [-np.sin(rot_y), 0., np.cos(rot_y)]])
    # Calculate rotation about z axis
    R_z = np.array([[np.cos(rot_z), -np.sin(rot_z), 0., ],
                    [np.sin(rot_z), np.cos(rot_z), 0., ],
                    [0., 0., 1.]])
    return R_z @ R_y @ R_x

def quaternion2rotmat(quad):
    qr, qi, qj, qk = quad[0], quad[1], quad[2], quad[3]
    rot_mat = np.zeros((3, 3))
    rot_mat[0,0] = 1 - 2 * (qj ** 2 + qk ** 2)
    rot_mat[0,1] = 2 * (qi * qj - qk * qr)
    rot_mat[0,2] = 2 * (qi * qk + qj * qr)
    rot_mat[1,0] = 2 * (qi * qj + qk * qr)
    rot_mat[1,1] = 1 - 2 * (qi ** 2 + qk ** 2)
    rot_mat[1,2] = 2 * (qj * qk - qi * qr)
    rot_mat[2,0] = 2 * (qi * qk - qj * qr)
    rot_mat[2,1] = 2 * (qj * qk + qi * qr)
    rot_mat[2,2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return rot_mat

def rotmat2quaternion(rotmat):
    from mathutils import Matrix
    quad = Matrix(rotmat).to_quaternion()
    return quad

