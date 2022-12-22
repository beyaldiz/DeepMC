"""
This script contains the data components of EPN.
This script is borrowed from [Chen et al. 2021].
https://github.com/nintendops/EPN_PointCloud/blob/main/SPConvNets/datasets/modelnet40.py
"""
import numpy as np
import os
import math
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
from scipy.spatial.transform import Rotation as sciR


def R_from_euler_np(angles):
    """
    angles: [(b, )3]
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angles[0]), -math.sin(angles[0])],
            [0, math.sin(angles[0]), math.cos(angles[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(angles[1]), 0, math.sin(angles[1])],
            [0, 1, 0],
            [-math.sin(angles[1]), 0, math.cos(angles[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(angles[2]), -math.sin(angles[2]), 0],
            [math.sin(angles[2]), math.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(R_z, np.dot(R_y, R_x))


def rotate_point_cloud(data, R=None, max_degree=None):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      Nx3 array, original point clouds
    R:
      3x3 array, optional Rotation matrix used to rotate the input
    max_degree:
      float, optional maximum DEGREE to randomly generate rotation
    Return:
      Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)

    if R is not None:
        rotation_angle = R
    elif max_degree is not None:
        rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
        rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or rotation_angle.ndim == 1:
        rotation_matrix = R_from_euler_np(rotation_angle)
    else:
        assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
        rotation_matrix = rotation_angle[:3, :3]

    if data is None:
        return None, rotation_matrix
    rotated_data = np.dot(rotation_matrix, data.reshape((-1, 3)).T)

    return rotated_data.T, rotation_matrix


# translation normalization
def centralize(pc):
    return pc - pc.mean(dim=2, keepdim=True)


def centralize_np(pc, batch=False):
    axis = 2 if batch else 1
    return pc - pc.mean(axis=axis, keepdims=True)


# scale/translation normalization
def normalize(pc):
    pc = centralize(pc)
    var = pc.pow(2).sum(dim=1, keepdim=True).sqrt()
    return pc / var.max(dim=2, keepdim=True)


def normalize_np(pc, batch=False):
    pc = centralize_np(pc, batch)
    axis = 1 if batch else 0
    var = np.sqrt((pc**2).sum(axis=axis, keepdims=True))
    return pc / var.max(axis=axis + 1, keepdims=True)


def uniform_resample_index_np(pc, n_sample, batch=False):
    if batch == True:
        raise NotImplementedError("resample in batch is not implemented")
    n_point = pc.shape[0]
    if n_point >= n_sample:
        # downsample
        idx = np.random.choice(n_point, n_sample, replace=False)
    else:
        # upsample
        idx = np.random.choice(n_point, n_sample - n_point, replace=True)
        idx = np.concatenate((np.arange(n_point), idx), axis=0)
    return idx


def uniform_resample_np(pc, n_sample, label=None, batch=False):
    if batch == True:
        raise NotImplementedError("resample in batch is not implemented")
    idx = uniform_resample_index_np(pc, n_sample, batch)
    if label is None:
        return idx, pc[idx]
    else:
        return idx, pc[idx], label[idx]


class ModelNet40Alignment(data.Dataset):
    def __init__(self, data_dir: str = "data/", mode="train", num_points=1024):
        super(ModelNet40Alignment, self).__init__()
        self.input_num = num_points

        # 'train' or 'eval'

        cats = ["airplane"]
        print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")

        self.dataset_path = data_dir + "EvenAlignedModelNet40PC/"
        self.all_data = []
        for cat in cats:
            for fn in glob.glob(os.path.join(self.dataset_path, cat, mode, "*.mat")):
                self.all_data.append(fn)
        print("[Dataloader] : Training dataset size:", len(self.all_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
        _, pc = uniform_resample_np(data["pc"], self.input_num)

        # normalization
        pc = normalize_np(pc.T)
        pc = pc.T
        pc_src, R_src = rotate_point_cloud(pc)
        pc_tgt = pc
        R = R_src
        pc_tensor = pc_src.transpose(-1, -2)

        return torch.from_numpy(pc_tensor.astype(np.float32)), torch.from_numpy(
            R.astype(np.float32)
        )
