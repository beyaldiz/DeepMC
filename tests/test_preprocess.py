import pytest
import numpy as np
from pathlib import Path


def get_marker_config(markers_pos, joints_pos, weights):
    '''
    :param markers_pos: The position of markers: (56, 3)
    :param joints_pos: The position of joints: (24, 3)
    :param joints_transform: The roration matrix of joints: (24, 3, 3)
    :param weights: The skinning weights: (56, 24)
    :return:
        marker_configuration: (56, 24, 3)
    '''
    _offset_list = list()
    mrk_pos_matrix = np.array(markers_pos)
    jts_pos_matrix = np.array(joints_pos)
    weights_matrix = np.array(weights)
    tile_num = joints_pos.shape[0]
    for mrk_index in range(mrk_pos_matrix.shape[0]):
        mark_pos = mrk_pos_matrix[mrk_index]
        jts_offset = mark_pos - jts_pos_matrix
        jts_offset_local = [
            np.int64(weights_matrix[mrk_index, i] > 1e-5) * (jts_offset[i]).reshape(3) for i in
            range(tile_num)]
        jts_offset_local = np.array(jts_offset_local)
        _offset_list.append(jts_offset_local)
    return np.array(_offset_list)


def compute_naive(file, topology):
    adj = [[] for _ in range(len(topology))]
    for i, x in enumerate(topology):
        if i == 0 and x == 0:
            continue
        adj[x].append(i)
    loaded = np.load(str(file))
    a = loaded['J_R']
    b = loaded['J']
    glob_r = a
    local_r = a.copy()
    glob_t = b
    local_t = b.copy()
    def dfs(x, par, frame):
        if x != 0:
            local_r[frame, x] = glob_r[frame, par].T @ glob_r[frame, x]
        for child in adj[x]:
            dfs(child, x, frame)
    def dfs2(x, par):
        if x != 0:
            local_t[x] = glob_t[x] - glob_t[par]
        for child in adj[x]:
            dfs2(child, x)
    
    for i in range(a.shape[0]):
        dfs(0, -1, i)
    dfs2(0, -1)
    return local_r, local_t


def compute_vectorized(file, topology):
    loaded = np.load(str(file))
    glob_r, glob_t = loaded['J_R'], loaded['J']

    # Compute local rotations
    local_r = glob_r[:, topology, :, :].transpose(0, 1, 3, 2) @ glob_r
    local_r[:, 0, :, :] = glob_r[:, 0, :, :]

    # Compute local translations
    local_t = glob_t - glob_t[topology, :]
    local_t[0] = glob_t[0]
    return local_r, local_t

@pytest.mark.parametrize("file", ["data/MS_Synthetic/train_sample_data/01_01_01_poses_0.npz", "data/MS_Synthetic/test_sample_data/02_02_02_poses_0.npz"])
@pytest.mark.parametrize("topology", [[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]])
def test_preprocess(file, topology):
    a, b = compute_naive(file, topology), compute_vectorized(file, topology)
    assert np.allclose(a[1], b[1])
    assert np.allclose(a[0], b[0])