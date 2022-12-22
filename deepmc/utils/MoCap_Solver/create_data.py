import os
import multiprocessing

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from tqdm import tqdm

from deepmc.utils.MoCap_Solver.parse_data import (
    get_transfer_matrix,
    rot_error,
    get_Quaternion_fromRotationMatrix,
    qfix,
)
from deepmc.utils.MoCap_Solver.parse_data import get_marker_config1 as get_marker_config
from deepmc.utils.MoCap_Solver.parse_data import corrupt
from deepmc.utils.MoCap_Solver.utils import simulate_outlier_remove
from deepmc.models.ms_module import TSLitModule, MCLitModule, MOLitModule

# Hard coded
ref_markers = [54, 3, 46, 32, 36, 21, 7, 11]


def preprocess_file_windows_msalign(args):
    """
    This function is borrowed from [Chen et al. 2020].
    """
    file_path, save_dir, topology = args
    h = np.load(str(file_path))
    weights = h["weights"]
    M = h["M"]
    M1 = h["M1"]
    J_t = h["J_t"]
    J_R = h["J_R"]
    J = h["J"]
    N = M.shape[0]
    Marker = h["Marker"]
    JOINTNUM = J.shape[0]
    MARKERNUM = M.shape[1]

    # Hard coded window size
    windows_size = 64
    ################ Get motions ###########################################################
    motions = np.zeros((N, 4 * (JOINTNUM) + 3))
    prevlist = topology
    for i in range(N):
        for j in range(JOINTNUM):
            rot = np.linalg.inv(J_R[i, prevlist[j], :, :]).dot(J_R[i, j, :, :])
            q = Quaternion(matrix=rot, atol=1.0, rtol=1.0)
            motions[i, (4 * j) : (4 * j + 4)] = np.array([q[0], q[1], q[2], q[3]])
        motions[i, (4 * JOINTNUM) :] = J_t[i, 0, :]

    #######Find the Rigid Transform########
    windows_count = np.int64(np.ceil(N / np.float64(windows_size / 2)))
    if windows_count <= 1:
        return
    step_size = np.int64(windows_size / 2)
    newM = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    newM1 = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    new_J_t = np.zeros((windows_count, windows_size, JOINTNUM, 3))
    new_J_R = np.zeros((windows_count, windows_size, JOINTNUM, 3, 3))
    new_motion = np.zeros((windows_count, 4 * (JOINTNUM - 1) + 3, windows_size))
    new_first_rot = np.zeros((windows_count, windows_size, 4))

    RigidR = list()
    RigidT = list()
    t_pos_marker = np.load("data/MS_Synthetic_preprocessed/mean_ref_markers.npy")
    for idx_select in range(1):
        for i in range(windows_count):
            rot_error_list = []
            identity_rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            for ind in range(windows_size):
                M_i = M1[min(i * step_size + ind, N - 1), ref_markers, :]
                non_occluded = np.any((M_i != 0.0), axis=-1)
                R, T = get_transfer_matrix(M_i[non_occluded], t_pos_marker[non_occluded])
                re = rot_error(identity_rot, R)
                rot_error_list.append(re)
            rot_error_list = np.array(rot_error_list)
            min_index_list = np.argsort(rot_error_list)
            min_index = min_index_list[idx_select]
            M_i = M1[min(i * step_size + min_index, N - 1), ref_markers, :]
            non_occluded = np.any((M_i != 0.0), axis=-1)
            R, T = get_transfer_matrix(M_i[non_occluded], t_pos_marker[non_occluded])
            T = T.reshape(3, 1)
            RigidR.append(R)
            RigidT.append(T)
            for j in range(windows_size):
                newM[i, j, :, :] = (
                    R.dot(M[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                newM1[i, j, :, :] = (
                    R.dot(M1[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                new_J_t[i, j, :, :] = (
                    R.dot(J_t[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                for l in range(JOINTNUM):
                    new_J_R[i, j, l, :, :] = R.dot(
                        J_R[min(i * step_size + j, N - 1), l, :, :]
                    )
                new_motion[i, :, j] = motions[min(i * step_size + j, N - 1), 4:]
                new_motion[i, -3:, j] = new_J_t[i, j, 0, :]
                new_first_rot[i, j, :] = get_Quaternion_fromRotationMatrix(
                    new_J_R[i, j, 0, :, :]
                )
            q_orig = new_first_rot[i, :, :].copy()
            if min_index < windows_size - 1:
                q_orig1 = q_orig[min_index:].copy()
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, min_index:] = q_orig1
            if min_index > 0:
                q_orig1 = q_orig[0 : min_index + 1].copy()
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, 0 : min_index + 1] = q_orig1
        #########compute the marker config and weighted marker config############
        mrk_config = get_marker_config(Marker, J, weights)
        mrk_config = np.tile(
            mrk_config.reshape(1, MARKERNUM, JOINTNUM, 3), (windows_count, 1, 1, 1)
        )
        offsets = J - J[prevlist]
        new_offsets = np.tile(offsets.reshape(1, JOINTNUM, 3), (windows_count, 1, 1))
        outfile = str(
            save_dir
            / ("msalign_" + file_path.parent.stem.replace("sample", "windows"))
            / (file_path.stem + "_" + str(idx_select) + ".npz")
        )
        np.savez(
            outfile,
            offsets=new_offsets,
            M=newM,
            M1=newM1,
            J_t=new_J_t,
            J_R=new_J_R,
            RigidR=RigidR,
            RigidT=RigidT,
            Marker=Marker,
            J=J,
            mrk_config=mrk_config,
            motion=new_motion,
            first_rot=new_first_rot,
        )


def preprocess_file_windows_noalign(args):
    """
    This function is borrowed from [Chen et al. 2020].
    """
    file_path, save_dir, topology = args
    h = np.load(str(file_path))
    weights = h["weights"]
    M = h["M"]
    M1 = h["M1"]
    J_t = h["J_t"]
    J_R = h["J_R"]
    J = h["J"]
    N = M.shape[0]
    Marker = h["Marker"]
    JOINTNUM = J.shape[0]
    MARKERNUM = M.shape[1]

    # Hard coded window size
    windows_size = 64
    ################ Get motions ###########################################################
    motions = np.zeros((N, 4 * (JOINTNUM) + 3))
    prevlist = topology
    for i in range(N):
        for j in range(JOINTNUM):
            rot = np.linalg.inv(J_R[i, prevlist[j], :, :]).dot(J_R[i, j, :, :])
            q = Quaternion(matrix=rot, atol=1.0, rtol=1.0)
            motions[i, (4 * j) : (4 * j + 4)] = np.array([q[0], q[1], q[2], q[3]])
        motions[i, (4 * JOINTNUM) :] = J_t[i, 0, :]

    #######Find the Rigid Transform########
    windows_count = np.int64(np.ceil(N / np.float64(windows_size / 2)))
    if windows_count <= 1:
        return
    step_size = np.int64(windows_size / 2)
    newM = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    newM1 = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    new_J_t = np.zeros((windows_count, windows_size, JOINTNUM, 3))
    new_J_R = np.zeros((windows_count, windows_size, JOINTNUM, 3, 3))
    new_motion = np.zeros((windows_count, 4 * (JOINTNUM - 1) + 3, windows_size))
    new_first_rot = np.zeros((windows_count, windows_size, 4))

    RigidR = list()
    RigidT = list()
    t_pos_marker = np.load("data/MS_Synthetic_preprocessed/mean_ref_markers.npy")
    for idx_select in range(1):
        for i in range(windows_count):
            identity_rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            R, T = identity_rot, np.array([0.0, 0.0, 0.0])
            T = T.reshape(3, 1)
            RigidR.append(R)
            RigidT.append(T)
            for j in range(windows_size):
                newM[i, j, :, :] = (
                    R.dot(M[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                newM1[i, j, :, :] = (
                    R.dot(M1[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                new_J_t[i, j, :, :] = (
                    R.dot(J_t[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                for l in range(JOINTNUM):
                    new_J_R[i, j, l, :, :] = R.dot(
                        J_R[min(i * step_size + j, N - 1), l, :, :]
                    )
                new_motion[i, :, j] = motions[min(i * step_size + j, N - 1), 4:]
                new_motion[i, -3:, j] = new_J_t[i, j, 0, :]
                new_first_rot[i, j, :] = get_Quaternion_fromRotationMatrix(
                    new_J_R[i, j, 0, :, :]
                )
            q_orig = new_first_rot[i, :, :].copy()
            min_index = 0
            if min_index < windows_size - 1:
                q_orig1 = q_orig[min_index:].copy()
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, min_index:] = q_orig1
            if min_index > 0:
                q_orig1 = q_orig[0 : min_index + 1].copy()
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, 0 : min_index + 1] = q_orig1
        #########compute the marker config and weighted marker config############
        mrk_config = get_marker_config(Marker, J, weights)
        mrk_config = np.tile(
            mrk_config.reshape(1, MARKERNUM, JOINTNUM, 3), (windows_count, 1, 1, 1)
        )
        offsets = J - J[prevlist]
        new_offsets = np.tile(offsets.reshape(1, JOINTNUM, 3), (windows_count, 1, 1))
        outfile = str(
            save_dir
            / ("noalign_" + file_path.parent.stem.replace("sample", "windows"))
            / (file_path.stem + "_" + str(idx_select) + ".npz")
        )
        np.savez(
            outfile,
            offsets=new_offsets,
            M=newM,
            M1=newM1,
            J_t=new_J_t,
            J_R=new_J_R,
            RigidR=RigidR,
            RigidT=RigidT,
            Marker=Marker,
            J=J,
            mrk_config=mrk_config,
            motion=new_motion,
            first_rot=new_first_rot,
        )


def preprocess_file_windows_gtalign(args):
    """
    This function is borrowed from [Chen et al. 2020].
    """
    file_path, save_dir, topology = args
    h = np.load(str(file_path))
    weights = h["weights"]
    M = h["M"]
    M1 = h["M1"]
    J_t = h["J_t"]
    J_R = h["J_R"]
    J = h["J"]
    N = M.shape[0]
    Marker = h["Marker"]
    JOINTNUM = J.shape[0]
    MARKERNUM = M.shape[1]

    # Hard coded window size
    windows_size = 64
    ################ Get motions ###########################################################
    motions = np.zeros((N, 4 * (JOINTNUM) + 3))
    prevlist = topology
    for i in range(N):
        for j in range(JOINTNUM):
            rot = np.linalg.inv(J_R[i, prevlist[j], :, :]).dot(J_R[i, j, :, :])
            q = Quaternion(matrix=rot, atol=1.0, rtol=1.0)
            motions[i, (4 * j) : (4 * j + 4)] = np.array([q[0], q[1], q[2], q[3]])
        motions[i, (4 * JOINTNUM) :] = J_t[i, 0, :]

    #######Find the Rigid Transform########
    windows_count = np.int64(np.ceil(N / np.float64(windows_size / 2)))
    if windows_count <= 1:
        return
    step_size = np.int64(windows_size / 2)
    newM = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    newM1 = np.zeros((windows_count, windows_size, MARKERNUM, 3))
    new_J_t = np.zeros((windows_count, windows_size, JOINTNUM, 3))
    new_J_R = np.zeros((windows_count, windows_size, JOINTNUM, 3, 3))
    new_motion = np.zeros((windows_count, 4 * (JOINTNUM - 1) + 3, windows_size))
    new_first_rot = np.zeros((windows_count, windows_size, 4))

    RigidR = list()
    RigidT = list()
    t_pos_marker = np.load("data/MS_Synthetic_preprocessed/mean_ref_markers.npy")
    for idx_select in range(1):
        for i in range(windows_count):
            rot_error_list = []
            identity_rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            for ind in range(windows_size):
                root_R_inv, root_t_inv = J_R[min(i * step_size + ind, N - 1), 0].T, -J_t[min(i * step_size + ind, N - 1), 0]
                R, T = root_R_inv, root_R_inv @ root_t_inv
                re = rot_error(identity_rot, R)
                rot_error_list.append(re)
            rot_error_list = np.array(rot_error_list)
            min_index_list = np.argsort(rot_error_list)
            min_index = min_index_list[idx_select]
            root_R_inv, root_t_inv = J_R[min(i * step_size + min_index, N - 1), 0].T, -J_t[min(i * step_size + min_index, N - 1), 0]
            R, T = root_R_inv, root_R_inv @ root_t_inv
            T = T.reshape(3, 1)
            RigidR.append(R)
            RigidT.append(T)
            for j in range(windows_size):
                newM[i, j, :, :] = (
                    R.dot(M[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                newM1[i, j, :, :] = (
                    R.dot(M1[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                new_J_t[i, j, :, :] = (
                    R.dot(J_t[min(i * step_size + j, N - 1), :, :].T) + T
                ).T
                for l in range(JOINTNUM):
                    new_J_R[i, j, l, :, :] = R.dot(
                        J_R[min(i * step_size + j, N - 1), l, :, :]
                    )
                new_motion[i, :, j] = motions[min(i * step_size + j, N - 1), 4:]
                new_motion[i, -3:, j] = new_J_t[i, j, 0, :]
                new_first_rot[i, j, :] = get_Quaternion_fromRotationMatrix(
                    new_J_R[i, j, 0, :, :]
                )
            q_orig = new_first_rot[i, :, :].copy()
            if min_index < windows_size - 1:
                q_orig1 = q_orig[min_index:].copy()
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, min_index:] = q_orig1
            if min_index > 0:
                q_orig1 = q_orig[0 : min_index + 1].copy()
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 1, 4)
                q_orig1 = qfix(q_orig1)
                q_orig1 = q_orig1[::-1]
                q_orig1 = q_orig1.reshape(-1, 4)
                new_first_rot[i, 0 : min_index + 1] = q_orig1
        #########compute the marker config and weighted marker config############
        mrk_config = get_marker_config(Marker, J, weights)
        mrk_config = np.tile(
            mrk_config.reshape(1, MARKERNUM, JOINTNUM, 3), (windows_count, 1, 1, 1)
        )
        offsets = J - J[prevlist]
        new_offsets = np.tile(offsets.reshape(1, JOINTNUM, 3), (windows_count, 1, 1))
        outfile = str(
            save_dir
            / ("gtalign_" + file_path.parent.stem.replace("sample", "windows"))
            / (file_path.stem + "_" + str(idx_select) + ".npz")
        )
        np.savez(
            outfile,
            offsets=new_offsets,
            M=newM,
            M1=newM1,
            J_t=new_J_t,
            J_R=new_J_R,
            RigidR=RigidR,
            RigidT=RigidT,
            Marker=Marker,
            J=J,
            mrk_config=mrk_config,
            motion=new_motion,
            first_rot=new_first_rot,
        )


def preprocess_file(args):
    file_path, save_dir, topology = args
    loaded = np.load(str(file_path))
    glob_r, glob_t = loaded["J_R"], loaded["J"]

    # Compute local rotations
    local_r = glob_r[:, topology, :, :].transpose(0, 1, 3, 2) @ glob_r
    local_r[:, 0, :, :] = glob_r[:, 0, :, :]

    # Compute local translations
    local_t = glob_t - glob_t[topology, :]
    local_t[0] = glob_t[0]

    raw_markers = loaded["M1"]
    for i in range(raw_markers.shape[0]):
        raw_markers[i, ref_markers] = corrupt(raw_markers[i, ref_markers], 0.1, 0.1, 0.3)

    # Save the preprocessed data
    np.savez(
        str(save_dir / file_path.parent.stem / file_path.name),
        M=loaded["M"],
        M1=raw_markers,
        J_t=loaded["J_t"],
        J_R=loaded["J_R"],
        Marker=loaded["Marker"],
        J=loaded["J"],
        marker_configuration=get_marker_config(
            loaded["Marker"], loaded["J"], loaded["weights"]
        ),
        J_R_local=local_r,
        J_t_local=local_t,
        weights=loaded["weights"],
    )

    if "train" in file_path.parent.stem:
        return loaded["Marker"]


def main():
    topology = [
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    load_dir = pathlib.Path("data") / "MS_Synthetic"
    save_dir = pathlib.Path("data") / "ours_Synthetic"

    if not save_dir.exists():
        save_dir.mkdir()
        (save_dir / "train_sample_data").mkdir()
        (save_dir / "test_sample_data").mkdir()
        (save_dir / "noalign_train_windows_data").mkdir()
        (save_dir / "noalign_test_windows_data").mkdir()
        (save_dir / "msalign_train_windows_data").mkdir()
        (save_dir / "msalign_test_windows_data").mkdir()
        (save_dir / "gtalign_train_windows_data").mkdir()
        (save_dir / "gtalign_test_windows_data").mkdir()

    file_paths = [f for f in load_dir.rglob("*.npz")]
    args = [(file_path, save_dir, topology) for file_path in file_paths]

    # print("Creating Ours Synthetic...")

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(preprocess_file, args)

    # print("Ours Synthetic created!")


    load_dir_train = pathlib.Path("data") / "ours_Synthetic" / "train_sample_data"
    load_dir_test = pathlib.Path("data") / "ours_Synthetic" / "test_sample_data"

    file_paths = [f for f in load_dir_train.rglob("*.npz")] + [f for f in load_dir_test.rglob("*.npz")]
    args = [(file_path, save_dir, topology) for file_path in file_paths]
    
    # print("Creating no align...")

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(preprocess_file_windows_noalign, args)

    # print("no align created!")

    print("Creating ms align...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(preprocess_file_windows_msalign, args)

    print("ms align created!")

    # print("Creating gt align...")

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(preprocess_file_windows_gtalign, args)

    # print("gt align created!")

if __name__ == "__main__":
    main()