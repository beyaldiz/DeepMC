import multiprocessing

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


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


def preprocess_file(args):
    file_path, save_dir, topology = args
    loaded = np.load(str(file_path))
    glob_r, glob_t = loaded['J_R'], loaded['J']

    # Compute local rotations
    local_r = glob_r[:, topology, :, :].transpose(0, 1, 3, 2) @ glob_r
    local_r[:, 0, :, :] = glob_r[:, 0, :, :]

    # Compute local translations
    local_t = glob_t - glob_t[topology, :]
    local_t[0] = glob_t[0]
    
    # Save the preprocessed data
    np.savez(str(save_dir / file_path.parent.stem / file_path.name),
        clean_markers=loaded['M'],
        raw_markers=loaded['M1'],
        J_t=loaded['J_t'],
        J_R=loaded['J_R'],
        Marker=loaded['Marker'],
        J=loaded["J"],
        marker_configuration=get_marker_config(loaded["Marker"], loaded["J"], loaded["weights"]),
        J_R_local=local_r,
        J_t_local=local_t)


def preprocess(
    data_dir: str = "data/",
    topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
):
    """
    Preprocess the data.
    """
    load_dir = pathlib.Path(data_dir) / "MS_Synthetic"
    save_dir = pathlib.Path(data_dir) / "MS_Synthetic_preprocessed"

    if not save_dir.exists():
        save_dir.mkdir()
        (save_dir / "train_sample_data").mkdir()
        (save_dir / "test_sample_data").mkdir()

    file_paths = [f for f in load_dir.rglob("*.npz")]
    args = [(file_path, save_dir, topology) for file_path in file_paths]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(preprocess_file, args)


def check_preprocess(data_dir: str = "data/"):
    preprocessed_dir = pathlib.Path(data_dir) / "MS_Synthetic_preprocessed"
    return preprocessed_dir.exists()


def read_file_ts(file_path):
    data = np.load(file_path)

    # Global joint positions
    # Expand dims for concatenation, (1, j, 3)
    t_pose = data["J"][None, ...]

    return t_pose


class TS_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        train: bool = False,
        topology: list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    ):
        # Check whether the data is preprocessed
        if not check_preprocess(data_dir):
            print("Preprocessed data not found, preprocessing...")
            preprocess(data_dir, topology)
            print("Preprocessed data saved.")

        files_dir = pathlib.Path(data_dir) / "MS_Synthetic_preprocessed" / ("train_sample_data" if train else "test_sample_data")
        file_paths = [f for f in files_dir.glob("*.npz")]

        # Load data in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            data_list = pool.map(read_file_ts, file_paths)

        # Concatenate data, (n, j, 3)
        data = np.concatenate(data_list, axis=0)

        # Get local joint positions with respect to the parent joint, (n, j, 3)
        data = data - data[:, topology, :]

        # Get statistics of the data
        stat_dir = files_dir / ".." / "ts_statistics.npy"
        if not stat_dir.is_file():
            if train:
                print("Template skeleton statistics not found, computing...")
                mean = np.mean(data, axis=0)
                std = np.std(data, ddof=1, axis=0)
                std[std < 1e-5] = 1.0

                np.save(files_dir / ".." / "ts_statistics.npy", np.array([mean, std]))
                print("Template skeleton statistics saved.")

            else:
                TS_Dataset(data_dir, train=True, topology=topology)

        self.X_t = data

    def __len__(self):
        return self.X_t.shape[0]

    def __getitem__(self, idx):
        return self.X_t[idx]