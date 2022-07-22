import multiprocessing

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


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
        files_dir = pathlib.Path(data_dir) / "MS_Synthetic" / ("train_sample_data" if train else "test_sample_data")
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
                self.mean = np.mean(data, axis=0)
                self.std = np.std(data, ddof=1, axis=0)
                self.std[self.std < 1e-5] = 1.0

                np.save(files_dir / ".." / "ts_statistics.npy", np.array([mean, std]))

            else:
                TS_Dataset(data_dir, train=True, topology=topology)
        
        else:
            mean, std = np.load(stat_dir)
            self.mean = mean
            self.std = std

        self.X_t = data

    def __len__(self):
        return len(self.X_t)

    def __getitem__(self, idx):
        return self.X_t[idx], self.mean, self.std
