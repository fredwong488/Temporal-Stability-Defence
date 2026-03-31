"""
attack_utils.py
---------------
Attack-pipeline utilities for the ORA notebooks.

Provides:
  convert_to_kitti_bin -- save an Open3D PointCloud + intensity list to a
                          KITTI-format velodyne binary file (.bin)
"""

import os
import numpy as np


def convert_to_kitti_bin(pcd, intensity, save_file_path):
    """Serialise a point cloud to a KITTI velodyne binary file.

    The KITTI velodyne format stores points as a flat float32 array with
    4 channels per point: (x, y, z, intensity).

    Parameters
    ----------
    pcd            : open3d.geometry.PointCloud
    intensity      : list or ndarray of per-point intensity values
    save_file_path : str  destination path (parent directories are created)

    Returns
    -------
    bin_array : ndarray (N, 4), dtype float32
    """
    points = np.asarray(pcd.points, dtype=np.float32)
    intensity_arr = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
    bin_array = np.hstack([points, intensity_arr])  # (N, 4)

    os.makedirs(os.path.dirname(os.path.abspath(save_file_path)), exist_ok=True)
    bin_array.tofile(save_file_path)
    print("File Saved as :", save_file_path)
    return bin_array
