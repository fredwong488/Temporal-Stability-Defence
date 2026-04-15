"""
kitti_utils.py
--------------
KITTI label/calibration/velodyne loading utilities.

Provides:
  KittiObject              -- parsed representation of a KITTI label line
  get_obj_data_intensity   -- load a KITTI scene (labels + calib + velodyne)
  get_obj_regions_intensity -- extract per-object point-cloud ROIs
"""

import os
import struct
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# KITTI label / calibration parsing
# ---------------------------------------------------------------------------

class KittiObject:
    """One annotated object from a KITTI label_2 file."""

    def __init__(self, line):
        parts = line.strip().split()
        self.type = parts[0]           # e.g. "Car", "Pedestrian", "Cyclist"
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        # 2-D bounding box in image coords
        self.bbox = [float(x) for x in parts[4:8]]
        # 3-D dimensions in camera coords (h, w, l) in metres
        self.height = float(parts[8])
        self.width  = float(parts[9])
        self.length = float(parts[10])
        # 3-D location in camera coords (bottom centre)
        self.x = float(parts[11])
        self.y = float(parts[12])
        self.z = float(parts[13])
        # rotation around camera Y axis
        self.rotation_y = float(parts[14])

    def __repr__(self):
        return (f"KittiObject(type={self.type}, xyz=({self.x:.2f},{self.y:.2f},"
                f"{self.z:.2f}), hwl=({self.height:.2f},{self.width:.2f},{self.length:.2f}))")


def _parse_label_file(label_file):
    objects = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(KittiObject(line))
    return objects


def _parse_calib_file(calib_file):
    """Return (R0_rect 3x3, Tr_velo_to_cam 3x4) as numpy arrays."""
    data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.array([float(v) for v in vals.split()])

    R0_rect = data['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = data['Tr_velo_to_cam'].reshape(3, 4)
    P2 = data['P2'].reshape(3, 4) if 'P2' in data else None
    return R0_rect, Tr_velo_to_cam, P2


def _load_velodyne_bin(lidar_file):
    """Return (Open3D PointCloud, intensity ndarray) from a KITTI .bin file."""
    pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, intensity


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def _compute_3d_bbox_corners_cam(obj):
    """Return (8, 3) array of 3-D box corners in *camera* coordinates."""
    h, w, l = obj.height, obj.width, obj.length
    ry = obj.rotation_y

    # Corners in object-local frame (before rotation/translation)
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([  0.,   0.,   0.,   0.,   -h,   -h,   -h,   -h])
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])

    # Rotation around the camera Y axis
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    R = np.array([
        [ cos_ry, 0., sin_ry],
        [     0., 1.,     0.],
        [-sin_ry, 0., cos_ry],
    ])

    corners = R @ np.vstack([x_c, y_c, z_c])   # (3, 8)
    corners[0] += obj.x
    corners[1] += obj.y
    corners[2] += obj.z
    return corners.T                              # (8, 3)


def _cam_corners_to_velo(corners_cam, R0_rect, Tr_velo_to_cam):
    """Transform (N, 3) corners from camera frame to velodyne frame."""
    # Build full 4x4 transforms
    R0 = np.eye(4)
    R0[:3, :3] = R0_rect

    Tr = np.eye(4)
    Tr[:3, :4] = Tr_velo_to_cam

    # Combined camera ← velodyne transform; invert to get velo ← cam
    cam_from_velo = R0 @ Tr
    velo_from_cam = np.linalg.inv(cam_from_velo)

    N = corners_cam.shape[0]
    corners_h = np.hstack([corners_cam, np.ones((N, 1))])  # (N, 4)
    corners_velo = (velo_from_cam @ corners_h.T).T          # (N, 4)
    return corners_velo[:, :3]                               # (N, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_obj_data_intensity(label_file, calib_file, lidar_file, show_plots=False):
    """Load a KITTI scene and return per-object metadata.

    Parameters
    ----------
    label_file : str   path to label_2/*.txt
    calib_file : str   path to calib/*.txt
    lidar_file : str   path to velodyne/*.bin
    show_plots : bool  unused (kept for API compatibility)

    Returns
    -------
    objects       : list[KittiObject]
    obj_coords    : list[ndarray (8, 3)]  -- 3-D bbox corners in velodyne frame
    pcd           : open3d.geometry.PointCloud
    pcd_intensity : ndarray (N,)
    """
    objects = _parse_label_file(label_file)
    R0_rect, Tr_velo_to_cam, _P2 = _parse_calib_file(calib_file)
    pcd, pcd_intensity = _load_velodyne_bin(lidar_file)

    obj_coords = []
    for obj in objects:
        corners_cam = _compute_3d_bbox_corners_cam(obj)
        corners_velo = _cam_corners_to_velo(corners_cam, R0_rect, Tr_velo_to_cam)
        obj_coords.append(corners_velo)

    return objects, obj_coords, pcd, pcd_intensity


def get_obj_regions_intensity(objects, obj_coords, pcd, pcd_intensity,
                               show_plots=False):
    """Extract the point-cloud region of interest for each annotated object.

    Parameters
    ----------
    objects       : list[KittiObject]
    obj_coords    : list[ndarray (8, 3)]  velodyne-frame bbox corners
    pcd           : open3d.geometry.PointCloud   full scene
    pcd_intensity : ndarray (N,)
    show_plots    : bool  unused (kept for API compatibility)

    Returns
    -------
    ROI            : list[open3d.geometry.PointCloud]  one per object
    draw1          : list[None]  placeholder for visualisation data
    draw2          : list[None]  placeholder for visualisation data
    intensity_list : list[ndarray]  per-object intensity arrays
    """
    ROI = []
    intensity_list = []
    draw1 = []
    draw2 = []

    pcd_intensity_arr = np.asarray(pcd_intensity)

    for corners in obj_coords:
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners))

        indices = bbox.get_point_indices_within_bounding_box(pcd.points)

        if len(indices) == 0:
            ROI.append(o3d.geometry.PointCloud())
            intensity_list.append(np.array([]))
        else:
            roi_pcd = pcd.select_by_index(indices)
            ROI.append(roi_pcd)
            intensity_list.append(pcd_intensity_arr[indices].copy())

        draw1.append(None)
        draw2.append(None)

    return ROI, draw1, draw2, intensity_list
