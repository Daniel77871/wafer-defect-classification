"""Handcrafted wafer-map features for the SVM baseline."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import radon


def extract_region_density(wafer: np.ndarray) -> np.ndarray:
    """Return defect density in 4 edge regions and a 3x3 interior grid."""
    wafer_array = np.asarray(wafer)
    if wafer_array.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got {wafer_array.ndim}D.")

    height, width = wafer_array.shape
    edge_h = max(1, height // 4)
    edge_w = max(1, width // 4)
    regions = [
        wafer_array[:edge_h, :],
        wafer_array[-edge_h:, :],
        wafer_array[:, :edge_w],
        wafer_array[:, -edge_w:],
    ]

    interior = wafer_array[edge_h : height - edge_h, edge_w : width - edge_w]
    if interior.size == 0:
        interior = wafer_array

    for row_band in np.array_split(interior, 3, axis=0):
        for cell in np.array_split(row_band, 3, axis=1):
            regions.append(cell)

    return np.asarray([_defect_density(region) for region in regions], dtype=np.float32)


def extract_radon_features(wafer: np.ndarray) -> np.ndarray:
    """Return summary statistics of Radon projections at 10 angles."""
    defect_mask = _defect_mask(wafer)
    angles = np.linspace(0.0, 180.0, 10, endpoint=False)
    sinogram = radon(defect_mask, theta=angles, circle=False)

    features = []
    for angle_index in range(sinogram.shape[1]):
        projection = sinogram[:, angle_index]
        features.extend(
            [
                float(projection.mean()),
                float(projection.std()),
                float(projection.max()),
                float(projection.min()),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def extract_geometric_features(wafer: np.ndarray) -> np.ndarray:
    """Return geometric properties of the largest defect component."""
    defect_mask = _defect_mask(wafer).astype(bool)
    labeled = label(defect_mask, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return np.zeros(7, dtype=np.float32)

    largest = max(props, key=lambda prop: prop.area)
    return np.asarray(
        [
            largest.area,
            largest.perimeter,
            largest.eccentricity,
            largest.solidity,
            largest.extent,
            largest.axis_major_length,
            largest.axis_minor_length,
        ],
        dtype=np.float32,
    )


def extract_all_features(wafer: np.ndarray) -> np.ndarray:
    """Return the concatenated 60-dimensional handcrafted feature vector."""
    return np.concatenate(
        [
            extract_region_density(wafer),
            extract_radon_features(wafer),
            extract_geometric_features(wafer),
        ]
    ).astype(np.float32)


def _defect_mask(wafer: np.ndarray) -> np.ndarray:
    wafer_array = np.asarray(wafer)
    if wafer_array.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got {wafer_array.ndim}D.")
    return (wafer_array == 2).astype(np.float32)


def _defect_density(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    active = region > 0
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        return 0.0
    return float(np.count_nonzero(region == 2) / active_count)
