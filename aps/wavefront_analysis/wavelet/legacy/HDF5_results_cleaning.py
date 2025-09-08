import argparse
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os
from aps.wavefront_analysis.wavelet.legacy.func import frankotchellappa

def clean_phase_profile_absolute_threshold(phase_profile, threshold=10.0, filter_size=3):
    """
    Clean a 2D phase profile by identifying bad points based on an absolute threshold
    and replacing them with a median filter.

    Parameters:
        phase_profile (numpy.ndarray): The 2D phase profile array.
        threshold (float): Absolute threshold for identifying bad points.
        filter_size (int): Size of the neighborhood for the median filter (default is 3x3).

    Returns:
        numpy.ndarray: The cleaned phase profile.
        numpy.ndarray: The bad point mask (True for bad points).
    """
    # Identify bad points as those exceeding the absolute threshold
    bad_point_mask = np.abs(phase_profile) > threshold

    # Apply a median filter to the entire phase profile
    filtered_phase = median_filter(phase_profile, size=filter_size)

    # Replace bad points with the filtered values
    phase_profile_cleaned = np.copy(phase_profile)
    phase_profile_cleaned[bad_point_mask] = filtered_phase[bad_point_mask]

    return phase_profile_cleaned, bad_point_mask


def clean_phase_profile_local(phase_profile, k=3, filter_size=3):
    """
    Clean a 2D phase profile by identifying bad points based on local deviations
    (difference from neighboring points) and replacing them with a median filter.

    Parameters:
        phase_profile (numpy.ndarray): The 2D phase profile array.
        k (float): Threshold multiplier for identifying bad points (local deviation).
        filter_size (int): Size of the neighborhood for the median filter (default is 3x3).

    Returns:
        numpy.ndarray: The cleaned phase profile.
        numpy.ndarray: The bad point mask (True for bad points).
    """
    # Apply a median filter to get the local neighborhood median
    local_median = median_filter(phase_profile, size=filter_size)

    # Compute the absolute difference from the local median
    local_deviation = np.abs(phase_profile - local_median)

    # Compute the local deviation threshold (mean absolute deviation of neighbors)
    mad = median_filter(local_deviation, size=filter_size)  # Median Absolute Deviation
    threshold = k * mad

    # Identify bad points as those deviating significantly from their local neighborhood
    bad_point_mask = local_deviation > threshold

    # Apply a median filter to the entire phase profile
    filtered_phase = median_filter(phase_profile, size=filter_size)

    # Replace bad points with the filtered values
    phase_profile_cleaned = np.copy(phase_profile)
    phase_profile_cleaned[bad_point_mask] = filtered_phase[bad_point_mask]

    return phase_profile_cleaned, bad_point_mask

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clean 2D phase profile with bad point detection and replacement.")
    parser.add_argument("--file_hdf5", type=str, required=True, help="Full path to the input HDF5 file.")
    parser.add_argument("--file_json", type=str, required=True, help="Full path to the JSON file with metadata.")
    parser.add_argument("--file_hdf5_cleaned", type=str, required=True, help="Full path to save the cleaned HDF5 file.")
    parser.add_argument("--k", type=float, default=10.0, help="Absolute threshold for bad point detection.")
    parser.add_argument("--filter_size", type=int, default=3, help="Filter size for the median filter.")
    parser.add_argument("--original_px", type=float, required=True, help="Original pixel size in meters.")

    args = parser.parse_args()

    with h5py.File(args.file_hdf5, "r") as hdf5_file:
        # Read all datasets and attributes into memory
        datasets = {key: hdf5_file[key][:] for key in hdf5_file.keys()}
        attributes = {key: dict(hdf5_file[key].attrs) for key in hdf5_file.keys()}

    with open(args.file_json, "r") as json_file:
        metadata = json.load(json_file)
    
    # Extract parameters from metadata
    p_x = metadata.get("p_x")
    p_y = metadata.get("p_y")
    d = metadata.get("d")
    wavelength = metadata.get("wavelength")
    if p_x is None or p_y is None or d is None or wavelength is None:
        raise ValueError("Missing required parameters in JSON file: 'p_x', 'p_y', 'd', 'wavelength'")

    cleaned_datasets = {}
    bad_points_masks = {}
    for key in ['displace_x', 'displace_y']:
        if key in datasets:
            print(f"Processing dataset '{key}'...")
            cleaned_data, bad_points_mask = clean_phase_profile_local(datasets[key], k=args.k, filter_size=args.filter_size)
            cleaned_datasets[key] = cleaned_data
            bad_points_masks[key] = bad_points_mask
        else:
            print(f"Dataset '{key}' not found in the HDF5 file.")    

    if 'displace_x' in cleaned_datasets and 'displace_y' in cleaned_datasets:
        displace_x = cleaned_datasets['displace_x']
        displace_y = cleaned_datasets['displace_y']

        DPC_y = (displace_y - np.mean(displace_y)) * args.original_px / d
        DPC_x = (displace_x - np.mean(displace_x)) * args.original_px / d

        phase = frankotchellappa(DPC_x, DPC_y, p_x, p_y) * 2 * np.pi / wavelength

        cleaned_datasets['DPC_x'] = DPC_x
        cleaned_datasets['DPC_y'] = DPC_y
        cleaned_datasets['phase'] = phase

    with h5py.File(args.file_hdf5_cleaned, "w") as hdf5_cleaned_file:
        for key, data in datasets.items():
            if key in cleaned_datasets:
                dataset = hdf5_cleaned_file.create_dataset(key, data=cleaned_datasets[key])
            else:
                dataset = hdf5_cleaned_file.create_dataset(key, data=data)
            # Restore attributes
            for attr_key, attr_value in attributes[key].items():
                dataset.attrs[attr_key] = attr_value

    print(f"Cleaned HDF5 file saved to {args.file_hdf5_cleaned}.")

    if 'displace_x' in bad_points_masks and 'displace_y' in bad_points_masks:
        output_dir = os.path.dirname(args.file_hdf5)
        output_path = os.path.join(output_dir, "bad_points_overlay.png")

        plt.figure(figsize=(12, 6))

        # Plot displace_x with bad points overlay
        plt.subplot(1, 2, 1)
        plt.title("Bad Points Overlay on displace_x")
        plt.imshow(datasets['displace_x'], cmap='gray', origin='upper')
        plt.colorbar(label="Displacement Value")
        bad_y, bad_x = np.where(bad_points_masks['displace_x'])
        plt.scatter(bad_x, bad_y, color='red', s=10, label='Bad Points')
        plt.legend()

        # Plot displace_y with bad points overlay
        plt.subplot(1, 2, 2)
        plt.title("Bad Points Overlay on displace_y")
        plt.imshow(datasets['displace_y'], cmap='gray', origin='upper')
        plt.colorbar(label="Displacement Value")
        bad_y, bad_x = np.where(bad_points_masks['displace_y'])
        plt.scatter(bad_x, bad_y, color='red', s=10, label='Bad Points')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    print(f"Cleaned HDF5 file saved to {args.file_hdf5_cleaned}.")