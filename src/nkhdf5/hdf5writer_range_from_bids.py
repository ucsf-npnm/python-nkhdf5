""" hdf5writer_range_from_bids.py

Create hdf5 file given a start timestamp (start_p) and an end timestamp (end_p).
Stitches and clips directly from HDF5-BIDS files, creates files with original timestamps.

Created: 2026-10-10
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import os
import numpy as np
import h5py

# Third-Party Packages
from nkhdf5 import hdf5nk
HDF5NK = hdf5nk.HDF5NK_0_1_0

# Custom Packages 
import config
from timestamps_conversion import datetime_to_unixnano, get_original_dt
from dotenv import load_dotenv

# Initialize session parameters from environmental config files
load_dotenv() 

# Vectorize conversion functions for fast element-wise operations
v_get_original_dt = np.vectorize(get_original_dt) 
v_datetime_to_unixnano = np.vectorize(datetime_to_unixnano) 


def loc_files(indir: str, ref: str, start_window: datetime, end_window: datetime) -> Tuple[List[Tuple[str, int, int]], Dict[str, Any], np.ndarray]:
    """
    Locates, validates, and calculates chunk-wise slice boundaries for HDF5-BIDS files falling within a target time window.

    Args:
        indir (str): Path to input directory containing HDF5-BIDS files.
        ref (str): Reference date matching format 'YYYY-MM-DD' required by get_original_dt to retrieve original time axis.
        start_window (datetime): The absolute chronological start of the target time window. This will be the timestamp of first sample in the output file.
        end_window (datetime): The absolute chronological end of the target time window. This timestamp won't be included in the output file. 

    Outputs:
        tuple: A compound tuple object containing:
            - valid_files (list of tuples): Confirmed valid HDF5-BIDS files with their slicing indexes. Structured as [(file_path, start_idx, end_idx), ...].
            - global_meta (dict): Global metadata mapping schema across valid files.
            - ref_channel_labels (np.ndarray): Validated, matching array of intracranial channel labels (mandatory dataset in output file).
    """
    in_dir_path = Path(indir)
    files = sorted([f for f in in_dir_path.iterdir() if f.is_file() and 'h5' in f.name]) # Extract list of all HDF5-BIDS files stored in indir

    if not files:
        raise ValueError("ERROR: No HDF5 files found in specified directory.")
    
    # Initialize target dictionary for verified global metadata
    meta_tracking = {k: set() for k in ["subject", "filter_lowpass", "filter_highpass", "sample_rate"]}
    valid_files_init = [] 
    
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as f:

                file_start = get_original_dt(ref, f.attrs["start"]) 
                file_end = get_original_dt(ref, f.attrs["end"]) 
                file_date = file_start.date() 

                # Skip file if its date does not span the target date(s)
                if (file_date < start_window.date()) or (file_date > end_window.date()):
                    continue

                # Skip file if it ends prior to the target start window
                if file_end < start_window:
                    continue

                # Break out early if file starts after target end window
                if file_start > end_window:
                    break 
                
                # If file is valid, add its path to list and its global metadata to dictionary
                valid_files_init.append(str(file_path))
                meta_tracking["subject"].add(f.attrs["subject_id"])
                meta_tracking["filter_lowpass"].add(f["intracranialEEG"].attrs["filter_lowpass"])
                meta_tracking["filter_highpass"].add(f["intracranialEEG"].attrs["filter_highpass"])
                meta_tracking["sample_rate"].add(f["intracranialEEG_time_axis"].attrs["sample_rate"])

        except OSError:
            print(f"{file_path.name} can't be opened for reading.")

    if not valid_files_init:
        raise ValueError("ERROR: No raw files found to create output file.")

    # A set size > 1 implies there is mixed sampling frequency inside the target time window
    if any(len(vals) != 1 for vals in meta_tracking.values()):
        raise ValueError("ERROR: Files to merge do not share the same metadata [subject, lowpass_filter, highpass_filter, sample_rate]")

    global_meta = {k: next(iter(v)) for k, v in meta_tracking.items()}
    valid_files = [] 
    total_output_samples = 0

    start_window_int = np.datetime64(start_window, 'ns').view(np.int64)
    end_window_int = np.datetime64(end_window, 'ns').view(np.int64)
    for valid_file in valid_files_init:
        with h5py.File(valid_file, "r") as f:
            timestamps_dset = v_get_original_dt(ref, np.array(f["intracranialEEG_time_axis"]))

        timestamps_ints = np.array(timestamps_dset, dtype='datetime64[ns]').view(np.int64)
        del timestamps_dset
        
        # Binary search extracts half-open bounding indices directly without heap fragmentation
        crop_start = np.searchsorted(timestamps_ints, start_window_int, side='left') 
        crop_end = np.searchsorted(timestamps_ints, end_window_int, side='left')
        del timestamps_ints

        if crop_start < crop_end:
            valid_files.append((valid_file, crop_start, crop_end)) 
            total_output_samples += (crop_end - crop_start) 
    
    # Audit final slice sizes to protect against unmapped sample drops
    total_expected_samples = int((end_window - start_window).total_seconds() * global_meta["sample_rate"])
    if total_output_samples != total_expected_samples:
        raise ValueError(f"ERROR: Expected {total_expected_samples} samples, but got {total_output_samples}")

    if not valid_files:
        raise ValueError("ERROR: No valid data slices found within the target window.")

    # Validate intracranial EEG channels labels array (same number and position of labels)
    with h5py.File(valid_files[0][0], "r") as f:
        ref_channel_labels = np.array(f["intracranialEEG_channellabel_axis"])
    for valid_file, _, _ in valid_files:
        with h5py.File(valid_file, "r") as f:
            current_labels = np.array(f["intracranialEEG_channellabel_axis"])
            if not np.array_equal(ref_channel_labels, current_labels):
                raise ValueError(f"ERROR: File {valid_file} does not share the same channel labels or exact index positioning as the reference file axis")

    print(f"Found files to merge and confirmed output target size: {total_output_samples} samples.")

    return valid_files, global_meta, ref_channel_labels

def _write_dataset_channels(dset_name: str, files_to_merge: List[Tuple[str, int, int]], file_data_handle: Any, output_metadata: Dict[str, Any], ref: str, chunk_size: int, valid_labels: Optional[np.ndarray] = None):

    """
    Internal helper to validate channel metadata consistency and append block-wise timeseries arrays to an active HDF5 volume endpoint.

    Args:
        dset_name (str): Name of target dataset group (only options: 'intracranialEEG', 'DCChannel', 'scalpEEG' or 'EKG').
        files_to_merge (list of tuples): Confirmed valid HDF5-BIDS files with their slicing indexes. Structured as [(file_path, start_idx, end_idx), ...].
        file_data_handle (Any): Open dataset node handler object provided by HDF5NK schema.
        output_metadata (dict): Global metadata mapping schema across files.
        ref (str): Reference date matching format 'YYYY-MM-DD' required by get_original_dt to retrieve original time axis.
        chunk_size (int): Maximum number of samples to write in output file during each writing block.
        valid_labels (np.ndarray, optional): Validated, matching array of intracranial channel labels (mandatory dataset in output file).
    """

    if len(files_to_merge)!=1:
        # Validate channels labels array of target dataset (same number and position of labels)
        if valid_labels is not None:
            ref_labels = valid_labels
        else:
            with h5py.File(files_to_merge[0][0], "r") as f:
                ref_labels = np.array(f[f"{dset_name}_channellabel_axis"])
            for valid_file, _, _ in files_to_merge:
                with h5py.File(valid_file, "r") as f:
                    if not np.array_equal(ref_labels, np.array(f[f"{dset_name}_channellabel_axis"])):
                        raise ValueError(f"ERROR: File {valid_file} does not share the same channel labels or positioning for {dset_name}")
    else:
        if valid_labels is not None:
            ref_labels = valid_labels
        else:
            with h5py.File(files_to_merge[0][0], "r") as f:
                ref_labels = np.array(f[f"{dset_name}_channellabel_axis"])

    if len(ref_labels) == 0:
        raise ValueError(f"Missing {dset_name} dataset axis dimensions, skipping writing sequence.")

    print(f"Writing {dset_name} dataset...")
    for file_path, start_idx, end_idx in files_to_merge:
        with h5py.File(file_path, "r") as f:
            # Get original time axis array
            timestamps_dset = v_get_original_dt(ref, np.array(f[f"{dset_name}_time_axis"]))
            
            # Stream the cropped region in safe chunks
            for i in range(start_idx, end_idx, chunk_size):
                chunk_end = min(i + chunk_size, end_idx)

                # Load chunks directly into RAM
                chunk_timestamps = v_datetime_to_unixnano(timestamps_dset[i:chunk_end])
                chunk_signals = np.array(f[dset_name][i:chunk_end, :])

                # Copy directly into pre-allocated memory slot
                file_data_handle.append(chunk_signals, component_kwargs={"timeseries": {"data": chunk_timestamps}})

    # Write out core structural attributes to target dataset
    file_data_handle.axes[1]["channellabel_axis"].append(ref_labels)
    file_data_handle.attributes["filter_lowpass"] = output_metadata["filter_lowpass"]
    file_data_handle.attributes["filter_highpass"] = output_metadata["filter_highpass"]
    file_data_handle.attributes["channel_count"] = ref_labels.shape[0]
    file_data_handle.axes[0]["time_axis"].attributes["sample_rate"] = output_metadata["sample_rate"]
    file_data_handle.axes[0]["time_axis"].attributes["time_zone"] = "US/Pacific"

def write_hdf5(subject: str, indir: str, outdir: str, start_window: str, end_window: str,
               day1: str, ref: str, task_label: Optional[str], time_label: Optional[str],
               incl_dc=False, incl_scalp=False, incl_all=False, chunk_size=100_000):
    
    """
    Stitches continuous segments from HDF5-BIDS files falling within the target time window and write new HDF5 file (not deidentified).

    Args:
        subject (str): Presidio study subject ID.
        indir (str): Path to input directory containing HDF5-BIDS files.
        outdir (str): Path to output directory where the output file(s) will be written.
        start_window (str): The chronological start of the target time window matching format 'YYYY-MM-DD HH:MM:SS.f'. This will be the timestamp of first sample in the output file. 
        end_window (str): The chronological end of the target time window matching format 'YYYY-MM-DD HH:MM:SS.f'. This will be the absolute end of output file, not the same as the timestamp of the last sample.
        day1 (str): Baseline recording anchor day baseline string matching format 'YYYY-MM-DD'.
        ref (str): Reference date matching format 'YYYY-MM-DD' required by get_original_dt to retrieve original time axis.
        task_label (str or None): Descriptive task label for file naming.
        time_label (str or None): Optional timestamp label used to tag output names. Defaults to the timestamp of first sample if empty.
        incl_dc (bool, optional): Activates logic checkpoints checking and writting DC channels dataset. Defaults to False.
        incl_scalp (bool, optional): Activates logic checkpoints checking and writting scalp EEG dataset. Defaults to False.
        incl_all (bool, optional): Activates logic checkpoints checking and writting all available datasets. Defaults to False.
        chunk_size (int, optional): Maximum number of samples to write in output file during each writing block. Defaults to 100_000.
    """

    try:
        output_start = datetime.strptime(start_window, "%Y-%m-%d %H:%M:%S.%f")
        output_end = datetime.strptime(end_window, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        raise ValueError("ERROR: Incorrect format for either start_window or end_window, should be YYYY-MM-DD HH:MM:SS.f")

    try:
        s1_day1 = datetime.strptime(day1, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("ERROR: Incorrect format for day1 date, should be YYYY-MM-DD")

    try:
        datetime.strptime(ref, "%Y-%m-%d")
    except ValueError:
        raise ValueError("ERROR: Incorrect format for ref date, should be YYYY-MM-DD")
    
    # Generate file naming tags representing study experiment days
    output_startdate = output_start.date()
    diff = (output_startdate - s1_day1).days
    day_label = f"day{str(diff + 1).zfill(2)}"

    if task_label is None:
        raise ValueError("ERROR: task_label variable can't be empty, please assign a task identifier.")
    if time_label is None:
        time_label = output_start.strftime("%H%M%S")

    file_name = f"sub-{subject}_ses-stage1_task-{task_label}_acq-{day_label}_run-{time_label}_ieeg.h5"
    file_out = Path(outdir, file_name)

    if file_out.is_file():
        raise ValueError(f"{file_name} was already created and saved in {outdir}")
    
    print(f"{file_name} does not exist in {outdir}")

    print("Locating files to merge and running checkpoints...")
    files_to_merge, output_metadata, channellabels_dset = loc_files(indir=indir,
                                                                    ref=ref,
                                                                    start_window=output_start,
                                                                    end_window=output_end)
    print("Creating output file...")
    f_obj = HDF5NK(file=file_out, mode="a", create=True, construct=True)
    try:
        # Commit global file header attributes
        f_obj.attributes["subject_id"] = output_metadata["subject"]
        f_obj.attributes["start"] = datetime_to_unixnano(output_start)
        f_obj.attributes["end"] = datetime_to_unixnano(output_end)

        # Write Intracranial EEG signals (Mandatory)
        _write_dataset_channels(dset_name="intracranialEEG",
                                files_to_merge=files_to_merge,
                                file_data_handle=f_obj["data_ieeg"],
                                output_metadata=output_metadata,
                                ref=ref,
                                chunk_size=chunk_size,
                                valid_labels=channellabels_dset)
        
        # Write DC Channels signals (Optional)
        if incl_dc or incl_all:
            _write_dataset_channels(dset_name="DCChannel",
                                    files_to_merge=files_to_merge,
                                    file_data_handle=f_obj["data_ttl"],
                                    output_metadata=output_metadata,
                                    ref=ref,
                                    chunk_size=chunk_size)
            
        # Write Scalp EEG signals (Optional)
        if incl_scalp or incl_all:
            _write_dataset_channels(dset_name="scalpEEG",
                                    files_to_merge=files_to_merge,
                                    file_data_handle=f_obj["data_scalpeeg"],
                                    output_metadata=output_metadata,
                                    ref=ref,
                                    chunk_size=chunk_size)
            
        # White EKG/EOG/EMG Modalities (Optional)
        if incl_all:
            _write_dataset_channels(dset_name="EKG",
                                    files_to_merge=files_to_merge,
                                    file_data_handle=f_obj["data_ekg"],
                                    output_metadata=output_metadata,
                                    ref=ref,
                                    chunk_size=chunk_size)
            
        print(f"SUCCESS! {file_name} was created and saved in {outdir}")

    except Exception as e:
        print(f"CRITICAL ERROR encountered during write operations: {e}")
        raise e

    # Guarantees file locks close cleanly even if a failure trips mid-stream
    finally:
        f_obj.close()
        print("HDF5 file handle closed safely.")

if __name__ == "__main__":

    start_run = datetime.now()
    print(f"\n----------- starting run {start_run.strftime('%y-%m-%d %H:%M:%S')}\n")

    # -------------------------------------------------------------------
    # User-specified inputs (edit this block only)
    subject_id = "PR10" # PRESIDIO subject, everything in .env file and config.py is coded based on this variable
    out_dir = "/path/to/output/directory" # Output file(s) will be written here.
    start_period_list = ["2026-08-22 10:27:34.450000"] # List the start of the file(s) you want to generate, should be in format 'YYYY-MM-DD HH:MM:SS.f'
    end_period_list = ["2026-08-22 10:27:54.380000"] # List the end of the file(s) you want to generate, should be in format 'YYYY-MM-DD HH:MM:SS.f'
    task_label = "test" # Add descriptive label for file naming, e.g. "efficacy-LNAc4-LNAc5-6.0mA100Hz", "EP5-3.0mA1Hz", "biomarker-survey"
    time_label = None # USE ONLY WHEN: 1) generating one new file and 2) when you want a specific time in your file name, otherwise it will take the start time of the new file
    # # Besides intracranial EEG, what other channel types do you want to include in files?
    incl_dc = False # adds DC dataset
    incl_scalp = False # adds scalp dataset
    incl_all = True # adds DC, scalp and EKG/EOG/EMG datasets
    # -------------------------------------------------------------------

    # DO NOT MODIFY CODE BELOW

    # Subject-associated inputs (from config.py and .env files)
    in_dir = config.HDF5_DIR.replace("subject_id", subject_id) # Where raw HDF5-BIDS files are stored
    stage1_day1 = os.getenv(f"{subject_id}_STAGE1_DAY1") # Start of neural data acquisition; for file naming this date correspond to "day01".
    ref_date = os.getenv(f"{subject_id}_CONSENT_DATE") # Reference date to retrieve original timestamps

    if stage1_day1 is None or ref_date is None:
        raise ValueError("ERROR: Tracking values [stage1_day1, ref_date] cannot settle from system profile environments.")
    
    for start_p, end_p in zip(start_period_list, end_period_list):
        print(f"\nRUNNING {task_label}:")
        try:
            write_hdf5(subject=subject_id,
                       indir=in_dir,
                       outdir=out_dir,
                       start_window=start_p,
                       end_window=end_p,
                       day1=stage1_day1,
                       ref=ref_date,
                       task_label=task_label,
                       time_label=time_label,
                       incl_dc=incl_dc,
                       incl_scalp=incl_scalp,
                       incl_all=incl_all,
                       chunk_size=100_000)
            
        except ValueError as error:
            print(error)

    end_run = datetime.now()
    print(f"\nDone! Files in queue were processed.\n----------- ending run {end_run.strftime('%y-%m-%d %H:%M:%S')}")
    print(f"total runtime: {end_run - start_run}\n")
    print("")

""" End of code """