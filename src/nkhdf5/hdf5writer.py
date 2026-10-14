""" hdf5writer.py

Creates copies of EDF files in hdf5 format.
v5.0

"""

# Standard libraries
import os
import re
import pathlib
import pandas as pd
import numpy as np
import h5py
import mne
from edfio import read_edf
import scipy.io
from datetime import datetime, timedelta

# Third-Party Packages
from nkhdf5 import hdf5nk

# Local Packages 
HDF5NK = hdf5nk.HDF5NK_0_1_0
import config
from timestamps_conversion import normalize_dates, datetime_to_unixnano, get_original_dt
from dotenv import load_dotenv
load_dotenv() #load .env file

#Reformat channel labels as bytes object (acccepted in H5 schema)
def chs_labels_to_bytes(labels):
    split_list = []
    updated_list = []
    for label in labels:
        split_list.append(re.sub("[A-Za-z]+", lambda ele: "" + ele[0] + " ", label))
    split_list = [x.split(" ") for x in split_list]
    for label in split_list:
        updated_list.append(tuple(label))
    del split_list
    updated_list = np.array(updated_list, dtype = h5py.special_dtype(vlen=str))
    
    return updated_list

def detect_padding(signals_array):

    min_duration_in_samples = 29 #change if needed, it's enough to detect padding at the end
    # Check for padding at the end of EDF file
    channel_sum = np.sum(np.abs(signals_array), axis=0) # Absolute values and sum across all channels for each sample
    change_mask = np.diff(channel_sum) != 0
    change_indices = np.where(change_mask)[0] + 1
    boundaries = np.concatenate(([0], change_indices, [len(channel_sum)])) # Adds boundaries for the start and end of the entire array
    run_lengths = np.diff(boundaries) # Calculates the length (duration) of each consecutive run
    padded_runs = np.where(run_lengths >= min_duration_in_samples)[0] # Filters runs that meet or exceed the minimum duration

    del channel_sum, change_mask, change_indices, run_lengths # Delete large arrays

    padding_idxs = []
    for run_idx in padded_runs:
        start = boundaries[run_idx]
        end = boundaries[run_idx + 1]
        padding_idxs.extend(range(start, end))

    return padding_idxs

def write_hdf5(subject: str, filepath: str, outdir: str, day1: str, ref: str, mapping_key, path_to_coords="", incl_dc=False, incl_scalp = False, incl_all=False, chunk_size=100_000):

    # file_obj = input_obj1/input_obj2
    # file_start = output_start
    # file_name = output_filename

    v_normalize_dates = np.vectorize(normalize_dates) # Vectorize custom func: normalize_dates

    # Step 1: Read file and get start timestamp
    input_obj1 = read_edf(filepath, lazy_load_data=True) # raw EDF file
    output_startdate = input_obj1.startdate
    output_starttime = input_obj1.starttime
    output_start = datetime.combine(output_startdate, output_starttime)
    del input_obj1

    # Step 2: Prepare file name and check if it already exists
    ## 2.1 Prepare file name:
    try:
        s1_day1 = datetime.strptime(day1, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("ERROR: Incorrect format for day1 date, should be YYYY-MM-DD")

    diff = (output_startdate-s1_day1).days
    day_label = "day" + str(diff+1).zfill(2)
    time_label = output_start.strftime("%H%M%S")
    output_filename = f"sub-{subject}_ses-stage1_task-continuous_acq-{day_label}_run-{time_label}_ieeg.h5" # get file name ready
    file_out  = pathlib.Path(outdir, output_filename)

    ## 2.2 If output file already exists in outdir, raise error.
    if file_out.is_file()==True:
        raise ValueError(f"{output_filename} was already created and saved in {outdir}")
    
    print(f"{output_filename} does not exist in {outdir}")

    # Step 3: Get datasets attributes
    input_obj2 = mne.io.read_raw_edf(filepath, preload=False, verbose='WARNING')
    output_samplerate = input_obj2.info["sfreq"]
    output_lowpassfilter = input_obj2.info["lowpass"]
    output_highpassfilter = input_obj2.info["highpass"]

    # Step 4: Prepare time_axis dataset
    ## 4.1 Detect padding if present (based in voltage signal):
    signals = input_obj2[[11, 27, 60, 92, 124], :][0] # Load 5 ieeg channels to look for padding, otherwise it will be too heavy. 
    padding_indices = detect_padding(signals_array=signals)
    signals = None

    ## 4.2 Remove padding from timestamps array
    timestamps = input_obj2.times # not actual timestamps, time data is saved as seconds from the start of the file
    output_timestamps = np.delete(timestamps, list(padding_indices)) # removing padding
    timestamps = None
    valid_num_samples = len(output_timestamps)
    crop_start = 0
    crop_end = valid_num_samples
    
    ## 4.3 De-identify timestamps
    output_timestamps_us = (output_timestamps * 1e6).astype("timedelta64[us]") # Convert float seconds to integer microseconds
    output_timestamps = np.datetime64(output_start) + output_timestamps_us # Fast vectorized addition, replace previous array with actual timestamps (numpy.datetime format)
    output_timestamps = output_timestamps.astype('M8[us]').astype(object) # Convert timestamps to datetime format
    output_deidentified_timestamps = v_normalize_dates(ref, output_timestamps)
    output_timestamps = None

    ## 4.4 Calculate output file duration and end timestamp
    output_duration = valid_num_samples/output_samplerate # in seconds, maximum duration of an EDF file is 300 without padding (unless another duration was programmed in SDS while collecting data, e.g. PR01)
    output_end = output_start + timedelta(seconds=(output_duration)) # note: this won't be the same as the timestamp of the last sample, this is the absolute end 

    # Step 5: Prepare channellabels dataset
    output_channels = np.array(input_obj2.ch_names)
    output_channels = np.char.replace(output_channels, "POL ", "")
    output_channels = np.char.replace(output_channels, "-Ref", "")
    output_channels = np.char.replace(output_channels, " ", "")

    # Step 6: Create new file object and initiate datasets
    print("Creating output file...")
    f_obj = HDF5NK(file=file_out, mode="a", create=True, construct=True)
    f_obj.attributes["subject_id"] = subject
    f_obj.attributes["start"] = normalize_dates(ref, output_start)
    f_obj.attributes["end"] = normalize_dates(ref, output_end)

    file_data_ieeg = f_obj["data_ieeg"]
    file_data_ttl = f_obj["data_ttl"] if incl_dc is True or incl_all is True else None
    file_data_scalpeeg = f_obj["data_scalpeeg"] if incl_scalp is True or incl_all is True else None
    file_data_ekg = f_obj["data_ekg"] if  incl_all is True else None

    # Step 7: Write intracranialEEG dataset
    dset_name = "intracranialEEG"

    ## 7.1 Retrieve correct channel labels and their indices in output_channels array
    channellabels_dset = [i for i, type in zip(mapping_key.channel, mapping_key.channel_dset) if type==dset_name]
    channellabels_dset_index = [i for i, val in enumerate(output_channels) if val in channellabels_dset] # Map channels by channel type, output indexes
    channellabels_dset = chs_labels_to_bytes(output_channels[channellabels_dset_index]) # Actual order of labels in array + convert to bytes object

    ## 7.2 Stream the cropped region of this file in safe chunks
    print(f"Writing {dset_name} dataset...")
    for i in range(crop_start, crop_end, chunk_size):
        chunk_end = min(i + chunk_size, crop_end)

        chunk_signals_dset  = input_obj2[channellabels_dset_index, i:chunk_end][0]
        chunk_signals_dset = chunk_signals_dset.T

        chunk_timestamps_dset = output_deidentified_timestamps[i:chunk_end]

        # Copy directly into pre-allocated memory slot
        file_data_ieeg.append(chunk_signals_dset, component_kwargs={"timeseries": {"data": chunk_timestamps_dset}})

        # Clear memory until next chunk
        chunk_signals_dset, chunk_timestamps_dset = None, None

    ## 7.3 Append channels dataset and other attributes
    file_data_ieeg.axes[1]["channellabel_axis"].append(channellabels_dset)
    file_data_ieeg.attributes["filter_lowpass"]  = output_lowpassfilter
    file_data_ieeg.attributes["filter_highpass"] = output_highpassfilter
    file_data_ieeg.attributes["channel_count"]   = channellabels_dset.shape[0] #add channel count
    file_data_ieeg.axes[0]["time_axis"].attributes["sample_rate"] = output_samplerate
    file_data_ieeg.axes[0]["time_axis"].attributes["time_zone"] = "US/Pacific"

    channellabels_dset, channellabels_dset_index = None, None

    print(f"Done writing {dset_name} dataset and attributes:")
    print("     signals array size: ", f_obj["data_ieeg"].shape)
    print("     timestamps array size: ", f_obj["data_ieeg"].axes[0]["time_axis"].shape)
    print("     channels array size: ", f_obj["data_ieeg"].axes[1]["channellabel_axis"].shape)

    ## 7.4 Append electrodes coordinates if path to mat files was provided:
    if path_to_coords != "":
        elecs_coords = scipy.io.loadmat(path_to_coords)
        file_data_ieeg.axes[1]["channelcoord_axis"].append(elecs_coords["elecmatrix"])
        del elecs_coords
        print("Electrodes coordinates added to output file!")

    # Step 8: Write DCChannel dataset
    if file_data_ttl is not None:
        dset_name = "DCChannel"
        if dset_name not in list(mapping_key.channel_dset):
            raise ValueError(f"Set to include DC dataset, but no channels labeled as DCChannel in mapping key")
        ## 8.1 Retrieve correct channel labels and their indices in output_channels array
        channellabels_dset = [i for i, type in zip(mapping_key.channel, mapping_key.channel_dset) if type==dset_name]
        channellabels_dset_index = [i for i, val in enumerate(output_channels) if val in channellabels_dset] # Map channels by channel type, output indexes
        channellabels_dset = chs_labels_to_bytes(output_channels[channellabels_dset_index]) # Actual order of labels in array + convert to bytes object

        ## 8.2 Stream the cropped region of this file in safe chunks
        print(f"Writing {dset_name} dataset...")
        for i in range(crop_start, crop_end, chunk_size):
            chunk_end = min(i + chunk_size, crop_end)

            chunk_signals_dset  = input_obj2[channellabels_dset_index, i:chunk_end][0]
            chunk_signals_dset = chunk_signals_dset.T

            chunk_timestamps_dset = output_deidentified_timestamps[i:chunk_end]

            file_data_ttl.append(chunk_signals_dset, component_kwargs={"timeseries": {"data": chunk_timestamps_dset}})

            chunk_signals_dset, chunk_timestamps_dset = None, None

        ## 8.3 Append channels dataset and other attributes
        file_data_ttl.axes[1]["channellabel_axis"].append(channellabels_dset)
        file_data_ttl.attributes["filter_lowpass"]  = output_lowpassfilter
        file_data_ttl.attributes["filter_highpass"] = output_highpassfilter
        file_data_ttl.attributes["channel_count"]   = channellabels_dset.shape[0] #add channel count
        file_data_ttl.axes[0]["time_axis"].attributes["sample_rate"] = output_samplerate
        file_data_ttl.axes[0]["time_axis"].attributes["time_zone"] = "US/Pacific"

        channellabels_dset, channellabels_dset_index = None, None

        print(f"Done writing {dset_name} dataset and attributes:")
        print("     signals array size: ", f_obj["data_ttl"].shape)
        print("     timestamps array size: ", f_obj["data_ttl"].axes[0]["time_axis"].shape)
        print("     channels array size: ", f_obj["data_ttl"].axes[1]["channellabel_axis"].shape)

    # Step 9: Write scalpEEG dataset
    if file_data_scalpeeg is not None:
        dset_name = "scalpEEG"
        if dset_name not in list(mapping_key.channel_dset):
            raise ValueError(f"Set to include DC dataset, but no channels labeled as DCChannel in mapping key")
        ## 9.1 Retrieve correct channel labels and their indices in output_channels array
        channellabels_dset = [i for i, type in zip(mapping_key.channel, mapping_key.channel_dset) if type==dset_name]
        channellabels_dset_index = [i for i, val in enumerate(output_channels) if val in channellabels_dset] # Map channels by channel type, output indexes
        channellabels_dset = chs_labels_to_bytes(output_channels[channellabels_dset_index]) # Actual order of labels in array + convert to bytes object

        ## 9.2 Stream the cropped region of this file in safe chunks
        print(f"Writing {dset_name} dataset...")
        for i in range(crop_start, crop_end, chunk_size):
            chunk_end = min(i + chunk_size, crop_end)

            chunk_signals_dset  = input_obj2[channellabels_dset_index, i:chunk_end][0]
            chunk_signals_dset = chunk_signals_dset.T

            chunk_timestamps_dset = output_deidentified_timestamps[i:chunk_end]

            file_data_scalpeeg.append(chunk_signals_dset, component_kwargs={"timeseries": {"data": chunk_timestamps_dset}})

            chunk_signals_dset, chunk_timestamps_dset = None, None

        ## 9.3 Append channels dataset and other attributes
        file_data_scalpeeg.axes[1]["channellabel_axis"].append(channellabels_dset)
        file_data_scalpeeg.attributes["filter_lowpass"]  = output_lowpassfilter
        file_data_scalpeeg.attributes["filter_highpass"] = output_highpassfilter
        file_data_scalpeeg.attributes["channel_count"]   = channellabels_dset.shape[0] #add channel count
        file_data_scalpeeg.axes[0]["time_axis"].attributes["sample_rate"] = output_samplerate
        file_data_scalpeeg.axes[0]["time_axis"].attributes["time_zone"] = "US/Pacific"

        channellabels_dset, channellabels_dset_index = None, None

        print(f"Done writing {dset_name} dataset and attributes:")
        print("     signals array size: ", f_obj["data_scalpeeg"].shape)
        print("     timestamps array size: ", f_obj["data_scalpeeg"].axes[0]["time_axis"].shape)
        print("     channels array size: ", f_obj["data_scalpeeg"].axes[1]["channellabel_axis"].shape)

    # Step 10: Write EKG/EOG/EMG dataset
    if file_data_ekg is not None:
        dset_name = ["EKG", "EOG", "EMG"]
        if "EKG" not in list(mapping_key.channel_dset) or "EOG" not in list(mapping_key.channel_dset) or "EMG" not in list(mapping_key.channel_dset):
            raise ValueError(f"Set to include DC dataset, but no channels labeled as DCChannel in mapping key")
        ## 9.1 Retrieve correct channel labels and their indices in output_channels array
        channellabels_dset = [i for i, type in zip(mapping_key.channel, mapping_key.channel_dset) if type in dset_name]
        channellabels_dset_index = [i for i, val in enumerate(output_channels) if val in channellabels_dset] # Map channels by channel type, output indexes
        channellabels_dset = chs_labels_to_bytes(output_channels[channellabels_dset_index]) # Actual order of labels in array + convert to bytes object

        ## 9.2 Stream the cropped region of this file in safe chunks
        print(f"Writing {dset_name} dataset...")
        for i in range(crop_start, crop_end, chunk_size):
            chunk_end = min(i + chunk_size, crop_end)

            chunk_signals_dset  = input_obj2[channellabels_dset_index, i:chunk_end][0]
            chunk_signals_dset = chunk_signals_dset.T

            chunk_timestamps_dset = output_deidentified_timestamps[i:chunk_end]

            file_data_ekg.append(chunk_signals_dset, component_kwargs={"timeseries": {"data": chunk_timestamps_dset}})

            chunk_signals_dset, chunk_timestamps_dset = None, None

        ## 9.3 Append channels dataset and other attributes
        file_data_ekg.axes[1]["channellabel_axis"].append(channellabels_dset)
        file_data_ekg.attributes["filter_lowpass"]  = output_lowpassfilter
        file_data_ekg.attributes["filter_highpass"] = output_highpassfilter
        file_data_ekg.attributes["channel_count"]   = channellabels_dset.shape[0] #add channel count
        file_data_ekg.axes[0]["time_axis"].attributes["sample_rate"] = output_samplerate
        file_data_ekg.axes[0]["time_axis"].attributes["time_zone"] = "US/Pacific"

        channellabels_dset, channellabels_dset_index = None, None

        print(f"Done writing {dset_name} dataset and attributes:")
        print("     signals array size: ", f_obj["data_ekg"].shape)
        print("     timestamps array size: ", f_obj["data_ekg"].axes[0]["time_axis"].shape)
        print("     channels array size: ", f_obj["data_ekg"].axes[1]["channellabel_axis"].shape)

    f_obj.close() # Make sure to close object that was just created!!
    del input_obj2
    output_deidentified_timestamps = None
    
    print(f"SUCCESS! {output_filename} was created and saved in {outdir}")

# Main #
if __name__ == "__main__":
    start_run = datetime.now()
    print("")
    print(f"----------- starting run {start_run.strftime('%y-%m-%d %H:%M:%S')}")
    print("")

    # User-specified inputs
    subject_id  = "PR10" # PRESIDIO subject, everything in .env file and config.py is coded based on this variable
    # --- What's the output directory to store new files?
    out_dir = config.HDF5_DIR.replace("subject_id", subject_id)
    # out_dir = "/userdata/dastudillo/patient_data" #TESTING
    # --- Specify channels key to separate channel types (based on hdf5 schema)
    channels_mapping_key = pd.read_csv(f"/data_store2/presidio/nihon_kohden/{subject_id}/catalogs/{subject_id}_channels_h5dataset.csv")
    # --- Specify path to mat file with electrodes coordinates, if exists, otherwise comment out or leave as ""
    path_to_coords = f"/data_store2/imaging/subjects/{subject_id}/ses-prestage1/elecs/clinical_elecs_all.mat"
    # --- Besides intracranial EEG, what other channel types do you want to include in files?
    incl_dc = False # iEEG + DC datasets
    incl_scalp = False # iEEG + scalp datasets
    incl_all = True # iEEG + everything
    
    # Subject-associated inputs (from config.py and .env files)
    in_dir = config.EDF_DIR.replace("subject_id", subject_id) # Where raw hdf5 files are stored
    stage1_day1 = os.getenv(f"{subject_id}_STAGE1_DAY1") # Start of neural data acquisition; for file naming this date correspond to "day01".
    ref_date = os.getenv(f"{subject_id}_CONSENT_DATE") # Subject's consent date, use in timestamps de-identification

    if stage1_day1 is None:
        raise ValueError("ERROR: day1 can't be None")

    if ref_date is None:
        raise ValueError("ERROR: ref can't be None")

    if channels_mapping_key is None:
        raise ValueError("ERROR: Mapping key can't be empty")

    files = sorted(filter(lambda x: True if "edf" in x else False, os.listdir(in_dir)))[899:] #TESTING

    for file in files:
        print("")
        print(f"RUNNING EDF {file}:")

        try:
            file_path = os.path.join(in_dir, file)
            write_hdf5(subject=subject_id,
                       filepath=file_path,
                       outdir=out_dir,
                       day1=stage1_day1,
                       ref=ref_date,
                       mapping_key=channels_mapping_key,
                       path_to_coords=path_to_coords,
                       incl_dc=incl_dc,
                       incl_scalp=incl_scalp,
                       incl_all=incl_all,
                       chunk_size=100_000)

        except ValueError as error:
            print(error)

    print("")
    print("Done! Files in queue were created")
    end_run = datetime.now()
    print(f"----------- ending run {end_run.strftime('%y-%m-%d %H:%M:%S')}")
    print(f"total runtime: {end_run - start_run}")
    print("")
        
    """ End of code"""