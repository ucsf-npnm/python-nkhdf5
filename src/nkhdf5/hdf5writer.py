"""hdf5writer.py
By providing a list of file paths, converts an EDF file to HDF5 file

v3.0

"""

# Standard Libraries #
import pathlib
import os
import numpy as np
import pandas as pd
import json
import datetime
from datetime import datetime, timedelta
import time
import h5py
import scipy.io
import ast

# Third-Party Packages #
from nkhdf5 import hdf5nk
HDF5NK = hdf5nk.HDF5NK_0_1_0

# Local Packages #
from edfreader import get_meastimestamp, edf_reader, normalize_dates

# Main #
if __name__ == "__main__":
    ## User-specified inputs
    subject_id  = "PR04"

    with open(f"/userdata/dastudillo/keys/subjects.json", "r") as f: #stored in user's directory, not part of repo files
        subjects = json.load(f)
    
    edf_dir = subjects[subject_id]["edf_dir"] #directory where raw EDF are stored
    outdir = subjects[subject_id]["BIDS_raw_stage1"] #directory where HDF5 files will be store
    stage1_day1 = datetime.strptime(subjects[subject_id]["stage1_day1"], "%Y-%m-%d").date() #for file naming, this date sets "day01"

    elecscoor_on = True #choose if you want to add electrode coordinates 
    elecscoor_file = subjects[subject_id]["eleccoor_file"] #file containing electrodes coordinates

    normalize_dates_on = True #choose if you want timestamps deidentified (keep local time, normalize date to subject's consent date)
    ref_date = datetime.strptime(subjects[subject_id]["consent_date"], "%Y-%m-%d")  #add consent date as reference to normalize dates, comment out if above is false

    ## Extract list of EDF files stored in directory
    edf_files = sorted(filter(lambda x: True if "edf" in x else False, os.listdir(edf_dir)))

    ## Start of actual code, loop through edf files
    for edf_file in edf_files:
        edfstart = get_meastimestamp(edf_dir, edf_file) #get measurement datetime (start of recording) using pyedflib, use this to name output file and check if it already exist
        
        edf_start_dt = edfstart.date()
        diff = (edf_start_dt-stage1_day1).days
        day_label = "day" + str(diff+1).zfill(2)
        time_label = edfstart.strftime("%H%M%S")

        file_name    = f"sub-{subject_id}_ses-stage1_task-continuous_acq-{day_label}_run-{time_label}_ieeg.h5" #get file name ready
        file_out     = pathlib.Path(outdir, file_name) #full path for new file

        if file_out.is_file()==True:
            print("")
            print(f"{edf_file} has already been converted to HDF5 as {file_name}")
            print("")
            print("Checking next file...")
            print("")

        if file_out.is_file()==False:
            edf_contents = edf_reader(edf_dir, edf_file, edfstart) #dictionary with metadata+timeseries
            
            ### Extract timeseries by channel type
            ieeg_array = np.array([k for k,v in zip(edf_contents["hdf5_data"], edf_contents["edf_chantypes"]) if v == "intracranial EEG"]).T
            scalpeeg_array = np.array([k for k,v in zip(edf_contents["hdf5_data"], edf_contents["edf_chantypes"]) if v == "scalp EEG"]).T
            ekg_array = np.array([k for k,v in zip(edf_contents["hdf5_data"], edf_contents["edf_chantypes"]) if v == "EKG"]).T
            ttl_array = np.array([k for k,v in zip(edf_contents["hdf5_data"], edf_contents["edf_chantypes"]) if v == "TTL"]).T

            ### Reformat datetime objects to unix nanoseconds
            if normalize_dates_on == True:
                print("")
                print("Deidentifying/normalizing dates in timeseries...")
                print("")
                start_unix = normalize_dates(ref_date, edf_contents["hdf5_start"]) #nanoseconds
                end_unix = normalize_dates(ref_date, edf_contents["hdf5_end"]) #nanoseconds
                time_array_unix = np.array([normalize_dates(ref_date, dt) for dt in edf_contents["hdf5_time_datetime"]])

            if normalize_dates_on == False:
                start_unix = int(1e9 * edf_contents["hdf5_start"].timestamp())
                end_unix = int(1e9 * edf_contents["hdf5_end"].timestamp())
                time_array_unix = np.array([int(1e9 * dt.timestamp()) for dt in edf_contents["hdf5_time_datetime"]]) 
                
            ### Extract channel labels by channel type in format accepted by class 
            chanlabs_ieeg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "intracranial EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_scalpeeg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "scalp EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ekg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "EKG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ttl_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "TTL"], dtype = h5py.special_dtype(vlen=str))

            ### Create the file
            print("Creating HDF5 file...")
            print("")
            f_obj = HDF5NK(file=file_out, mode="a", create=True, construct=True)
            f_obj.attributes["subject_id"] = subject_id
            f_obj.attributes["start"]      = start_unix
            f_obj.attributes["end"]        = end_unix
            
            print("Writing ieeg data...")
            print("")
            file_data_ieeg = f_obj["data_ieeg"]
            file_data_ieeg.append(ieeg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
            file_data_ieeg.axes[1]["channellabel_axis"].append(chanlabs_ieeg_array)

            if elecscoor_on == True:
                elecscoor_mat = scipy.io.loadmat(elecscoor_file) # Extract electrodes coordinates (only for depth electrodes, data_ieeg)
                elecscoor = elecscoor_mat["elecmatrix"]
                file_data_ieeg.axes[1]["channelcoord_axis"].append(elecscoor)

            file_data_ieeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
            file_data_ieeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
            file_data_ieeg.attributes["channel_count"]   = ieeg_array.shape[1]
            file_data_ieeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
            file_data_ieeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(scalpeeg_array)!=0:
                print("Writing scalp eeg data...")
                print("")
                file_data_scalpeeg = f_obj["data_scalpeeg"]
                file_data_scalpeeg.append(scalpeeg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
                file_data_scalpeeg.axes[1]["channellabel_axis"].append(chanlabs_scalpeeg_array)

                file_data_scalpeeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_scalpeeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_scalpeeg.attributes["channel_count"]   = scalpeeg_array.shape[1]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ekg_array)!=0:
                print("Writing ekg data...")
                print("")
                file_data_ekg = f_obj["data_ekg"]
                file_data_ekg.append(ekg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
                file_data_ekg.axes[1]["channellabel_axis"].append(chanlabs_ekg_array)

                file_data_ekg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_ekg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_ekg.attributes["channel_count"]   = ekg_array.shape[1]
                file_data_ekg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_ekg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ttl_array)!=0:
                print("Writing DC channels data...")
                file_data_ttl = f_obj["data_ttl"]
                file_data_ttl.append(ttl_array, component_kwargs={"timeseries": {"data": time_array_unix}})
                file_data_ttl.axes[1]["channellabel_axis"].append(chanlabs_ttl_array)

                file_data_ttl.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_ttl.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_ttl.attributes["channel_count"]   = ttl_array.shape[1]
                file_data_ttl.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_ttl.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            print("")
            print(f"{edf_file} saved as: ", file_name)
            print("")
        
            f_obj.close()


    print("Conversion completed!")
    print("")


"""End of code"""
