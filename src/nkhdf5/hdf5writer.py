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
import time
import h5py
import scipy.io
import ast

# Third-Party Packages #
from nkhdf5 import hdf5nk
HDF5NK = hdf5nk.HDF5NK_0_1_0

# Local Packages #
from edfreader import edf_reader
from concatenator_tools import FilesForBiomarker

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    subject_id  = "PR06"
    with open("/userdata/dastudillo/subjects.json", "r") as f:
        subjects = json.load(f)

    edf_dir = subjects[subject_id]["edf_dir"]
    outdir = subjects[subject_id]["h5_dir"]
    eleccoor_file = subjects[subject_id]["eleccoor_file"]

    ## Extract list of EDF files stored in directory
    edf_files = sorted(filter(lambda x: True if "edf" in x else False, os.listdir(edf_dir)))
    
    ## Start of actual code, loop through edf files
    for edf_file in edf_files:
        edf_contents = edf_reader(edf_dir, edf_file) #dictionary with metadata+timeseries
        date_string  = edf_contents["edf_start"].strftime("%Y%m%d") #date use in filename
        time_string  = edf_contents["edf_start"].strftime("%H%M%S") #time use in filename
        file_name    = f"sub-{subject_id}_ses-stage1_task-continuous_acq-{date_string}_run-{time_string}_ieeg.h5" #get file name ready
        file_out     = pathlib.Path(outdir, file_name) #full path for new file

        if file_out.is_file()==True:
            print("")
            print(f"{edf_file} has already been converted to HDF5 as {file_name}")
            print("")
            print("Checking next file...")
            print("")

        if file_out.is_file()==False:
            ### Extract timeseries by channel type
            ieeg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantypes"]) if v == "intracranial EEG"]).T
            scalpeeg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantypes"]) if v == "scalp EEG"]).T
            ekg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantypes"]) if v == "EKG"]).T
            ttl_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantypes"]) if v == "TTL"]).T

            ### Reformat datetime objects to unix nanoseconds 
            time_array_unix = np.array([int(x.timestamp() * 1e9) for x in edf_contents["edf_time_datetime"]]) 

            ### Extract channel labels by channel type in format accepted by class 
            chanlabs_ieeg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "intracranial EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_scalpeeg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "scalp EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ekg_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "EKG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ttl_array = np.array([k for k,v in zip(edf_contents["edf_chanlabels_bytes"], edf_contents["edf_chantypes"]) if v == "TTL"], dtype = h5py.special_dtype(vlen=str))

            ### Extract electrodes coordinates (only for depth electrodes, data_ieeg)
            elecs_mat_file = scipy.io.loadmat(eleccoor_file)
            elecs_coor = elecs_mat_file["elecmatrix"]

            ### Create the file
            f_obj = HDF5NK(file=file_out, mode="a", create=True, construct=True)
            f_obj.attributes["subject_id"] = subject_id
            f_obj.attributes["start"]      = int(edf_contents["edf_start"].timestamp() * 1e9)
            f_obj.attributes["end"]        = int(edf_contents["edf_end"].timestamp() * 1e9)
    
            file_data_ieeg = f_obj["data_ieeg"]
            file_data_ieeg.append(ieeg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
            file_data_ieeg.axes[1]["channellabel_axis"].append(chanlabs_ieeg_array)
            file_data_ieeg.axes[1]["channelcoord_axis"].append(elecs_coor)

            file_data_ieeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
            file_data_ieeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
            file_data_ieeg.attributes["channel_count"]   = ieeg_array.shape[1]
            file_data_ieeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
            file_data_ieeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(scalpeeg_array)!=0:
                file_data_scalpeeg = f_obj["data_scalpeeg"]
                file_data_scalpeeg.append(scalpeeg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
                file_data_scalpeeg.axes[1]["channellabel_axis"].append(chanlabs_scalpeeg_array)

                file_data_scalpeeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_scalpeeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_scalpeeg.attributes["channel_count"]   = scalpeeg_array.shape[1]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ekg_array)!=0:
                file_data_ekg = f_obj["data_ekg"]
                file_data_ekg.append(ekg_array, component_kwargs={"timeseries": {"data": time_array_unix}})
                file_data_ekg.axes[1]["channellabel_axis"].append(chanlabs_ekg_array)

                file_data_ekg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_ekg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_ekg.attributes["channel_count"]   = ekg_array.shape[1]
                file_data_ekg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_ekg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ttl_array)!=0:
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

            #print("Converting next file...")
            #print("")
        #After closing check if the file exists #
        #print(f"File Exists: {out_path.is_file()}")
        #print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
        #print("")


"""End of code

"""
