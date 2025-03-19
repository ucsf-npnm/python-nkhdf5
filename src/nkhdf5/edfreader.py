"""
Read and extract metadata and timeseries from raw EDF file
v3.0
"""


# Import standard libraries #
import pandas as pd
import numpy as np
import os
import re
import pathlib
import subprocess
from datetime import datetime, timedelta
import mne
import pyedflib

#Define common labels for channel type
ieeg_chan = ["OFC", "SGC", "RA", "LA", "RH", "LH", "VC"]
dc_chan   = ["DC"]
ekg_chan  = ["EKG", "EOG"] #todo: create separate variables for EOG in the future, for now pooled with EKG
emg_chan  = ["EMG"]

#Extracts metadata and timeseries from each EDF file and compiles it into a dictionary 
def edf_reader(files_dir, filename):
    f = os.path.join(files_dir, filename)
    if os.path.isfile(f):
        error_found = False
        temp_dir = "/userdata/dastudillo/temp/" #temporary directory to save continuous EDF
        pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True) #create temporary directory if it doesn't exist
        bash_cmd = "./edfplcnv/edfplusdcnv --dest-dir=" + temp_dir + " " + f
        process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE) #run conversion of discontinuous to continuous EDF
        return_code = process.wait()

        if return_code != 0:
            error_found = True
            print("Error converting ", f)

        if error_found == False:
            temp_file = temp_dir + "/" + filename.replace(".edf", "_0001.edf")
            try:
                edf_f = pyedflib.EdfReader(temp_file)
                edf_start = edf_f.getStartdatetime()
                edf_f.close()
            except:
                print("Error opening ", f)
            os.remove(temp_file)

            #read EDF file using mne to extract timeseries and other parameters
            edf_obj = mne.io.read_raw_edf(f)
            edf_nsample = len(edf_obj)
            edf_sfreq = edf_obj.info["sfreq"]
            edf_speriod = 1/edf_sfreq
            edf_duration = timedelta(seconds=edf_nsample/edf_sfreq)
            edf_end = edf_start + edf_duration #ATTENTION: edf_len includes overlapping timestamps so edf_end will overlap with edf_start of next edf file
            edf_path = f
            edf_timezone = "US/Pacific"
            edf_lowpass = edf_obj.info["lowpass"] #low pass filter
            edf_highpass = edf_obj.info["highpass"] #high pass filter
            edf_nchan = edf_obj.info["nchan"] #number of total channels
            channel_labels = [ch.replace("POL ", "").replace("-Ref", "").replace(" ", "") for ch in edf_obj.ch_names]

            data_array, time_array = edf_obj[:,:]

            #Get time_array as datetime objects
            cumm_seconds_array = [x*edf_speriod for x in list(range(1,edf_nsample+1))]
            time_array_datetime = [edf_start + timedelta(seconds=x) for x in cumm_seconds_array]
            
            #Reformat channel labels as bytes object (acccepted in H5 schema)
            def convert_channel_labels(labels):
                split_list = []
                updated_list = []
                for label in labels:
                    split_list.append(re.sub("[A-Za-z]+", lambda ele: "" + ele[0] + " ", label))
                split_list = [x.split(" ") for x in split_list]
                for label in split_list:
                    updated_list.append(tuple(label))
                return updated_list

            channel_labels_bytes = convert_channel_labels(channel_labels)
            
            #Identify channel type (ieeg, scalp, ekg, etc, that you previously defined under common labels)
            def find_index(lst, condition):
                return [i for i, elem in enumerate(lst) if condition(elem)]
            def find_channel_type(types_lst, index_lst=[]):
                for ch_type in types_lst:
                    index_lst = index_lst + find_index(channel_labels, lambda e: True if ch_type in e else False)
                return index_lst

            ieeg_index_lst = find_channel_type(ieeg_chan)
            dc_index_lst = find_channel_type(dc_chan)
            ekg_index_lst = find_channel_type(ekg_chan)
            emg_index_lst = find_channel_type(emg_chan)
            not_scalp_index_lst = ieeg_index_lst + dc_index_lst + ekg_index_lst + emg_index_lst
            scalp_index_lst = list(set(list(range(len(channel_labels)))) - set(not_scalp_index_lst))

            channel_types = channel_labels.copy()
            for ieeg_index in ieeg_index_lst:
                channel_types[ieeg_index] = "intracranial EEG"
            for dc_index in dc_index_lst:
                channel_types[dc_index] = "TTL"
            for ekg_index in ekg_index_lst:
                channel_types[ekg_index] = "EKG"
            for emg_index in emg_index_lst:
                channel_types[emg_index] = "EMG"
            for scalp_index in scalp_index_lst:
                channel_types[scalp_index] = "scalp EEG"

            #Build dictionary object
            edf_dic = {
                    "edf_name": filename, #former edf_fn
                    "edf_start": edf_start,
                    "edf_end": edf_end,
                    "edf_timezone": edf_timezone,
                    "edf_duration": edf_duration,
                    "edf_nsample": edf_nsample,
                    "edf_sfreq": edf_sfreq,
                    "edf_speriod": edf_speriod,
                    "edf_path": edf_path,
                    "edf_lowpass": edf_lowpass,
                    "edf_highpass": edf_highpass,
                    "edf_nchan": edf_nchan,
                    "edf_chanlabels": channel_labels, #former edf_raw_chanlabs
                    "edf_chanlabels_bytes": channel_labels_bytes, #former edf_channellabel_axis
                    "edf_chantypes": channel_types, #former edf_chantype
                    "edf_axis": list(["chan","sample"]),
                    "edf_data": data_array,
                    "edf_time_array": time_array, #former edf_time_axis
                    "edf_time_datetime": time_array_datetime
                    }

     return edf_dic

print("EDF reader is ready to use")

"""End of code"""
