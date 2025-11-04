"""
Read and extract metadata and timeseries from raw EDF file
v4.0 (uses edfio instead of pyedflib package)
"""


# Import standard libraries #
import pandas as pd
import numpy as np
import os
import re
import pathlib
import subprocess
from datetime import datetime, timedelta
import pytz
from pytz import timezone
import mne
from edfio import read_edf

#Set maximum duration of EDF file (in seconds, regardless of sampling frequency)
edf_maxduration = 300

#Define common labels for channel type
#ieeg_chan = ["OFC", "SGC", "RA", "LA", "RH", "LH", "VC"] #up to PR06
ieeg_chan = ["OFC", "SGC", "RA", "LA", "RH", "LH", "NAc", "BNS", "RMD", "LMD"] #PR07 onwards
dc_chan   = ["DC"]
ekg_chan  = ["EKG", "EOG"] #todo: create separate variables for EOG in the future, for now pooled with EKG
emg_chan  = ["EMG"]
#channels_to_remove = ["BP1"] #PR07

def get_meastimestamp(files_dir, filename):
    error_files = []
    f = os.path.join(files_dir, filename)
    if os.path.isfile(f):
        try:
            edf_f = read_edf(f) #read EDF file
            start_f = datetime.combine(edf_f.startdate, edf_f.starttime)
        except:
            print("Error opening ", f)
            error_files.append(f)
            start_f = None
    return start_f

#Extracts metadata and timeseries from each EDF file and compiles it into a dictionary 
def edf_reader(files_dir, filename, file_start):
    f = os.path.join(files_dir, filename)
    edf_start = file_start

    #read EDF file using mne to extract timeseries and other parameters
    edf_obj = mne.io.read_raw_edf(f)
    edf_nsample = len(edf_obj)
    edf_sfreq = edf_obj.info["sfreq"]
    edf_speriod = 1/edf_sfreq
    edf_duration = edf_nsample/edf_sfreq
    edf_end = edf_start + timedelta(seconds=edf_duration) #ATTENTION: edf_len includes overlapping timestamps so edf_end will overlap with edf_start of next edf file
    edf_path = f
    edf_timezone = "US/Pacific"
    edf_lowpass = edf_obj.info["lowpass"] #low pass filter
    edf_highpass = edf_obj.info["highpass"] #high pass filter

    edf_nchan = edf_obj.info["nchan"] #number of channels
    channel_labels = [ch.replace("POL ", "").replace("-Ref", "").replace(" ", "") for ch in edf_obj.ch_names]
    edf_data_array, edf_time_array = edf_obj[:,:]

    #####Remove channels if needed#####
#    raw_channel_labels = [ch.replace("POL ", "").replace("-Ref", "").replace(" ", "") for ch in edf_obj.ch_names]
#    raw_edf_data_array, edf_time_array = edf_obj[:,:]
#    remove_idx = []
#    for ch in channels_to_remove:
#        remove_idx.append(raw_channel_labels.index(ch))
#
#    edf_data_array = np.delete(raw_edf_data_array, remove_idx, axis=0)
#    channel_labels = raw_channel_labels.copy()
#    for ch in channels_to_remove:
#        channel_labels.remove(ch)
#
#    edf_nchan = len(channel_labels)
    #######################

    #Remove buffer period from data arrasy if edf_duration is over edf_maxduration
    h5_start = edf_start
    if edf_duration > edf_maxduration:
        max_loc = int(edf_maxduration*edf_sfreq)
        h5_data_array = edf_data_array[:,:max_loc]
        h5_time_array = edf_time_array[:max_loc]
        h5_duration = float(len(h5_time_array)/edf_sfreq)
        h5_end = h5_start + timedelta(seconds=h5_duration)
    if edf_duration <= edf_maxduration:
        h5_data_array = edf_data_array.copy()
        h5_time_array = edf_time_array.copy()
        h5_duration = edf_duration
        h5_end = edf_end

    #Get time_array as datetime objects
    edf_time_array_datetime = [edf_start + timedelta(seconds=x) for x in edf_time_array]
    h5_time_array_datetime = [h5_start + timedelta(seconds=x) for x in h5_time_array]

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
            "edf_data": edf_data_array,
            "edf_time_array": edf_time_array, #former edf_time_axis
            "edf_time_datetime": edf_time_array_datetime,
            "hdf5_start": h5_start,
            "hdf5_end": h5_end,
            "hdf5_duration": h5_duration,
            "hdf5_data": h5_data_array,
            "hdf5_time_array": h5_time_array,
            "hdf5_time_datetime": h5_time_array_datetime
            }
            
    edf_obj.close()

    return edf_dic

#de-identify dates in timestamps
def normalize_dates(ref_dt, dt): #both inputs have
    ref_dt_utc = ref_dt.replace(tzinfo=pytz.timezone('UTC'))
    timestamp_nanoseconds = int(1e9 * (dt.timestamp()-ref_dt_utc.timestamp()))
    return timestamp_nanoseconds

print("")
print("EDF reader is ready to use")
print("")

"""
End of code
"""
