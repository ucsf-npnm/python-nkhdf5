'''
Saves a metadata catalog csv of filename, start date, end date, shortened filename and file
lenght to the same directory level as the EDF directory provided.
'''


# Import standard libraries #
import os
import pandas as pd
import numpy as np
import subprocess
import mne
import re
from datetime import datetime, timedelta

#Define common labels for channel type
ieeg_chan = ['OFC', 'SGC', 'RA', 'LA', 'RH', 'LH', 'VC']
dc_chan   = ['DC']
ekg_chan  = ['EKG', 'EOG'] #todo: create separate variables for EOG in the future, for now pooled with EKG
emg_chan  = ['EMG']

#Gets list of all EDF files in patient's main folder 
def get_edf_list(edf_dir):
    edf_list = sorted(filter(lambda x: True if 'edf' in x else False, os.listdir(edf_dir)))
    return edf_list

#Extracts metadata and timeseries from each EDF file and compiles it into a dictionary 
def edf_reader(edf_dir, edf_fn):

    #Read EDF file
    raw = mne.io.read_raw_edf(os.path.join(edf_dir, edf_fn))
    
    #Extract available metadata
    edf_len = timedelta(seconds=len(raw)/raw.info['sfreq']) # seconds
    edf_start = raw.info['meas_date'] #ATTENTION: date and time are coded in local time, but mne incorrectly assigns UTC as timezone, maybe due to missing info in raw file(?)
    edf_end = edf_start + edf_len #ATTENTION: edf_len includes overlapping timestamps so edf_end will overlap with edf_start of next edf file
    ch_names_clean = [ch.split('-')[0].split('POL ')[1].replace(" ", "") for ch in raw.ch_names]

    #Identify channel type (ieeg, scalp, ekg, etc, that you previously defined under common labels)
    def find_indices(lst, condition):
        return [i for i, elem in enumerate(lst) if condition(elem)]
    
    def find_chantype(chan_list, index_list):
        for i in range(len(chan_list)):
            index_list = index_list + find_indices(ch_names_clean, lambda e: True if chan_list[i] in e else False)
        return index_list

    ieeg_idx = find_chantype(ieeg_chan, [])
    dc_idx = find_chantype(dc_chan, [])
    ekg_idx = find_chantype(ekg_chan, [])
    emg_idx = find_chantype(emg_chan, [])
    not_scalp_idx = ieeg_idx + dc_idx + ekg_idx + emg_idx
    scalp_idx = list(set(list(range(len(ch_names_clean)))) - set(not_scalp_idx))
    
    chantype = ch_names_clean.copy()

    for i in range(len(ieeg_idx)):
        chantype[ieeg_idx[i]] = 'intracranial EEG'
    for i in range(len(dc_idx)):
        chantype[dc_idx[i]] = 'TTL'
    for i in range(len(ekg_idx)):
        chantype[ekg_idx[i]] = 'EKG'
    for i in range(len(emg_idx)):
        chantype[emg_idx[i]] = 'EMG'
    for i in range(len(scalp_idx)):
        chantype[scalp_idx[i]] = 'scalp EEG'

    #Code channel labels as bytes dtype (format accepted in hdf5 class currently used)
    def get_channel_labels(all_labels):
        old_list = []
        new_list = []
        for i in range(len(all_labels)):
            old_list.append(re.sub("[A-Za-z]+", lambda ele: "" + ele[0] + " ", all_labels[i]))
        old_list = [x.split(" ") for x in old_list]
        for i in range(len(old_list)):
            new_list.append(tuple(old_list[i]))
        return new_list

    channel_labels = get_channel_labels(ch_names_clean)

    #Extract timeseries
    data_array, time = raw[:,:]
    
    #Build dictionary object
    edf_dic = {
        'edf_fn': edf_fn,
        'edf_path': os.path.join(edf_dir, edf_fn),
        'edf_start': edf_start.replace(tzinfo=None), #extract timestamp only to then convert to nanoseconds when writing file
        'edf_end': edf_end.replace(tzinfo=None), #extract timestamp only to then convert to nanoseconds when writing file
        'edf_timezone': 'US/Pacific', #assign correct timezone TO-DO:fix timezone in datetime object and then call object.tzinfo instead of str 
        'edf_duration': edf_len,
        'edf_nsample': len(raw),
        'edf_sfreq': raw.info['sfreq'],
        'edf_lowpass': raw.info['lowpass'],
        'edf_highpass': raw.info['highpass'],
        'edf_nchan': raw.info['nchan'],
        'edf_raw_chanlabs': raw.info['ch_names'],
        'edf_chantype': chantype,
        'edf_axis': list(['chan','sample']),
        'edf_data': data_array,
        'edf_time_axis': time,
        'edf_channellabel_axis': channel_labels
    }
        
    return edf_dic

print('EDF reader is ready to use')

"""End of code"""
