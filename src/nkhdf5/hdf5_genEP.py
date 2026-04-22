"""hdf5_genEP.py

Creates new h5 files for each EP train.

"""

# Standard Libraries #
import pandas as pd
import numpy as np
import scipy.interpolate
import datetime
from datetime import datetime, timedelta
import time
import json
import os
import pathlib
import pytz
from pytz import timezone
import h5py
import ast
import redcap
from redcap import Project
from multiprocessing import Pool

# Third-Party Packages #
from nkhdf5 import hdf5nk

# Local Packages #
import EP_utils

HDF5NK = hdf5nk.HDF5NK_0_1_0 

# Function #
def process_ep_epoch(data_dict):
    subject_id = data_dict["subject_id"]
    out_dir = data_dict["out_dir"]
    hdf5_catalog = data_dict["hdf5_catalog"]
    nkstim_entry = data_dict["nkstim_entry"]
    pulse_min_height = data_dict["pulse_min_height"]
    pulse_ipi_tol = data_dict["pulse_ipi_tol"]
    clip_pre_pulse_dur = data_dict["clip_pre_pulse_dur"]
    clip_post_pulse_dur = data_dict["clip_post_pulse_dur"]
    reference_date = data_dict["reference_date"]

    target_start = nkstim_entry["stim_start_adjusted"]
    target_end = nkstim_entry["stim_end_adjusted"]

    print(f"Processing: {subject_id} - {target_start} - {target_end}")

    #files to merge
    files_to_merge = _h5_merge(
        hdf5_catalog,
        target_start,
        target_end
    )

    #merge timeseries and retrieve ieeg metadata
    timestamps_merged = []
    data_merged = []
    for fn in files_to_merge:
        print(fn)
        hdf5_f = h5py.File(fn, "r")
        data = np.array(hdf5_f["intracranialEEG"])
        data_merged = data_merged + list(data)
        timestamps = np.array(hdf5_f["intracranialEEG_time_axis"])
        timestamps_merged = timestamps_merged + list(timestamps)
        ieeg_channellabels =  np.array(hdf5_f["intracranialEEG_channellabel_axis"])
        ieeg_channelcoord = np.array(hdf5_f["intracranialEEG_channelcoord_axis"])
        ieeg_channelcount = hdf5_f["intracranialEEG"].attrs["channel_count"]
        ieeg_lowpass = hdf5_f["intracranialEEG"].attrs["filter_lowpass"]
        ieeg_highpass = hdf5_f["intracranialEEG"].attrs["filter_highpass"]
        ieeg_samplerate = hdf5_f["intracranialEEG_time_axis"].attrs["sample_rate"]
        ieeg_timezone = hdf5_f["intracranialEEG_time_axis"].attrs["time_zone"]
        hdf5_f.close()
        del hdf5_f

    #retrieve original datetimes
    original_datetimes = []
    for timestamp in timestamps_merged:
        original_datetimes.append(get_original_dt(reference_date, timestamp))

    #find closest datetime in time array to survey start, this will be the actual end of recording
    actual_end = min(original_datetimes, key=lambda x: abs(x - target_end))
    actual_start = min(original_datetimes, key=lambda x: abs(x - target_start))
    datetime_array = original_datetimes[original_datetimes.index(actual_start):original_datetimes.index(actual_end)]

    data_array = np.array(data_merged[original_datetimes.index(actual_start):original_datetimes.index(actual_end)])
    timestamps_array = np.array([int(datetime.timestamp(x)*1e9) for x in datetime_array])
    ieeg_start = int(datetime.timestamp(actual_start)*1e9)
    ieeg_end = int(datetime.timestamp(actual_end)*1e9)

    #find pulses
    pulse_inds = EP_utils.find_stim_pulse_times(
        data_array,
        ieeg_samplerate,
        float(nkstim_entry['frequency']),
        pulse_min_height,
        pulse_ipi_tol,
    )

    for pind in pulse_inds:
        pulse_ts = timestamps_array[pind]
        clip_start_ind = int(pind-ieeg_samplerate*clip_pre_pulse_dur)
        clip_stop_ind = int(pind+ieeg_samplerate*clip_post_pulse_dur)
        clip_data_array = data_array[clip_start_ind:clip_stop_ind]
        clip_abs_timestamps = timestamps_array[clip_start_ind:clip_stop_ind]
        clip_rel_timestamps = clip_abs_timestamps - pulse_ts
        clip_samplerate = ieeg_samplerate
        clip_channellabels = ieeg_channellabels

        out_clip_path = pathlib.Path(out_dir, str(pulse_ts) + ".npz")
        np.savez(out_clip_path,
            data_array=clip_data_array,
            timestamps_abs=clip_abs_timestamps,
            timestamps_rel=clip_rel_timestamps,
            sample_freq=clip_samplerate,
            channel_labels=clip_channellabels,
            allow_pickle=True
        )
        print(f"Saved: {out_clip_path}")


def _clean_nkstim_catalog(nkstim_catalog, ep_frequency_max, stim_offset):
    ##Format nkstim catalog
    nkstim_catalog = nkstim_catalog[[
        "stim_start",
        "stim_end",
        "stim_condition",
        "lead",
        "contacts",
        "pos_contact",
        "neg_contact",
        "amplitude",
        "train_duration",
        #"pulse_duration",
        "frequency"
    ]]
    nkstim_catalog[["stim_start", "stim_end"]] = nkstim_catalog[
        ["stim_start", "stim_end"]].map(
            lambda z: pd.to_datetime(z, errors="ignore"))
    nkstim_catalog = nkstim_catalog.loc[(~nkstim_catalog[
        ["stim_start", "stim_end"]].isna().any(axis=1))]
    nkstim_catalog['stim_start_adjusted'] = (
        nkstim_catalog['stim_start'] - stim_offset)
    nkstim_catalog['stim_end_adjusted'] = (
        nkstim_catalog['stim_end'] + stim_offset)
    nkstim_catalog = nkstim_catalog.loc[
        nkstim_catalog['frequency'] <= ep_frequency_max]
    nkstim_catalog = nkstim_catalog.reset_index(drop=True)
    return nkstim_catalog


def _reference_dates(subject_id):
    ##Get path to HDF5 files converted from EDF files and relevant dates for file name and timestamps retrieval
    with open(f"/home/akhambhati/.config/presidio_subjects.json", "r") as f: #stored in user's directory, not part of repo files
        subjects = json.load(f)
    ref_date = datetime.strptime(subjects[subject_id]["consent_date"], "%Y-%m-%d") #use to retrieve original timestamps
    return ref_date


def _h5_merge(hdf5_catalog, target_start, target_end):
    # extract hdf5 catalog data
    hdf5_start_lst = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in hdf5_catalog.hdf5_start if type(x) is str]
    hdf5_end_lst = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in hdf5_catalog.hdf5_end if type(x) is str]
    filenames = list(hdf5_catalog.hdf5_name)
    filepaths = list(hdf5_catalog.hdf5_path)

    #find files to merge 
    for start, end, fp in zip(hdf5_start_lst, hdf5_end_lst, filepaths):
        if (target_start>=start) & (target_start<=end):
            start_idx = filepaths.index(fp)
        if (target_end>=start) & (target_end<=end):
            end_idx = filepaths.index(fp)
    files_to_merge = filepaths[start_idx:end_idx+1]
    files_to_merge.sort()
    return files_to_merge


def get_original_dt(ref_dt, norm_t):
    #ref_dt is naive datetime object representing local reference date
    #norm_t is normalized/deidentified timestamp coming from hdf5 file
    refdt_utc_t = ref_dt.replace(tzinfo=pytz.timezone('UTC')).timestamp() #UTC timestamp of reference date
    orig_t = norm_t/1e9 + refdt_utc_t #original timestamp in local time
    orig_dt = datetime.fromtimestamp(orig_t) #original datetime object (with correct local date and time)
    return orig_dt


def get_pulse_times(path):
    found_pulse_times = np.array([
        int(fn.stem)
        for fn in pathlib.Path(path).glob("*.npz")
    ])
    return found_pulse_times


def generate_ep_catalog(nkstim_catalog, pulse_times):
    pulse_times = pd.DataFrame({"pulse_time": [datetime.fromtimestamp(x/1e9) for x in pulse_times]}).sort_values("pulse_time")

    cj = nkstim_catalog.merge(pulse_times, how="cross")
    expanded = cj[(cj["pulse_time"] >= cj["stim_start_adjusted"]) &
                  (cj["pulse_time"] < cj["stim_end_adjusted"])]

    return expanded.sort_values("pulse_time").reset_index(drop=True)


def generate_ep_hdf5(
    path,
    ep_catalog,
    pre_pulse_dur,
    post_pulse_dur,
    max_sample_freq,
    max_channel,
    out_path=None
):
    if out_path is None:
        out_path = path

    T_MAX = np.arange(
        -1*pre_pulse_dur*1e9,
        post_pulse_dur*1e9,
        1/max_sample_freq*1e9
    )
    C_MAX = np.array([[],[]], dtype=object).T

    LEN_PRE = (T_MAX < 0).sum()
    LEN_POST = (T_MAX >= 0).sum()

    h5 = h5py.File(pathlib.Path(out_path, "ep_db.h5"), "w")
    ds = h5.create_dataset(
        "pulse_data",
        shape=(0,len(T_MAX),max_channel),
        maxshape=(None,len(T_MAX),max_channel),
        dtype="f",
        track_order=True
    )

    catalog_inds = []
    for ev_ii, ep_ev in ep_catalog.iloc[:].iterrows():
        print(f"Writing event: {ev_ii+1} of {ep_catalog.shape[0]}")

        pulse_time = int(
            datetime.timestamp(pd.to_datetime(ep_ev['pulse_time']))*1e9
        )

        ep_file = np.load(
             pathlib.Path(path, f"{pulse_time}.npz"), allow_pickle=True
        )
        shape = ep_file["data_array"].shape
        padded_array = np.nan*np.zeros((len(T_MAX), max_channel))
        ts_rel = T_MAX

        if (shape[0] > 0) & (shape[1] > 0):
            data_array = ep_file["data_array"][...]
            ts_abs = ep_file["timestamps_abs"][...]
            ts_rel = ts_abs - pulse_time
            fs = ep_file["sample_freq"]
            ch_lbl = ep_file["channel_labels"]

            ##### Time alignment
            upsample_factor = max_sample_freq / fs                                   
            if upsample_factor >= 1.5:      

                ## Piecewise interpolation (pre-stim)
                ts_rel_pre = ts_rel[ts_rel <= 0]
                ts_rel_pre_upsample = np.linspace(                                       
                    min(ts_rel_pre),                                                     
                    max(ts_rel_pre),                                                     
                    int(len(ts_rel_pre)*upsample_factor),                                
                )    
                data_array_pre_upsample = []
                for data_vec in data_array.T:
                    f_upsample = scipy.interpolate.interp1d(ts_rel_pre, data_vec[ts_rel <= 0], kind='linear')
                    data_array_pre_upsample.append(f_upsample(ts_rel_pre_upsample))
                data_array_pre_upsample = np.array(data_array_pre_upsample).T
                
                ## Piecewise interpolation (post-stim)
                ts_rel_post = ts_rel[ts_rel >= 0]
                ts_rel_post_upsample = np.linspace(                                       
                    min(ts_rel_post),                                                     
                    max(ts_rel_post),                                                     
                    int(len(ts_rel_post)*upsample_factor),                                
                )    
                data_array_post_upsample = []
                for data_vec in data_array.T:
                    f_upsample = scipy.interpolate.interp1d(ts_rel_post, data_vec[ts_rel >= 0], kind='linear')
                    data_array_post_upsample.append(f_upsample(ts_rel_post_upsample))
                data_array_post_upsample = np.array(data_array_post_upsample).T
                
                ## Concatenate, drop duplicate fixed point from pre-stim
                ts_rel_upsample = np.concatenate((ts_rel_pre_upsample[:-1], ts_rel_post_upsample))
                data_array_upsample = np.concatenate((data_array_pre_upsample[:-1], data_array_post_upsample), axis=0)
            else:                                                                    
                ts_rel_upsample = ts_rel
                data_array_upsample = data_array                                     
            assert ts_rel_upsample.shape[0] == data_array_upsample.shape[0]          
            
            n_pre = (ts_rel_upsample < 0).sum()                                      
            n_post = (ts_rel_upsample >= 0).sum()                                     
            if n_pre > LEN_PRE:
                data_array_upsample = data_array_upsample[n_pre-LEN_PRE:]
                ts_rel_upsample = ts_rel_upsample[n_pre-LEN_PRE:]
            if n_post > LEN_POST:
                data_array_upsample = data_array_upsample[:LEN_POST-n_post]
                ts_rel_upsample = ts_rel_upsample[:LEN_POST-n_post]
            n_pre = (ts_rel_upsample < 0).sum()                                      
            n_post = (ts_rel_upsample >= 0).sum()     
            
            for lbl_i, lbl in enumerate(ch_lbl):
                C_idx = np.flatnonzero(
                    (C_MAX[:,0] == lbl[0]) & 
                    (C_MAX[:,1] == lbl[1])
                )
                if len(C_idx) == 0:
                    C_MAX = np.append(C_MAX, [lbl], axis=0)
                    C_idx = C_MAX.shape[0]-1
                else:
                    assert len(C_idx) == 1
                    C_idx = C_idx[0]
                
                padded_array[                                                            
                    LEN_PRE-n_pre:LEN_PRE+n_post,
                    C_idx
                ] = data_array_upsample[:, lbl_i]   
            #####
        ep_file.close()

        #### Append to the open dataset
        ds.resize((ds.shape[0]+1, ds.shape[1], ds.shape[2]))
        ds[-1, :, :] = padded_array
        catalog_inds.append(ev_ii)
    ds.attrs.create("ax0_catalog", data=catalog_inds, dtype=int)
    ds.attrs.create("ax1_timestamps", data=T_MAX, dtype=int)
    ds.attrs.create("ax2_channels", data=C_MAX)
    h5.close()


# Main #
if __name__ == "__main__":
    # User-specified inputs  
    subject_id  = "PR06"
    out_dir = f"/userdata/akhambhati/patient_data/{subject_id}/evoked_potentials/nkhdf5" 
    epdb_dir = f"/scratch/akhambhati/patient_data/{subject_id}/evoked_potentials/nkhdf5"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(epdb_dir, exist_ok=True)

    found_pulse_times = get_pulse_times(out_dir)

    ep_frequency_max = 1
    stim_offset = timedelta(seconds=5)
    pulse_min_height = 3000
    pulse_ipi_tol = 0.95
    clip_pre_pulse_dur=0.5
    clip_post_pulse_dur=0.5
    max_sample_freq = 10000
    max_channel = 256

    hdf5_catalog = pd.read_csv(f"/data_store2/presidio/nihon_kohden/{subject_id}/catalogs/sub-{subject_id}_hdf5-catalog.csv")
    nkstim_catalog = _clean_nkstim_catalog(pd.read_csv(f"/userdata/akhambhati/patient_data/{subject_id}/{subject_id}_Stage_1-CleanNKStim.csv"), ep_frequency_max, stim_offset)
    reference_date = _reference_dates(subject_id)
    ################################################

    #### Process each NK Stim Entry (takes time, can be code blocked)
    """
    N_PROC=24
    proc_dict = [{
        "subject_id": subject_id,
        "out_dir": out_dir,
        "nkstim_entry": nkstim_entry,
        "hdf5_catalog": hdf5_catalog,
        "pulse_min_height": pulse_min_height,
        "pulse_ipi_tol": pulse_ipi_tol,
        "clip_pre_pulse_dur": clip_pre_pulse_dur,
        "clip_post_pulse_dur": clip_post_pulse_dur,
        "reference_date": reference_date
        } for ii, nkstim_entry in nkstim_catalog.iterrows() 
        if not ((found_pulse_times >= int(datetime.timestamp(nkstim_entry["stim_start_adjusted"])*1e9)) & 
                (found_pulse_times <= int(datetime.timestamp(nkstim_entry["stim_end_adjusted"])*1e9))).any()
    ]

    print(len(proc_dict))

    pool = Pool(N_PROC)
    pool.map(process_ep_epoch, proc_dict)

    print("All files in queue created!")
    print("")
    """

    #### Compile Extracted Pulses into an EP Catalog file
    ep_catalog = generate_ep_catalog(nkstim_catalog, found_pulse_times)
    ep_catalog.to_csv(f"{out_dir}/catalog.csv", index=False)

    #### Compile Extracted Pulses into an HDF5 Dataset
    generate_ep_hdf5(
        out_dir,
        ep_catalog,
        clip_pre_pulse_dur,
        clip_post_pulse_dur,
        max_sample_freq,
        max_channel,
        out_path=epdb_dir
    )



"""End of code

"""
