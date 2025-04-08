"""hdf5concat.py

Creates new 6-min duration h5 file preceding each biomarker survey, accounts for duplicated timestamps

"""

# Standard Libraries #
import pandas as pd
import numpy as np
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

# Third-Party Packages #
from nkhdf5 import hdf5nk

# Local Packages #
HDF5NK = hdf5nk.HDF5NK_0_1_0 

# Main #
if __name__ == "__main__":
    # Input Parameters  
    subject_id  = "PR07"
    catalogs_dir = f"/data_store0/presidio/nihon_kohden/{subject_id}/catalogs"
    out_dir = f"/data_store0/presidio/nihon_kohden/{subject_id}/nkhdf5"
    hdf5_catalog = pd.read_csv(f"{catalogs_dir}/sub-{subject_id}_hdf5-catalog.csv")
    redcap = pd.read_csv(f"{catalogs_dir}/sub-{subject_id}_surveys-catalog.csv")
    biomarker_idxs = [record_id for record_id, condition in zip(redcap.record_id, redcap.condition) if "Biomarker" in condition]
    biomarker_surveys = redcap.loc[redcap.record_id.isin(biomarker_idxs)].reset_index(drop=True)

    elecscoord_on = False

    with open(f"/userdata/dastudillo/subjects.json", "r") as f: #stored in user's directory, not part of repo files
        subjects = json.load(f)
    hdf5_dir = subjects[subject_id]["BIDS_raw_stage1"] #where HDF5 files are stored
    ref_date = datetime.strptime(subjects[subject_id]["consent_date"], "%Y-%m-%d") #use to retrieve original timestamps
    stage1_day1 = datetime.strptime(subjects[subject_id]["stage1_day1"], "%Y-%m-%d").date() #for file naming, this date sets "day01"
    
    def get_original_dt(ref_dt, norm_t):
        #ref_dt is naive datetime object representing local reference date
        #norm_t is normalized/deidentified timestamp coming from hdf5 file
        refdt_utc_t = ref_dt.replace(tzinfo=pytz.timezone('UTC')).timestamp() #UTC timestamp of reference date
        orig_t = norm_t/1e9 + refdt_utc_t #original timestamp in local time
        orig_dt = datetime.fromtimestamp(orig_t) #original datetime object (with correct local date and time)
        return orig_dt

    # Main code
    target_duration = 6 #minutes
    hdf5_start_lst = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in hdf5_catalog.hdf5_start]
    hdf5_end_lst = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in hdf5_catalog.hdf5_end]
    filenames = list(hdf5_catalog.hdf5_name)
    target_end_lst = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in biomarker_surveys.start_local_timestamp][32:]

    for target_end in target_end_lst:
        target_start = target_end - timedelta(minutes=target_duration)

        ###prepare filename###
        start_date = target_end.date()
        diff = (start_date-stage1_day1).days
        day_label = "day" + str(diff+1).zfill(2)
        time_label = target_end.strftime("%H%M%S")
        file_name = f"sub-{subject_id}_ses-stage1_task-survey_acq-{day_label}_run-{time_label}_ieeg.h5" #get file name ready
        file_out  = pathlib.Path(out_dir, file_name)
        ######################

        if file_out.is_file()==True:
            print(f"{file_name} was already created and saved in {out_dir}")
            print("")
            print("Checking next files...")
            print("")

        if file_out.is_file()==False:
            #find files to merge 
            for start, end, fn in zip(hdf5_start_lst, hdf5_end_lst, filenames):
                if (target_start>=start) & (target_start<=end):
                    start_idx = filenames.index(fn)
                if (target_end>=start) & (target_end<=end):
                    end_idx = filenames.index(fn)
            files_to_merge = filenames[start_idx:end_idx+1]
            files_to_merge.sort()

            #merge timeseries and retrieve ieeg metadata
            timestamps_merged = []
            data_merged = []
            for fn in files_to_merge:
                f = os.path.join(hdf5_dir, fn)
                hdf5_f = h5py.File(f, "r")
                data = np.array(hdf5_f["intracranialEEG"])
                data_merged = data_merged + list(data)
                timetamps = np.array(hdf5_f["intracranialEEG_time_axis"])
                timestamps_merged = timestamps_merged + list(timetamps)
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
                original_datetimes.append(get_original_dt(ref_date, timestamp))

            #find closest datetime in time array to survey start, this will be the actual end of recording
            actual_end = min(original_datetimes, key=lambda x: abs(x - target_end))
            actual_start = actual_end - timedelta(minutes=target_duration)
            datetime_array = original_datetimes[original_datetimes.index(actual_start):original_datetimes.index(actual_end)]
            
            data_array = np.array(data_merged[original_datetimes.index(actual_start):original_datetimes.index(actual_end)])
            timestamps_array = np.array([int(datetime.timestamp(x)*1e9) for x in datetime_array])
            ieeg_start = int(datetime.timestamp(actual_start)*1e9)
            ieeg_end = int(datetime.timestamp(actual_end)*1e9)

            print("Creating: ", file_name)
            print("")
            # Create the file #
            f_obj = HDF5NK(file=file_out, mode="a", create=True, construct=True)
            f_obj.attributes["subject_id"] = subject_id
            f_obj.attributes["start"] = ieeg_start
            f_obj.attributes["end"] = ieeg_end
    
            file_data_ieeg = f_obj["data_ieeg"]
            file_data_ieeg.append(data_array, component_kwargs={"timeseries": {"data": timestamps_array}})
            file_data_ieeg.axes[1]["channellabel_axis"].append(ieeg_channellabels)
            if elecscoord_on == True:
                file_data_ieeg.axes[1]["channelcoord_axis"].append(ieeg_channelcoord)

            file_data_ieeg.attributes["filter_lowpass"]  = ieeg_lowpass
            file_data_ieeg.attributes["filter_highpass"] = ieeg_highpass
            file_data_ieeg.attributes["channel_count"]   = ieeg_channelcount
            file_data_ieeg.axes[0]["time_axis"].attributes["sample_rate"] = ieeg_samplerate 
            file_data_ieeg.axes[0]["time_axis"].attributes["time_zone"] = ieeg_timezone 

            print("File after appending:")
            print("ieeg data size: ", f_obj["data_ieeg"].shape)
            print("ieeg time axis size: ", f_obj["data_ieeg"].axes[0]["time_axis"].shape)
            print("")
            print(f"{file_name} was created and saved in {out_dir}")
            print("Checking next files...")
            print("")
            #print(f"File Exists: {file_out.is_file()}")
            #print(f"File is Openable: {HDF5NK.is_openable(out_path)}")
            #print("")

            f_obj.close()

    print("All files in queue created!")
    print("")
"""End of code

"""
