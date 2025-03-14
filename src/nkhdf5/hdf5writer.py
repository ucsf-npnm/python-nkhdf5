"""hdf5writer.py

By providing a list of file paths, converts an EDF file to HDF5 file

"""

# Standard Libraries #
import pathlib
import numpy as np
import pandas as pd
import datetime
import time
import h5py
import scipy.io
import ast

# Third-Party Packages #
from nkhdf5 import hdf5nk
HDF5NK = hdf5nk.HDF5NK_0_1_0

# Local Packages #
from edfreader import get_edf_list, edf_reader
from concatenator_tools import FilesForBiomarker

# Main #
if __name__ == "__main__":
    ## Input Parameters 
    patient_id  = "PR06"
    stage1_path = "/data_store0/presidio/nihon_kohden"
    edf_path    = pathlib.Path(stage1_path,patient_id,patient_id)
    outpath     = pathlib.Path(stage1_path, patient_id, "nkhdf5/edf_to_hdf5")

    #Define correct path for imaging data according to patient_id (only relevant to Presidio study)
    imaging_path = f"/data_store2/imaging/subjects/{patient_id}/elecs"
    if (patient_id == "PR01") or (patient_id == "PR06"):
        mat_file = "elecs_all.mat"
    if patient_id == "PR03":
        mat_file = "PR03_elecs_all.mat"
    if (patient_id == "PR04") or (patient_id == "PR05"):
        mat_file = "stereo_elecs_all.mat"
    elecscoor_filepath = pathlib.Path(imaging_path, mat_file) 

    edf_catalog = pd.read_csv(f"{stage1_path}/{patient_id}/nkhdf5/{patient_id}_edf_catalog.csv")
    biomarker_surveys = pd.read_csv(f"{stage1_path}/{patient_id}/clinical_scores/BiomarkerSurveys.csv")
    biomarker_survey_times = pd.to_datetime(biomarker_surveys["SurveyStart"])

    ## Extract list of all edfs
    edf_all = get_edf_list(edf_path)
    #edf_all = list(edf_catalog["edf_name"]) #another way to extract list of edf files from available edf catalog (Presidio only)

    ## Extract list of edf associated to biomarker surveys
    #edf_for_bm = FilesForBiomarker(10, "EDF", biomarker_survey_times, edf_catalog) #convert edf files during biomarker periods 
    
    ## Start of actual code, loop through edf files
    for i in range(len(edf_all)):
        edf_contents = edf_reader(edf_path, edf_all[i]) #dictionary with metadata+timeseries
        date_string  = edf_contents["edf_start"].strftime("%Y%m%d") #date use in filename
        time_string  = edf_contents["edf_start"].strftime("%H%M%S") #time use in filename
        start_rec    = time.mktime(edf_contents["edf_start"].timetuple())*1e9 #unix epoch time
        file_name    = f"sub-{patient_id}_ses-stage1_task-continuous_acq-{date_string}_run-{time_string}_ieeg.h5" #get file name ready
        out_path     = pathlib.Path(outpath, file_name) #full path for new file

        if out_path.is_file()==True:
            print("")
            print(f"{edf_all[i]} has already been converted to HDF5 as {file_name}")
            print("")
            print("Checking next file...")
            print("")

        if out_path.is_file()==False:
            ### Extract raw data by channel type
            ieeg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantype"]) if v == "intracranial EEG"]).T
            scalpeeg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantype"]) if v == "scalp EEG"]).T
            ekg_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantype"]) if v == "EKG"]).T
            ttl_array = np.array([k for k,v in zip(edf_contents["edf_data"], edf_contents["edf_chantype"]) if v == "TTL"]).T
            ### Extract raw time and convert to absolute timestamps (nanoseconds) 
            time_array = np.array((edf_contents["edf_time_axis"]*1e9).astype(int))

            def get_abs_timestamps(nanostamps_array):
                abs_timestamps = []
                for i in range(len(nanostamps_array)):
                    abs_timestamps.append(start_rec+nanostamps_array[i])
                return abs_timestamps

            new_time_array = np.array(get_abs_timestamps(time_array))

            ### Extract channel labels by channel type in format accepted by class 
            chanlabs_ieeg_array = np.array([k for k,v in zip(edf_contents["edf_channellabel_axis"], edf_contents["edf_chantype"]) if v == "intracranial EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_scalpeeg_array = np.array([k for k,v in zip(edf_contents["edf_channellabel_axis"], edf_contents["edf_chantype"]) if v == "scalp EEG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ekg_array = np.array([k for k,v in zip(edf_contents["edf_channellabel_axis"], edf_contents["edf_chantype"]) if v == "EKG"], dtype = h5py.special_dtype(vlen=str))
            chanlabs_ttl_array = np.array([k for k,v in zip(edf_contents["edf_channellabel_axis"], edf_contents["edf_chantype"]) if v == "TTL"], dtype = h5py.special_dtype(vlen=str))

            ### Extract electrodes coordinates (only for depth electrodes, data_ieeg)
            elecs_mat_file = scipy.io.loadmat(elecscoor_filepath)
            elecs_coor = elecs_mat_file["elecmatrix"]

            ### Create the file
            f_obj = HDF5NK(file=out_path, mode="a", create=True, construct=True)
            f_obj.attributes["subject_id"] = patient_id
            f_obj.attributes["start"] = int(time.mktime(edf_contents["edf_start"].timetuple())*1e9)
            f_obj.attributes["end"] = int(time.mktime(edf_contents["edf_end"].timetuple())*1e9)
    
            file_data_ieeg = f_obj["data_ieeg"]
            file_data_ieeg.append(ieeg_array, component_kwargs={"timeseries": {"data": new_time_array}})
            file_data_ieeg.axes[1]["channellabel_axis"].append(chanlabs_ieeg_array)
            file_data_ieeg.axes[1]["channelcoord_axis"].append(elecs_coor)

            file_data_ieeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
            file_data_ieeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
            file_data_ieeg.attributes["channel_count"]   = ieeg_array.shape[1]
            file_data_ieeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
            file_data_ieeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(scalpeeg_array)!=0:
                file_data_scalpeeg = f_obj["data_scalpeeg"]
                file_data_scalpeeg.append(scalpeeg_array, component_kwargs={"timeseries": {"data": new_time_array}})
                file_data_scalpeeg.axes[1]["channellabel_axis"].append(chanlabs_scalpeeg_array)

                file_data_scalpeeg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_scalpeeg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_scalpeeg.attributes["channel_count"]   = scalpeeg_array.shape[1]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_scalpeeg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ekg_array)!=0:
                file_data_ekg = f_obj["data_ekg"]
                file_data_ekg.append(ekg_array, component_kwargs={"timeseries": {"data": new_time_array}})
                file_data_ekg.axes[1]["channellabel_axis"].append(chanlabs_ekg_array)

                file_data_ekg.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_ekg.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_ekg.attributes["channel_count"]   = ekg_array.shape[1]
                file_data_ekg.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_ekg.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            if len(ttl_array)!=0:
                file_data_ttl = f_obj["data_ttl"]
                file_data_ttl.append(ttl_array, component_kwargs={"timeseries": {"data": new_time_array}})
                file_data_ttl.axes[1]["channellabel_axis"].append(chanlabs_ttl_array)

                file_data_ttl.attributes["filter_lowpass"]  = edf_contents["edf_lowpass"]
                file_data_ttl.attributes["filter_highpass"] = edf_contents["edf_highpass"]
                file_data_ttl.attributes["channel_count"]   = ttl_array.shape[1]
                file_data_ttl.axes[0]["time_axis"].attrs["sample_rate"] = edf_contents["edf_sfreq"]
                file_data_ttl.axes[0]["time_axis"].attrs["time_zone"] = edf_contents["edf_timezone"]

            print("")
            print(f"{edf_all[i]} saved as: ", file_name)
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
