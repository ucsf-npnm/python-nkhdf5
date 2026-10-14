import os
from redcap import Project

"""
Paths to files/datasets in server.

"""
BASE_DIR = "/data_store2/presidio/nihon_kohden" # path in server where Stage 1 data is located
BIDS_DIR = "/data_store2/presidio/BIDS/raw" # path in server where HDF5-BIDS files (with de-identified timestamps) are located.

# from base directory
EDF_DIR = os.path.join(BASE_DIR, "subject_id/subject_id") # path in server where EDF files are located
NK_ANNOTS_DIR = os.path.join(BASE_DIR, "subject_id/NK_annotations") # path in server where NK annotations are located
CAT_DIR = os.path.join(BASE_DIR, "subject_id/catalogs") # path in server where catalogs with files metadata or survey records are stored.
NKHDF5_DIR = os.path.join(BASE_DIR, "subject_id/nkhdf5") # path in server where 6-min HDF5 raw recordings preceding surveys are stored (mainly for biomarker analysis)

# from BIDS directory
HDF5_DIR = os.path.join(BIDS_DIR, "sub-subject_id/ses-stage1/ieeg") # subject's directory with HDF5-BIDS files

"""
Custom fuctions to access datasets.

"""
def import_redcap_records(tk):
    ## Connect to REDCap server and access project using token 
    api_url = "https://redcap.ucsf.edu/api/"
    project = Project(api_url, tk)

    ## Get project info (title, creation date, etc)
    project_info = project.export_project_info()

    ## Get records from project with raw data
    project_records = project.export_records(format_type="df", export_survey_fields=True).dropna(subset="start_local_timestamp").reset_index()
    print(f"{len(project_records)} survey(s) found in {project_info['project_title']}")

    return project_records
