import pytz
from datetime import datetime

def normalize_dates(ref, dt):
    """
    Deidentifies a datetime timestamp using the subject's consent date and converts it into a unix nanosecond timestamp.

    Args:
        ref_dt (str): subject's consent date matching format 'YYYY-MM-DD' imported from .env file.
        dt (datetime): datetime timestamp (usually from an EDF file).

    Outputs:
        unix_t (int): deidentified unix nanosecond timestamp.
    """
    ref_dt = datetime.strptime(ref, "%Y-%m-%d")
    ref_dt_utc = ref_dt.replace(tzinfo=pytz.timezone("UTC"))
    unix_t = int(1e9 * (dt.timestamp()-ref_dt_utc.timestamp()))

    return unix_t

def get_original_dt(ref, unix_t):
    """
    Converts a deidentified unix nanosecond timestamp back to its original timestamp (as datetime object).

    Args:
        ref (str): subject's consent date matching format 'YYYY-MM-DD' imported from .env file.
        unix_t (int): deidentified unix nanosecond timestamp from a HDF5-BIDS file.

    Outputs:
        orig_dt (datetime): original timestamp
    """
    
    # Convert reference date to naive datetime object
    ref_dt = datetime.strptime(ref, "%Y-%m-%d")
    # Convert naive datetime object to UTC timestamp
    refdt_utc_t = ref_dt.replace(tzinfo=pytz.timezone("UTC")).timestamp()
    # Get original timestamp (still in unix format)
    orig_t = unix_t/1e9 + refdt_utc_t
    # Convert original timestamp to readable datetime object
    orig_dt = datetime.fromtimestamp(orig_t) #original datetime object (with correct local date and time)
    
    return orig_dt

def datetime_to_unixnano(dt):
    """
    Converts a datetime timestamp into a unix nanosecond timestamp.

    Args:
        dt (datetime): datetime timestamp
    
    Outputs:
        unix nanosecond timestamp
    """
    return int(datetime.timestamp(dt)*1e9)