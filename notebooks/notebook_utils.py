import numpy as np
import warnings
from datetime import datetime

def check_input_validity(save_nc, output_fns, single_file_nc, n_comp, model_fn_ids, n_mod_fns,
                         plot, save_plot, plot_fn, loc = None, regrid=False, extent=None, grid_size=None):
    """
    Checks the validity of inputs to DynamicThickness/GIS_Compare_Dynamic_Thickness.ipynb and 
    Gravimetry/GIS_or_AIS_model_onto_GSFCmascons.ipynb

    If no issues, returns (True, ""). If there is an error, returns (False, error), where error is a string
    describing the error
    """
    
    if regrid:
        if (extent is None) or (grid_size is None):
            return True, "If the regrid input variable is true, the extent and grid_size variables must be supplied"
            
        if np.size(extent) != 4:
            return True, "extent must have 4 entries (left, right, top, bottom) in units of meters in polar stereographic"
    
    if save_nc:
        # Check that all output filenames are unique
        if len(np.unique(output_fns)) < len(output_fns):
            return True, "Error: At least two paths in output_fns are identical"
    
        # Check that there are enough filenames
        if single_file_nc:
            if len(output_fns) < 1:
                return True, f"Error: No filename for saving the output netCDF file was provided, see the output_fns input"
        else:
            if (len(output_fns) != n_comp):
                return True, f"Error: {n_comp} comparisons requested but {len(output_fns)} output filenames provided"
    
        # Check that model_fn_ids is either None or has the same length as model_fns
        if model_fn_ids is not None:
            if (len(model_fn_ids) != n_mod_fns):
                return True, f"Error: model_fn_ids is not None but does not have the same length as model_fns"
    
            if len(np.unique(model_fn_ids)) < len(model_fn_ids):
                return True, "Error: At least two ids in model_fn_ids are identical"
    
    if np.sum(plot):
        if len(plot) != n_comp:
            return True, f"Error: Input variable plot has length {len(plot)}, but {n_comp} comparisons requested"
        
        if len(save_plot) != len(plot):
            return True, f"Error: Input variable plot has length {len(plot)}, but input variable save_plot has length {len(plot)}. Should be the same length"
        
        if np.sum(save_plot):
            # Check that all plotting filenames that will be used are unique
            if len(np.unique(plot_fn)) < len(plot_fn):
                return True, "Error: At least two paths in plot_fn are identical"
    if not (save_nc or np.sum(plot)):
        return True, f"Warning: You have chosen not to plot and not to save a netcdf file"

    if (loc is not None) & (loc != "GIS") & (loc != "AIS"):
        return True, f"Error: Input loc is equal to '{loc}'. Allowed values are 'AIS' or 'GIS'"
    return False, ""   # Passed all tests





########## CHECKING TIME ##############

def time_within_bounds(year_dt, time_arr, string_1, string_2, func_time_arr_to_dt = None):
    """
    Returns False if year_dt is outside of bounds of a array of years, and prints a description
    of the problem. Otherwise, returns True

    Parameters:
    year_dt: A python datetime object
    time_arr: A list or array of objects which can be converted to python datetime objects. Expected that
              the first entry of time_arr is chronologically first in time_arr and the last entry of time_arr
              is chronologically last
    string_1, string_2: Strings used in error formatting
    func_time_arr_to_dt: A function which takes one input and returns one output. Converts an entry of
                         time_arr to a python datetime object. If None, it is assumed that time_arr contains
                         only python datetime objects
    """
    if func_time_arr_to_dt is None:
        time_arr_start, time_arr_end = time_arr[0], time_arr[-1]
    else:
        time_arr_start, time_arr_end = func_time_arr_to_dt(time_arr[0]), func_time_arr_to_dt(time_arr[-1])
        
    if time_arr_start > year_dt:
        print(f"Error: {string_1} ({str(year_dt)}) occurs before first date of {string_2} ({str(time_arr_start)})")
        return False
    if time_arr_end < year_dt:
        print(f"Error: {string_1} ({str(year_dt)}) occurs after last date of {string_2} ({str(time_arr_end)})")
        return False
    return True

def str_int_or_float_to_dt(str_int_or_float):
    return datetime.strptime(str(int(str_int_or_float)), "%Y")

def cftime_to_dt(cf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dt64 = np.datetime64(cf)
        return dt64_to_dt(dt64)

def dt64_to_dt(dt64):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        timestamp = ((dt64 - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)

def all_comparisons_in_time_bounds(tups, obs_start, obs_end, obs_to_dt=None):
    """
    Parameters:
    tups: A list of tuples of the form ("YYYY-MM-DD", "YYYY-MM-DD"), where the first entry is 
    presumed to be the start date of a requested comparison and the last entry is presumed to be
    the end date of the same comparison
    obs_start: An object which represents the start of observation data
    obs_end: An object which represents the end of observation data
    obs_to_dt: A function which can take either obs_start or obs_end as an input and output a 
    corresponding datetime object. If this input is not given, it is assumed that obs_start and 
    obs_end are datetime objects

    Returns:
    False if any entry of any tuple occurs before obs_start or after obs_end, and prints appropriate errors.
    Otherwise returns True
    """
    if obs_to_dt is None:
        obs_dt = [obs_start, obs_end]
    else:
        obs_dt = [obs_to_dt(obs_start), obs_to_dt(obs_end)]
        
    for start_date, end_date in tups:
        comp_start_dt, comp_end_dt = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
        if not time_within_bounds(comp_start_dt, obs_dt, "Start of comparison", "observation data"):
            return False
        if not time_within_bounds(comp_end_dt, obs_dt, "End of comparison", "observation data"):
            return False
    return True