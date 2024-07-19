import numpy as np
import netCDF4
import concurrent.futures
import xarray as xr
from datetime import datetime

################# Interpretting Inputs ################# 
def interpret_comparisons_obs(desired_comparisons):
    """
    Outputs comp_info and comp_2_obs_idx, whose purpose is to reduce the information in desired_comparisons 
    to the minimum necessary as it relates to the observation.

    comp_info is a dictionary with the following structure:
        Key: time_id, a 2-tuple of strings of the form ("YYYY-MM-DD", "YYYY-MM-DD")
        Value: List of indices in desired_comparisons which ask for the time range in time_id

    comp_2_obs_idx: For the comparison at index i in desired_comparisons
    """
    comp_info = {}  # Key: 2-Tuple of the form (start_date, end_date), Value: list of indices of desired_comparisons which ask for it
    for i, comp in enumerate(desired_comparisons):
        _, start_date_i, end_date_i = comp
        time_id = (start_date_i, end_date_i)
        comp_info.setdefault(time_id, [])
        comp_info[time_id].append(i)

    comp_2_obs_idx = np.empty((len(desired_comparisons),), dtype = "int")
    for i, tup in enumerate(comp_info):  # Iterating over time spans needed
        for j in comp_info[tup]:    # Iterating over comparisons that need the given time span
            comp_2_obs_idx[j] = i      # i is the index in cmwe_deltas, j is the index of desired_comparisons
    return comp_info, comp_2_obs_idx

def interpret_comparisons_mod(desired_comparisons):
    """
    Outputs comp_info, whose purpose is to reduce the information in desired_comparisons to the minimum necessary as it relates
    to the observation

    comp_info is a dictionary with the following structure:
        Key: An index in model_fns
        Value: A dictionary with the following structure
            Key: time_id, a 2-tuple of strings of the form ("YYYY-MM-DD", "YYYY-MM-DD")
            Value: List of indices in desired_comparisons which ask for the time range in time_id
    """
    comp_info = {}  # Key: 2-Tuple of the form (start_date, end_date), Value: list of indices of desired_comparisons which ask for it
    for i, comp in enumerate(desired_comparisons):
        mod_idx, start_date_i, end_date_i = comp
        time_id = (start_date_i, end_date_i)
        comp_info.setdefault(mod_idx, {})
        (comp_info[mod_idx]).setdefault(time_id, [])
        ((comp_info[mod_idx])[time_id]).append(i)
    return comp_info
    

################# Writing Outputs ################# 
def write_to_netcdf(all_cmwe_res, single_file_nc, output_fns, desired_comparisons, model_ids, model_fns):
    if len(all_cmwe_res) == 1:
        write_one_residual_to_netcdf((all_cmwe_res[0], output_fns[0]))
    elif single_file_nc:
        if model_ids is None:
            model_ids = model_fns
        write_all_residuals_to_one_netcdf(all_cmwe_res, output_fns, desired_comparisons, model_ids)
    else:
        inputs = [(all_cmwe_res[i], output_fns[i]) for i in range(len(all_cmwe_res))]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(write_one_residual_to_netcdf, inputs)
            for result in results:
                pass


def write_one_residual_to_netcdf(tup):
    cmwe_res, fn = tup
    
    # Create a DataArray
    data_array = xr.DataArray(cmwe_res, dims=['mascons'], coords={'mascons': np.arange(len(cmwe_res))})

    # Create a Dataset and add the DataArray under the variable name 'cmwe_res'
    ds = xr.Dataset({'cmwe_res': data_array})

    # Write the dataset to a NetCDF file, including it in the 'residue' group
    ds.to_netcdf(fn, group='residue', mode='w')

def write_all_residuals_to_one_netcdf(all_cmwe_res, output_fns, desired_comparisons, model_ids):
    rootgrp = netCDF4.Dataset(output_fns[0], "w", format="NETCDF4")

    # Read in desired comparisons as three lists, not a list of 3-tuples
    id_idxs, str_ys, end_ys = [], [], []
    for comp in desired_comparisons:
        id_idxs.append(comp[0])
        str_ys.append(comp[1])
        end_ys.append(comp[2])

    # Set up comparison information dimension
    comp_dim = rootgrp.createDimension("comparison", len(desired_comparisons))
    model_id_var = rootgrp.createVariable("model_id", np.dtype('U'), ("comparison",))
    start_year_var = rootgrp.createVariable("start_year", np.dtype('U'), ("comparison",))
    end_year_var = rootgrp.createVariable("end_year", np.dtype('U'), ("comparison",))
        
    for j in range(len(desired_comparisons)):
        model_id_var[j] = model_ids[id_idxs[j]]
        start_year_var[j] = str_ys[j]
        end_year_var[j] = end_ys[j]

    # Fill in dynamic thickness information
    mascon_dim = rootgrp.createDimension("mascon", len(all_cmwe_res[0]))
    all_cmwe_res_var = rootgrp.createVariable("all_cmwe_res", "f8", ("comparison","mascon",))
    all_cmwe_res_var[:] = np.stack(all_cmwe_res, axis = 0)
    all_cmwe_res_var.units = "centimeters of water equivalent"

    rootgrp.close()