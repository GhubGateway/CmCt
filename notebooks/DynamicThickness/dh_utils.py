import numpy as np
from datetime import datetime
import concurrent.futures
import netCDF4

######################### INTERPRETTING INPUTS #########################
def manage_comparisons(desired_comparisons, obs = True):
    """
    Outputs comp_info, whose purpose is to reduce the information in desired_comparisons to the minimum necessary as it relates
    to the observation (if obs = True) or the model (if obs = False). This is used by the read_obs_file and read_mod_file
    functions of the GIS_Compare_Dynamic_Thickness.ipynb notebook of the CmCt.

    comp_info is a dictionary with the following structure:
        Key: source id
        Value: Dictionary with the following structure:
            Key: time_id, a 2-tuple of ints of the form (start_year, end_year)
            Value: List of indices in desired_comparisons which ask for the time range in time_id for the source in source id

    If obs is True, the source id for each comparison will be drawn from the first entry (aka it will be of the form "IMAU", "GSFC"
    or "GEMB". If obs is False, the source id will be drawn from the first entry and will be an int representing an index in the 
    model_fns input variable to the CmCt
    """
    comp_info = {}   

    def deconstruct_comp(comp, obs):
        if obs:
            src, _, start_year, end_year = comp
        else:
            _, src, start_year, end_year = comp
        return src, (start_year, end_year)

    # Find out which of the files need to be read for the comparisons, and which years of those files need to be read
    for i, comp in enumerate(desired_comparisons):
        src, time_id = deconstruct_comp(comp, obs)
        comp_info.setdefault(src, {})
        (comp_info[src]).setdefault(time_id, [])
        (comp_info[src])[time_id].append(i)
    return comp_info


######################### SAVING OUTPUTS #########################
def save_residuals_to_netcdf(output_fns, all_dh_res, x_UTM, y_UTM, desired_comparisons, crs_wkt,
                            single_file_nc, model_fn_ids):
    """
    If single_file_nc is false, this function acts as a wrapper function for helper_save_one_residual_to_netcdf 
    and parallelizes the saving of all residuals. If single_file_nc is true, this function calls 
    helper_save_all_residuals_to_one_netcdf. In the special case that there is only one comparison, the function
    will call helper_save_one_residual_to_netcdf only once, without parallelizing
    """
    
    if len(all_dh_res) == 1:
        helper_save_one_residual_to_netcdf((output_fns[0], all_dh_res[0], (desired_comparisons[0])[2], (desired_comparisons[0])[3]))
    elif single_file_nc:
        helper_save_all_residuals_to_one_netcdf(output_fns[0], all_dh_res, x_UTM, y_UTM, desired_comparisons, 
                                                crs_wkt, model_fn_ids)
    else:
        inputs = []
        for i in range(len(desired_comparisons)):
            inputs.append((output_fns[i], all_dh_res[i], (desired_comparisons[i])[2], (desired_comparisons[i])[3],
                           crs_wkt, x_UTM, y_UTM))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(helper_save_one_residual_to_netcdf, inputs)
            for result in results:
                pass


def helper_save_one_residual_to_netcdf(tuple):
    """
    Saves the residual of a single comparison to a netcdf file
    """
    fn, dh_res, start_year, end_year, crs_wkt, x_UTM, y_UTM = tuple
    
    rootgrp = netCDF4.Dataset(fn, "w", format="NETCDF4")
    
    # Record coordinate system
    spatial_ref_var = rootgrp.createVariable("spatial_ref", "i8")
    spatial_ref_var[:] = 0
    spatial_ref_var.crs_wkt = crs_wkt
    
    # Set up x and y variables
    x_dim = rootgrp.createDimension("x", len(x_UTM))
    y_dim = rootgrp.createDimension("y", len(y_UTM))
    x_var = rootgrp.createVariable("x", "f4", ("x",))
    y_var = rootgrp.createVariable("y", "f4", ("y",))
    x_var[:], y_var[:] = x_UTM, y_UTM
    x_var.units, y_var.units = "meter", "meter"
    
    # Set up dynamic thickness residual variable
    dh_res_var = rootgrp.createVariable("dh_res", "f4", ("y","x",))
    dh_res_var[:,:] = dh_res
    dh_res_var.coordinates = "spatial_ref"
    dh_res_var.grid_mapping = "spatial_ref"
    dh_res_var.long_name = ("Residual of annual dynamic mass/ice thickness (assume ice density of 917 kg/m3) change from Sep 1st " + str(start_year) + 
                            " to Aug 31st " + str(end_year))
    dh_res_var.units = "meters"
    rootgrp.close()

def helper_save_all_residuals_to_one_netcdf(fn, all_dh_res, x_UTM, y_UTM, desired_comparisons, crs_wkt, model_fn_ids):
    rootgrp = netCDF4.Dataset(fn, "w", format="NETCDF4")
    
    # Record coordinate system
    spatial_ref_var = rootgrp.createVariable("spatial_ref", "i8")
    spatial_ref_var[:] = 0
    spatial_ref_var.crs_wkt = crs_wkt
    
    # Set up x and y variables
    x_dim = rootgrp.createDimension("x", len(x_UTM))
    y_dim = rootgrp.createDimension("y", len(y_UTM))
    x_var = rootgrp.createVariable("x", "f4", ("x",))
    y_var = rootgrp.createVariable("y", "f4", ("y",))
    x_var[:], y_var[:] = x_UTM, y_UTM
    x_var.units, y_var.units = "meter", "meter"

    # Read in desired comparisons as four lists, not a list of 4-tuples
    srcs, id_idxs, str_ys, end_ys = [], [], [], []
    for comp in desired_comparisons:
        srcs.append(comp[0])
        id_idxs.append(comp[1])
        str_ys.append(comp[2])
        end_ys.append(comp[3])

    # Set up comparison information dimension
    comp_dim = rootgrp.createDimension("comparison", len(desired_comparisons))
    obs_src_var = rootgrp.createVariable("obs_src", np.dtype('U4'), ("comparison",))
    model_id_var = rootgrp.createVariable("model_id", np.dtype('U'), ("comparison",))
    start_year_var = rootgrp.createVariable("start_year", 'i2', ("comparison",))
    end_year_var = rootgrp.createVariable("end_year", 'i2', ("comparison",))
    if model_fn_ids is None:
        model_fn_ids = model_fns
        
    for j in range(len(desired_comparisons)):
        obs_src_var[j] = srcs[j]
        model_id_var[j] = model_fn_ids[id_idxs[j]]
        start_year_var[j] = str_ys[j]
        end_year_var[j] = end_ys[j]

    # Fill in dynamic thickness information
    all_dh_res_var = rootgrp.createVariable("all_dh_res", "f8", ("comparison","y", "x",))
    all_dh_res_var[:] = np.stack(all_dh_res, axis = 0)
    all_dh_res_var.units = "meters"
    all_dh_res_var.coordinates = "spatial_ref"
    all_dh_res_var.grid_mapping = "spatial_ref"

    rootgrp.close()