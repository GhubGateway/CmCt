import numpy as np
import pyproj
import cartopy.crs as ccrs
import time
import warnings


def flatten_to_list_of_points(x, y):
    """
    Example: x = [0, 1, 2, 3], y = [-0.5, -0.33, -0.25]
    x_flat = [   0,     0,     0,    1,     1,     1,    2,     2,     2,    3,     3,    3]
    y_flat = [-0.5, -0.33, -0.25, -0.5, -0.33, -0.25, -0.5, -0.33, -0.25, -0.5, -0.33, -0.25]

    
    Parameters:
    x, y: 1D arrays that need not be the same length. x and y denote the coordinates of a
          regular grid such that there is data at every point x[i], y[j] for any i and j 
          in range. 

    Returns:
    x_flat, y_flat: 1D arrays. If the length of the input x is nx and the length of the input y 
                    is ny, then the length of both x_flat and y_flat is nx*ny. For the point 
                    (x[i], y[j]), x_flat[k] = x[i] and y_flat[k] = y[j] for k = nx*i + j
    """
    xv, yv = np.meshgrid(x, y)
    x_flat, y_flat = xv.transpose().flatten(), yv.transpose().flatten()
    return x_flat, y_flat

def crs_ps():
    # Returns a pyproj CRS object representing the ISMIP6 coordinate system
    return pyproj.crs.CRS(ccrs.Stereographic(central_latitude= 90.0, central_longitude= -45.0, false_easting= 0.0, 
                            false_northing= 0.0, true_scale_latitude= 70.0, globe=ccrs.Globe('WGS84')))

def grid_centers_from_extent_and_res(extent, grid_size):
    x_centers = np.arange(extent[0]+grid_size/2, extent[1]-grid_size/2, grid_size)
    y_centers = np.arange(extent[2]+grid_size/2, extent[3]-grid_size/2, grid_size)
    return x_centers, y_centers
    

def regrid_data(x_centers, y_centers, x, y, data):
    x_span, y_span = x_centers[1] - x_centers[0], y_centers[1] - y_centers[0]
    
    """
    Takes the information in data and moves it into a grid specified by x_centers and y_centers. 
    data is a list of 1D numpy arrays, which all map onto the same x and y coordinates x and y.

    Parameters:
    x_centers, y_centers: 1D arrays which need not have the same length. Assumed to be 
    evenly spaced and increasing. May be in any projection
    x, y: 1D arrays which must have the same length. Assumed to be in the same projection as
    x_centers and y_centers.
    data: A list of 1D arrays. All arrays must have the same length as x and y. For the sake of
    documentation, let data have a length of n and let each entry of data have a length m

    Returns:
    arrs: arrs is a list (of length n) of arrays each with shape (len(y_centers), len(x_centers))
    arrs[w][i, j] is the average of all points in data[w] that are contained within the grid 
    cell specified by x_centers[j], y_centers[i]. In other words, arr[i, j] is the average of 
    data[w][k] for all k such that x_centers[j] - x_span/2 =< x[k] < x_centers[j] + x_span/2 and 
    y_centers[i] - y_span/2 =< y[k] < y_centers[i] + y_span/2. In the case that there are no datapoints
    that fit these conditions, arrs[w][i,j] will be NaN
    """
    n = len(data)
    arrs = []
    for i in range(n):
        arrs.append(np.empty((len(y_centers), len(x_centers)), dtype = "float") * np.nan)

    sorter = np.argsort(x)   # Gives indices that would sort x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for i, x_c in enumerate(x_centers):
            # Find indices that have x values in the right range
            left = np.searchsorted(x, [x_c - x_span/2], side = "left", sorter = sorter)
            right = np.searchsorted(x, [x_c + x_span/2], side = "right", sorter = sorter)
            left, right = left[0], right[0]
            
            for j, y_c in enumerate(y_centers):
                # Search all points for which x is in range
                K_ = (y[left:right] >= y_c - y_span/2) & (y[left:right] < y_c + y_span/2)
                for w in range(n):
                    (arrs[w])[j, i] = np.nanmean(((data[w])[left:right])[K_])
    return arrs