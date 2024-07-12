import numpy as np

def bilinear_interp(x, y, arr, x_query, y_query):
    """
    Bilinearly interpolates the value of arr, defined on a rectangular grid by x and y, at the set of points defined 
    by x_query and y_query. x, y, x_query, and y_query must all be in the same coordinate system.

    Parameters:
    arr: An array of floats of shape (ny, nx)
    x, y: x has shape (nx,) and y has shape (ny,). The entry arr[i][j] corresponds to the point in space defined by 
    y[i] and x[j]. The entries of x and y must be evenly spaced
    x_query, y_query: Both must have shape (n,), where n is the number of query points. It is assumed that x, y, 
    x_query, and y_query are defined in the same coordinate system and that x and x_query represent the same dimension, 
    while y and y_query represent the other dimension 

    Returns: 
    arr_query: An array of floats of shape (n,). arr_query[i] represents the estimated value of arr at the position 
    defined by x_query[i] and y_query[i], estimated using bilinear interpolation. If any query point i is outside the 
    bounds defined by x and y, arr_query[i] will be NaN
    """

    # Query points
    x_query = np.asarray(x_query)
    y_query = np.asarray(y_query)

    # Find the spacing of the x and y arrays
    sx, sy = x[1] - x[0], y[1] - y[0]
    a = sx * sy  # area of each grid cell

    # Find the indices of the four closest points in arr to each query point
    x0 = np.floor((x_query - x[0]) / sx).astype(int)   # Left
    x1 = x0 + 1                                        # Right
    y0 = np.floor((y_query - y[0]) / sy).astype(int)   # Bottom
    y1 = y0 + 1                                        # Top

    # ma[i] is True if both x_query[i] and y_query[i] are within the bounds of the grid defined by x and y
    ma = (x0 >= 0) & (x1 < len(x)) & (y0 >= 0) & (y1 < len(y))

    # Values at each of the reference coordinates
    Ia, Ib, Ic, Id = (np.empty_like(x_query, dtype=np.float64), np.empty_like(x_query, dtype=np.float64),
                     np.empty_like(x_query, dtype=np.float64), np.empty_like(x_query, dtype=np.float64))  # Allocate memory
    Ia[~ma], Ib[~ma], Ic[~ma], Id[~ma] = np.nan, np.nan, np.nan, np.nan    # Set to NaN where x or y are out of bounds
    Ia[ma] = arr[ y0[ma], x0[ma] ]    # Bottom left
    Ib[ma] = arr[ y1[ma], x0[ma] ]    # Top left
    Ic[ma] = arr[ y0[ma], x1[ma] ]    # Bottom right
    Id[ma] = arr[ y1[ma], x1[ma] ]    # Top right

    # Weights for each reference coordinate
    wa, wb, wc, wd = (np.empty_like(x_query, dtype=np.float64), np.empty_like(x_query, dtype=np.float64),
                     np.empty_like(x_query, dtype=np.float64), np.empty_like(x_query, dtype=np.float64))  # Allocate memory
    wa[~ma], wb[~ma], wc[~ma], wd[~ma] = np.nan, np.nan, np.nan, np.nan    # Set to NaN where x or y are out of bounds
    wa[ma] = ((x[x1[ma]] - x_query[ma]) * (y[y1[ma]] - y_query[ma])) / a
    wb[ma] = ((x[x1[ma]] - x_query[ma]) * (y_query[ma] - y[y0[ma]])) / a
    wc[ma] = ((x_query[ma] - x[x0[ma]]) * (y[y1[ma]] - y_query[ma])) / a
    wd[ma] = ((x_query[ma] - x[x0[ma]]) * (y_query[ma] - y[y0[ma]])) / a

    return wa*Ia + wb*Ib + wc*Ic + wd*Id     # arr_query in documentation