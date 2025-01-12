# Shared utilities
import numpy as np
import cftime 
import datetime

def check_datarange(time_var, start_date_cftime, end_date_cftime):
    calendar_type = time_var.to_index().calendar
       
    # Get the minimum and maximum values directly from the time variable
    min_time = time_var.values.min()
    max_time = time_var.values.max()
    
    # Check if the selected start and end dates are within the range
    if min_time <= start_date_cftime <= max_time and min_time <= end_date_cftime <= max_time:
        print(f"The selected dates {{start_date_cftime}} and {{end_date_cftime}} are within the range of the model data.")
    else:
        raise ValueError(f"Error: The selected dates {{start_date_cftime}} or {{end_date_cftime}} are out of range. Model data time range is from {{min_time}} to {{max_time}}.")


def days_in_year(date):
    if date.calendar == '365_day' or date.calendar == 'noleap':
        diy = 365
    elif date.calendar == '366_day' or date.calendar == 'all_leap':
        diy = 366
    elif date.calendar == '360_day':
        diy = 360
    else:
        if cftime.is_leap_year(date.year, date.calendar):
            diy = 366
        else:
            diy = 365

    return diy


def dt64_to_fractional_year(dt64):
    year = dt64.astype('datetime64[Y]').astype(int) + 1970
    start_of_year = np.datetime64(f'{year}-01-01')
    end_of_year = np.datetime64(f'{year + 1}-01-01')
    return year + (dt64 - start_of_year) / (end_of_year - start_of_year)


def dt64_to_years_months(times_dt64):
    years  = np.array([i.astype('datetime64[Y]').astype(int) + 1970   for i in times_dt64])
    months = np.array([i.astype('datetime64[M]').astype(int) % 12 + 1 for i in times_dt64])

    return years, months
