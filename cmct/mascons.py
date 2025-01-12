import numpy as np
import xarray as xr
import h5py

from pyproj import Geod

from urllib import request
from http.cookiejar import CookieJar
import netrc
import requests
import s3fs
import getpass

class GSFCmascons:
    def __init__(self, f, lon_wrap='pm180'):
        self.lat_centers = f['/mascon/lat_center'][0][:]
        self.lat_spans = f['/mascon/lat_span'][0][:]
        self.lon_centers = f['/mascon/lon_center'][0][:]
        self.lon_spans = f['/mascon/lon_span'][0][:]
        self.locations = f['/mascon/location'][0][:]
        self.basins = f['/mascon/basin'][0][:]
        self.areas = f['/mascon/area_km2'][0][:]
        self.cmwe = f['/solution/cmwe'][:]
        
        self.days_start = f['/time/ref_days_first'][0][:]
        self.days_middle = f['/time/ref_days_middle'][0][:]
        self.days_end = f['/time/ref_days_last'][0][:]
        self.times_start = self._set_times_as_datetimes(self.days_start)
        self.times_middle = self._set_times_as_datetimes(self.days_middle)
        self.times_end = self._set_times_as_datetimes(self.days_end)

        self.N_mascons = len(self.lat_centers)
        self.N_times = len(self.days_middle)
        self.labels = np.array([i for i in range(self.N_mascons)])
        
        self.reset_lon_bounds(lon_wrap)
        
        self.min_lats = self.lat_centers - self.lat_spans/2
        self.max_lats = self.lat_centers + self.lat_spans/2
        self.max_lats[self.min_lats < -90.0] = -89.5
        self.min_lats[self.min_lats < -90.0] = -90.0
        self.min_lats[self.max_lats > 90.0] = 89.5
        self.max_lats[self.max_lats > 90.0] = 90.0
        self.min_lons = self.lon_centers - self.lon_spans/2
        self.max_lons = self.lon_centers + self.lon_spans/2

    def reset_lon_bounds(self, lon_wrap):
        if lon_wrap == 'pm180':
            self.lon_centers[self.lon_centers > 180] -= 360
        elif lon_wrap == '0to360':
            self.lon_centers[self.lon_centers < 0] += 360

    def _set_times_as_datetimes(self, days):
        return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')
    
    def as_dataset(self):
        ds = xr.Dataset({'cmwe': (['label', 'time'], self.cmwe),
                         'lat_centers': ('label', self.lat_centers),
                         'lat_spans': ('label', self.lat_spans),
                         'lon_centers': ('label', self.lon_centers),
                         'lat_spans': ('label', self.lon_spans),
                         'areas': ('label', self.areas),
                         'basins': ('label', self.basins),
                         'locations': ('label', self.locations),
                         'basins': ('label', self.basins),
                         'lats_max': ('label', self.max_lats),
                         'lats_min': ('label', self.min_lats),
                         'lons_max': ('label', self.max_lons),
                         'lons_min': ('label', self.min_lons),
                         'times_start': ('time', self.times_start),
                         'times_end': ('time', self.times_end),
                         'days_start': ('time', self.days_start),
                         'days_middle': ('time', self.days_middle),
                         'days_end': ('time', self.days_end)
                        }, coords={'label': self.labels, 'time': self.times_middle})
        return ds

def load_gsfc_solution(h5_filename, lon_wrap='pm180'):
    with h5py.File(h5_filename, mode='r') as f:
        mascons = GSFCmascons(f, lon_wrap)
    return mascons

def setup_earthdata_login_auth(endpoint):
    """
    Set up the request library so that it authenticates against the given Earthdata Login
    endpoint and is able to track cookies between requests.  This looks in the .netrc file
    first and if no credentials are found, it prompts for them.
    Valid endpoints:
        urs.earthdata.nasa.gov - Earthdata Login production
    """
    try:
        username, _, password = netrc.netrc().authenticators(endpoint)
        print('# Your URS credentials were securely retrieved from your .netrc file.')
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        print("There's no .netrc file or the The endpoint isn't in the netrc file. Please provide...")
        print('# Your info will only be passed to %s and will not be exposed in Jupyter.' % (endpoint))
        username = input('Username: ')
        password = getpass.getpass('Password: ')

    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)

    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)

def begin_s3_direct_access():
    url="https://archive.podaac.earthdata.nasa.gov/s3credentials"
    r = requests.get(url)
    response = r.json()
    return s3fs.S3FileSystem(key=response['accessKeyId'],secret=response['secretAccessKey'],token=response['sessionToken'],client_kwargs={'region_name':'us-west-2'})

def calculate_area(lats, lons):
    """Calculates the area of a polygon given its latitude and longitude coordinates."""

    geod = Geod(ellps='WGS84')  # Use WGS84 ellipsoid
    poly_area, poly_perimeter = geod.polygon_area_perimeter(lons, lats)

    return abs(poly_area)  # Area is always positive

def load_jpl_solution(s3_bucket, lon_wrap='pm180'):
    # Initiate PO.DAAC S3 connection
    fs = begin_s3_direct_access()

    # Get list of JPL mascon solutions
    s3_files = fs.glob(s3_bucket + '/*nc')

    # Get data from last (latest) JPL solution:
    nc_file = s3_files[-1]

    lats = None
    lons = None
    cmwe = None
    time_dt64 = None
    with fs.open(nc_file, 'rb') as f:
        ds = xr.open_dataset(f, engine='h5netcdf')
        #print(ds)

        lats = ds.lat.values
        lons = ds.lon.values
        cmwe = ds.lwe_thickness.values
        time_dt64 = ds.time.values

        mascon_ID = ds.mascon_ID.values
        land_mask = ds.land_mask.values
        scale_factor = ds.scale_factor.values

        lat_bounds = ds.lat_bounds.values
        lon_bounds = ds.lon_bounds.values

        ds.close()

        lonsm, latsm = np.meshgrid(lons, lats)
        lat_spansm, lon_spansm = np.meshgrid(np.diff(lat_bounds), np.diff(lon_bounds))
        ds_new = dict()
        ds_new['/mascon/lat_center'] = [latsm.ravel()]
        ds_new['/mascon/lat_span'] = [lat_spansm.ravel()]
        ds_new['/mascon/lon_center'] = [lonsm.ravel()]
        ds_new['/mascon/lon_span'] = [lon_spansm.ravel()]

        # Assign locations (GrIS == 1; AIS == 3)
        gis_ids = [17,18,33,34,35,36,56,57,58,59,86,87,88,89,90,123,124,125,126,164,165,166,167,168,211,212,213,214,265,266,324]
        #gis_area = 2189202.95 # km2
        ais_ids = [4324,4325,4340,4341,4344,4345,4346,4347,4348,4349,4350,4351,4352,4353,4372,4373,4374,4382,4383,4385,4386,4387,4388,4389,4390,4391,4392,4393,4394,4395,4396,4397,4398,4399,4400,4415,4416,4417,4424,4425,4426,4427,4428,4429,4430,4431,4432,4433,4434,4435,4436,4437,4438,4439,4440,4441,4448,4450,4451,4452,4453,4454,4459,4460,4461,4462,4463,4464,4465,4466,4467,4468,4469,4470,4471,4472,4473,4474,4475,4479,4480,4481,4482,4483,4484,4485,4486,4489,4490,4491,4492,4493,4494,4495,4496,4497,4498,4499,4500,4501,4502,4505,4506,4507,4508,4509,4510,4511,4512,4513,4514,4515,4516,4517,4518,4519,4520,4521,4522,4523,4524,4525,4526,4527,4528,4529,4530,4531,4532,4533,4534,4535,4536,4537,4538,4539,4540,4541,4542,4543,4544,4545,4546,4547,4548,4549,4550,4551]
        #ais_area = 12425189.72 # km2
        location = np.zeros(ds_new['/mascon/lat_span'][0].shape)
        for id in gis_ids:
            location[mascon_ID.ravel()==id] = 1
        for id in ais_ids:
            location[mascon_ID.ravel()==id] = 3
        location[land_mask.ravel()==0] = 0
        ds_new['/mascon/location'] = [location]
        ds_new['/mascon/basin'] = [np.nan * np.ones(mascon_ID.shape).ravel()]

        # Calculate mascon areas
        areas = np.zeros(lonsm.shape)
        for r in range(lonsm.shape[0]):
            for c in range(lonsm.shape[1]):
                lats_poly = [lat_bounds[r,0], lat_bounds[r,1], lat_bounds[r,1], lat_bounds[r,0], lat_bounds[r,0]]
                lons_poly = [lon_bounds[c,0], lon_bounds[c,0], lon_bounds[c,1], lon_bounds[c,1], lon_bounds[c,0]]
                area = calculate_area(lats_poly, lons_poly)
                areas[r,c] = area*1e-6
    
        ds_new['/mascon/area_km2'] = [areas.ravel()]
        ds_new['/solution/cmwe'] = cmwe.transpose(1,2,0).reshape(-1,cmwe.shape[0])

        epoch = np.datetime64('2002-01-01T00:00:00')
        ds_new['/time/ref_days_first'] = [np.zeros(len(time_dt64))]
        ds_new['/time/ref_days_middle'] = [(time_dt64 - epoch) / np.timedelta64(1, 'D')]
        ds_new['/time/ref_days_last'] = [np.zeros(len(time_dt64))]

        ds_new['time_dt64'] = time_dt64
        ds_new['lats'] = lats
        ds_new['lons'] = lons
        ds_new['lat_bounds'] = lat_bounds
        ds_new['lon_bounds'] = lon_bounds
        ds_new['mascon_ID'] = mascon_ID
        ds_new['land_mask'] = land_mask
        ds_new['cmwe'] = cmwe

    mascons = GSFCmascons(ds_new, lon_wrap)
    return ds_new, mascons

def points_to_mascons(mascons, lats, lons, values):
    d2r = np.pi/180
    
    min_lats = mascons.lat_centers - mascons.lat_spans/2
    max_lats = mascons.lat_centers + mascons.lat_spans/2
    min_lons = mascons.lon_centers - mascons.lon_spans/2
    max_lons = mascons.lon_centers + mascons.lon_spans/2
    
    mscn_mean = np.nan * np.ones(mascons.N_mascons)
    for i in range(mascons.N_mascons):
        
        if np.min(lats) > max_lats[i]:
            continue
        if np.max(lats) < min_lats[i]:
            continue
        if np.min(lons) > max_lons[i]:
            continue
        if np.max(lons) < min_lons[i]:
            continue
        
        I_ = (lats >= min_lats[i]) & (lats < max_lats[i]) & (lons >= min_lons[i]) & (lons < max_lons[i])
        m = values[I_]
        m_lats = lats[I_]
        
        m_lats = m_lats[~np.isnan(m)]
        m = m[~np.isnan(m)]
        
        if len(m) == 0:
            continue
        if np.sum(~np.isnan(m)) == 0:
            continue

        cos_weight = np.cos(m_lats*d2r)
        mscn_mean[i] = np.nanmean(m) # * cos_weight) / (np.sum(cos_weight) * len(m))
    
    return mscn_mean

def calc_mascon_delta_cmwe(mascons, start_date, end_date):
    t_0 = np.datetime64(start_date)
    t_1 = np.datetime64(end_date)
    
    i_0 = np.abs(mascons.times_middle - t_0).argmin()
    i_1 = np.abs(mascons.times_middle - t_1).argmin()
    
    return mascons.cmwe[:,i_1] - mascons.cmwe[:,i_0]

def calc_mass_delta_Gt_timeseries(mascons, start_date, end_date, I_):
    t_0 = np.datetime64(start_date)
    t_1 = np.datetime64(end_date)
     
    i_0 = np.abs(mascons.times_middle - t_0).argmin()
    i_1 = np.abs(mascons.times_middle - t_1).argmin()
     
    time_timeseries = list()
    mass_delta_Gt_timeseries = list()
    for i in range(i_0,i_1+1):
        mascon_delta_cmwe = mascons.cmwe[I_,i] - mascons.cmwe[I_,i_0]
        mass_delta_Gt_timeseries.append(np.sum(mascons.areas[I_]*1000*1000*mascon_delta_cmwe/100) * 1000 * 1e-12)
        time_timeseries.append(mascons.times_middle[i])

    return np.array(time_timeseries), np.array(mass_delta_Gt_timeseries)
