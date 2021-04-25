# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:38:14 2021

@author: Jack Blunt
"""
t0 = 288. # units.kelvin
p0 = 1013.25 # units.hPa
gamma = 6.5 # unit = 'K/km'
Rd = 287.058 #dry_air_gas_constant J/(kg. K)
g = 9.806 # m/s2

def height_to_pressure_std(height, difference=True):
    height = int(height)
    t0 = 288. # units.kelvin
    p0 = 1013.25 # units.hPa
    gamma = 6.5 # unit = 'K/km'
    Rd = 287.058 #dry_air_gas_constant J/(kg. K)
    g = 9.806 # m/s2
    """Convert height data to pressures using the U.S. standard atmosphere [NOAA1976]_.

    The implementation inverts the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    height : `pint.Quantity`
        Atmospheric height

    Returns
    -------
    `pint.Quantity`
        Corresponding pressure value(s)

    Notes
    -----
    .. math:: p = p_0 e^{\frac{g}{R \Gamma} \text{ln}(1-\frac{Z \Gamma}{T_0})}

    """
    
    if difference:
        preassure = p0 * (1 - (gamma / t0) * height) ** (g / (Rd * gamma)) - p0 * (1 - (gamma / t0) * 0) ** (g / (Rd * gamma))
    else:
        preassure = p0 * (1 - (gamma / t0) * height) ** (g / (Rd * gamma))
        
    
    return int(preassure.real)


#using the function for static density https://fluids.readthedocs.io/fluids.atmosphere.html
"""
Rh is humidity
P is preasure (Pa)
Ps is watervapor preassure
"""
#density = (0.0034848/(t+273.15))(P−0.0037960⋅Rh⋅Ps)

def density_(T, P):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        density of air as a function of `T` and `P`. MW is defined as 28.9644
        g/mol, and R as 8314.32 J/kmol/K

        .. math::
            \rho_g = \frac{P\cdot MW}{T\cdot R\cdot 1000}

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]

        Returns
        -------
        rho : float
            Mass density, [kg/m^3]
        '''
        # 0.00348367635597379 = M0/R
        return P*0.00348367635597379/T
#%%
#https://psl.noaa.gov/thredds/catalog/Datasets/cmap/enh/catalog.html
import datetime as dt  # Python standard library datetime  module
from scipy.io import netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim )
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars



#nc_f = "data_precip/precip.mon.ltm.1991-2020.nc"
nc_f = "data_precip/precip.pentad.mean.nc"
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
                             
nc_f_air = 'data_precip/air.sig995.2012.nc'  # Your filename
nc_fid_air = Dataset(nc_f_air, 'r')  # Dataset is the class behavior to open the file

nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

# Extract data from NetCDF file
lats = nc_fid.variables['lat'][:]  # extract/copy the data
lons = nc_fid.variables['lon'][:]
time = nc_fid.variables['time'][:]
precip = nc_fid.variables['precip'][:] #orderd time, lat, long

nc_attrs_air, nc_dims_air, nc_vars_air = ncdump(nc_fid)
# Extract data from NetCDF file
lats_air = nc_fid_air.variables['lat'][:]  # extract/copy the data
lons_air = nc_fid_air.variables['lon'][:]
time_air = nc_fid_air.variables['time'][:]
air = nc_fid_air.variables['air'][:]  # shape is time, lat, lon as shown above

#%%


time_idx = -295  # most recent time
# Python and the renalaysis are slightly off in time so this fixes that problem
offset = dt.timedelta(hours=48)
# List of all times in the file as datetime objects
#start date = 1800-01-01 00:00:0.0
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time]

cur_time = dt_time[time_idx]
print(cur_time)

def get_precipitaion_and_temp(coord):
    lon180, lat = coord
    #coord lon is from -180 to 180
    #there fore -180 needs to be set to zero => add 180 to 
    longitude360 = lon180 + 180
    
    lat_id = np.argmin(np.abs(np.array(lats)-lat))
    lon_id = np.argmin(np.abs(np.array(lons)-longitude360))
    print("returning precipitation {} at time {}".format(precip[-1][lat_id][lon_id], cur_time))
    
    lat_id_air = np.abs(lats_air - lat).argmin()
    lon_id_air = np.abs(lons_air - longitude360).argmin()
    
    
    return precip[-1][lat_id][lon_id], air[time_idx, lat_id_air, lon_id_air] - 273.15
    
#%%
"""
nc_f_air = 'data_precip/air.sig995.2012.nc'  # Your filename
nc_fid_air = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
nc_attrs_air, nc_dims_air, nc_vars_air = ncdump(nc_fid)
# Extract data from NetCDF file
lats_air = nc_fid_air.variables['lat'][:]  # extract/copy the data
lons_air = nc_fid_air.variables['lon'][:]
time_air = nc_fid_air.variables['time'][:]
air = nc_fid_air.variables['air'][:]  # shape is time, lat, lon as shown above

print(lats_air)
print(lons_air)

time_idx = -1  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
offset = dt.timedelta(hours=48)
# List of all times in the file as datetime objects
dt_time = [dt.date(1, 1, 1) + dt.timedelta(hours=t) - offset\
           for t in time]
cur_time = dt_time[time_idx]

darwin = {'name': 'Darwin, Australia', 'lat': -12.45, 'lon': 130.83}
# Find the nearest latitude and longitude for Darwin


fig = plt.figure()
plt.plot(dt_time, air[:, lat_idx, lon_idx], c='r')
plt.plot(dt_time[time_idx], air[time_idx, lat_idx, lon_idx], c='b', marker='o')
plt.text(dt_time[time_idx], air[time_idx, lat_idx, lon_idx], cur_time,\
         ha='right')
fig.autofmt_xdate()
plt.ylabel("%s (%s)" % (nc_fid.variables['air'].var_desc,\
                        nc_fid.variables['air'].units))
plt.xlabel("Time")
plt.title("%s from\n%s for %s" % (nc_fid.variables['air'].var_desc,\
                                  darwin['name'], cur_time.year))

#%%
print(air[time_idx, lat_idx, lon_idx] - 273.15)"""
