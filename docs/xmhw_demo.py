# The code uses features only available in xarray 0.16.0 +
# this should be ok both for conda/analysis3 and analysis3-unstable

import xarray as xr
import numpy as np
import dask
from xmhw.xmhw import threshold, detect

# using NOAA oisst as timeseries
ds =xr.open_mfdataset('/g/data/ua8/NOAA_OISST/AVHRR/v2-1_modified/timeseries/oisst_timeseries_*.nc',
                        concat_dim='time', combine='nested', chunks={'time':-1, 'lat': 10, 'lon': 10})
# removing zlev dimension, you can actually skip this since code will stack all dimensions which aren't time
sst =ds['sst'].squeeze()
sst = sst.drop('zlev')
# for the moment getting small region to test
# This correspond to ... ocean cell grid points
tos = sst.sel(lat=slice(-44,-41),lon=slice(144, 149))
# data is small enough to have 1 chunk)
# NB for each cell the timeseries should be in same chunk, fo this reason chunk({'time-dimension': -1})
# is included n the module where necessary
ts = tos.chunk({'time':-1, 'lat':-1, 'lon':-1})
ts

clim = threshold(tos)
# Save results to netcdf file
climds = xr.merge([clim['thresh'],clim['seas']])
climds.to_netcdf('climds.nc')
#climds = xr.open_dataset('climds.nc')

# Identify marine heat waves
mhw  = detect(tos, clim['thresh'], clim['seas'])
#mhw  = detect(tos, climds['threshold'], climds['seasonal'])

# unstack cell dimension to get back lat and lon
mhwds = mhw.unstack('cell')

# save mhw to yearly netcdf files (to split size)
#years, datasets = zip(*mhwds.groupby("time.year"))
#paths = ["mhw_%s.nc" % y for y in years]
#xr.save_mfdataset(datasets, paths)

# you can use this if only doing a subset
mhwds.to_netcdf('mhwds.nc')
