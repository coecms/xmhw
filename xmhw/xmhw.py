#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 ARC Centre of Excellence for Climate Extremes
# author: Paola Petrelli <paola.petrelli@utas.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import xarray as xr
import numpy as np
import dask
import sys
from .helpers import join_gaps, mhw_filter, runavg, dask_percentile, window_roll
from .helpers import land_check, feb29, add_doy 


def threshold(temp, climatologyPeriod=[None,None], pctile=90, windowHalfWidth=5, smoothPercentile=True, 
                   smoothPercentileWidth=31, maxPadLength=False, coldSpells=False, Ly=False):
    """Calculate threshold and seasonal climatology (varying with day-of-year)

    Inputs:

      temp    Temperature array

    Outputs:
        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle
        'missing'              A vector of TRUE/FALSE indicating which elements in 
                               temp were missing values for the MHWs detection

    Options:

      climatologyPeriod      Period over which climatology is calculated, specified
                             as list of start and end years. Default is to calculate
                             over the full range of years in the supplied time series.
                             Alternate periods suppled as a list e.g. [1983,2012].
      pctile                 Threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      windowHalfWidth        Width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      smoothPercentile       Boolean switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days])
      maxPadLength           Specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN. Set as an integer.
                             (DEFAULT = False, interpolates over all missing values).
      coldSpells             Specifies if the code should detect cold events instead of
                             heat events. (DEFAULT = False)
      Ly                     Specifies if the length of the year is < 365/366 days (e.g. a 
                             360 day year from a climate model). This affects the calculation
                             of the climatology. (DEFAULT = False)
    """

    # Set climatology period, if unset use full range of available data
    if all(climatologyPeriod):
        temp = temp.sel(time=slice(f'{climatologyPeriod[0]}-01-01',
                                   f'{climatologyPeriod[1]}-12-31'))
    # return an array stacked on lat/lon with land cells removed
    # new dimensions are (time,cell)
    ts = land_check(temp)
    ts = add_doy(ts,dim='time')

    # Flip temp time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        ts = pad(ts, maxPadLength=maxPadLength)

    # Rewrite ts array to get for each doy the values in a window +/-windowHalfWidth
    ts = ts.chunk({'time': -1, 'cell':1})
    twindow = ts.groupby('cell').apply(window_roll,
                  args=[windowHalfWidth])

    # rechunk twindow otherwise it is passed to dask_percentile as a numpy array 
    twindow = twindow.chunk({'z': -1})
    
     # Calculate threshold and seasonal climatology across years
    thresh_climYear = (twindow
                       .groupby('doy')
                       .reduce(dask_percentile, dim='z', q=pctile,
                       allow_lazy=True)).compute()
    seas_climYear = (twindow
                       .groupby('doy')
                       .reduce(np.nanmean)).compute()
    # calculate value for 29 Feb from mean fo 28-29 feb and 1 Mar
    thresh_climYear.loc[dict(doy=59)] = feb29(thresh_climYear)
    seas_climYear.loc[dict(doy=59)] = feb29(seas_climYear)

    # Smooth if desired
    if smoothPercentile:
        # If the length of year is < 365/366 (e.g. a 360 day year from a Climate Model)
        if Ly:
            valid = ~np.isnan(thresh_climYear)
        # >= 365-day year
        else:
            valid =  np.ones(len(thresh_climYear), dtype=bool)
        thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
        seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)
  # fix name of arrays
    thresh_climYear.name = 'threshold'
    seas_climYear.name = 'seasonal'
    # Save vector indicating which points in temp are missing values
    missing = xr.ufuncs.isnan(ts)
    # Set all remaining missing temp values equal to the climatology
    #seas_climYear = xr.where(missing, ts, seas_climYear)

    # Save in dictionary to follow what Eric does
    clim = {}
    #clim['thresh'] = thresh_climYear.reindex_like(ts.groupby('doy').mean()).unstack('cell')
    clim['thresh'] = thresh_climYear.unstack('cell')
    clim['seas'] = seas_climYear.unstack('cell')
    clim['missing'] = missing.unstack('cell')

    return clim
