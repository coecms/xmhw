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
import time
from .identify import (join_gaps, mhw_filter, runavg, dask_percentile, window_roll,
                      land_check, feb29, add_doy) 
#from .detail import mhw_ds
from .features import mhw_ds
from .exception import XmhwException


def threshold(temp, tdim='time', climatologyPeriod=[None,None], pctile=90, windowHalfWidth=5, smoothPercentile=True, 
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

      tdim                   String: time dimension name, default='time'
      climatologyPeriod      List of integers: period over which climatology is calculated, specified
                             as list of start and end years. Default is to calculate
                             over the full range of years in the supplied time series.
                             Alternate periods suppled as a list e.g. [1983,2012].
      pctile                 Integer: threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      windowHalfWidth        Integer: width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      smoothPercentile       Boolean: switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Integer: width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days], should be odd number)
      maxPadLength           Integer: specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN.
                             (DEFAULT = None, interpolates over all missing values).
      coldSpells             Boolean: specifies if the code should detect cold events instead of
                             heat events. (DEFAULT = False)
      Ly                     Boolean: specifies if the length of the year is < 365/366 days (e.g. a 
                             360 day year from a climate model). This affects the calculation
                             of the climatology. (DEFAULT = False)
    """

    # check smooth percentile window width is odd
    if smoothPercentileWidth%2 == 0:
        raise XmhwException("smoothPercentileWidth should be odd")

    # Set climatology period, if unset use full range of available data
    if all(climatologyPeriod):
        tslice = {tdim: slice(f'{climatologyPeriod[0]}-01-01', f'{climatologyPeriod[1]}-12-31')}
        temp = temp.sel(**tslice)
        #temp = temp.sel(time=slice(f'{climatologyPeriod[0]}-01-01',
        #                           f'{climatologyPeriod[1]}-12-31'))
    # return an array stacked on all dimensions excluded time
    # Land cells are removed
    # new dimensions are (time,cell)
    ts = land_check(temp, tdim=tdim)
    ts = add_doy(ts, tdim=tdim)

    # Flip ts time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        ts = pad(ts, maxPadLength=maxPadLength)

    # Apply window_roll to each cell.
    # Window_roll first finds for each day of the year all the ts values that falls in
    # a window +/-windowHalfWidth, then concatenates them in a new timeseries
    ts = ts.chunk({tdim: -1, 'cell':1})
    twindow = ts.groupby('cell').map(window_roll,
                  args=[windowHalfWidth, tdim])

    # rechunk twindow otherwise it is passed to dask_percentile as a numpy array 
    twindow = twindow.chunk({'z': -1})
    
     # Calculate threshold and seasonal climatology across years
    thresh_climYear = (twindow
                       .groupby('doy')
                       .reduce(dask_percentile, dim='z', q=pctile)).compute()
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
    missing = np.isnan(ts)
    # Set all remaining missing temp values equal to the climatology
    #seas_climYear = xr.where(missing, ts, seas_climYear)

    # Save in dictionary to follow what Eric does
    clim = {}
    #clim['thresh'] = thresh_climYear.reindex_like(ts.groupby('doy').mean()).unstack('cell')
    clim['thresh'] = thresh_climYear.unstack('cell')
    clim['seas'] = seas_climYear.unstack('cell')
    clim['missing'] = missing.unstack('cell')

    return clim

def detect(temp, thresh, seas, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=None, coldSpells=False, tdim='time'): 
    """

    Applies the Hobday et al. (2016) marine heat wave definition to an input time
    series of temp ('temp') along with a time vector ('t'). Outputs properties of
    all detected marine heat waves.

    Inputs:

      temp    Temperature array [1D  xarray of length T]
      clim    Climatology of SST. Each key (following list) is a seasonally-varying
              time series [1D numpy array of length T] of a particular measure:

        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle
        'missing'              A vector of TRUE/FALSE indicating which elements in 
                               temp were missing values for the MHWs detection

      
    
    Outputs:

      mhw     Detected marine heat waves (MHWs). Each key (following list) is a
              list of length N where N is the number of detected MHWs:
              ....


    Options:

      minDuration            Integer: minimum duration for acceptance detected MHWs
                             (DEFAULT = 5 [days])
      joinAcrossGaps         Boolean: switch indicating whether to join MHWs      
                             which occur before/after a short gap (DEFAULT = True)
      maxGap                 Maximum length of gap allowed for the joining of MHWs
                             (DEFAULT = 2 [days])
      maxPadLength           Integer: specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN.
                             (DEFAULT = None, interpolates over all missing values, boolean).
      coldSpells             Boolean: specifies if the code should detect cold events instead of
                             heat events. (DEFAULT = False)
      tdim                   String: name of time dimension. (DEFAULT='time')
    """
  
   
    
    # check maxGap < minDuration 
    if maxGap >= minDuration:
        raise XmhwException("Maximum gap between mhw events should be smaller than event minimum duration")

    ts = land_check(temp)
    thresh = land_check(thresh, tdim='doy')
    seas = land_check(seas, tdim='doy')
    # assign doy 
    ts = add_doy(ts)

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        ts = pad(ts, maxPadLength=maxPadLength)
    # Flip temp time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Find MHWs as exceedances above the threshold
    #

    # Time series of "True" when threshold is exceeded, "False" otherwise
    start = time.process_time()
    exceed_bool = ts > thresh.sel(doy=ts.doy)
    print('after exceed', (time.process_time()-start)/60.)
    exceed_bool = exceed_bool.chunk(chunks={tdim: -1})

    # Find all MHW events of duration >= minDuration   
    ds = mhw_filter(exceed_bool, minDuration, joinAcrossGaps, maxGap, tdim=tdim)
    # Save mhw characteristic in dataset still working on this 
    mhw = mhw_ds(ds, ts, thresh, seas, tdim=tdim)
    #mhw = m.compute()
    # Flip climatology and intensties in case of cold spell detection
    # do I need to flip climatologies here?
    if coldSpells:
        mhw = flip_cold(mhw)
    return   mhw
