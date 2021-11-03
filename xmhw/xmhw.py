#!/usr/bin/env
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
import pandas as pd
import dask
import sys
import time
from .identify import (join_gaps, define_events, runavg, window_roll, calculate_thresh,
                      land_check, feb29, add_doy, get_calendar, annotate_ds, calculate_seas)
from .features import flip_cold
from .exception import XmhwException


def threshold(temp, tdim='time', climatologyPeriod=[None,None], pctile=90, windowHalfWidth=5, smoothPercentile=True, 
                   smoothPercentileWidth=31, maxPadLength=None, coldSpells=False, Ly=False, anynans=False, skipna=False):
    """Calculate threshold and seasonal climatology (varying with day-of-year)

    Parameters
    ----------
    temp    Temperature array
    tdim: str
        Name of time dimension. (DEFAULT='time')
    climatologyPeriod: list(int)
        Period over which climatology is calculated, specified as list of start and end years. Optional,
        default is to use the full time series.
    pctile: int 
        Threshold percentile (%) for detection of extreme values (dafault 90)
    windowHalfWidth: int
        Width of window (one sided) about day-of-year used for the pooling of values and calculation of
        threshold percentile (default 5)
    smoothPercentile: bool 
        If True smooth the threshold percentile timeseries with a moving average (default True)
    smoothPercentileWidth: int
        Width of moving average window for smoothing threshold in days, has to be odd number (dafault 31)
    maxPadLength: int
        Specifies the maximum length in days over which to interpolate NaNs in input temp time series.
        i.e., any consecutive blocks of NaNs with length greater than maxPadLength will be left as NaN.
        If None it does not interpolate (default None).
    coldSpells: bool
        Specifies if the code should detect cold events instead of heat events (default False)
    Ly                     Boolean: specifies if the length of the year is < 365/366 days (e.g. a 
                             360 day year from a climate model). This affects the calculation
                             of the climatology. (DEFAULT = False)
    anynans: bool
        Define in land_check which cells will be dropped, if False only ones with all nans values, if True
        then all cells with even 1 NaN along time dimension will be dropped (dafault False)
    skipna: bool 
        Determines if percentile and mean will use skipna=True or False, the second is default as it is faster.
        (default False)

    Returns
    -------
    clim : xarray Dataset
        includes thresh seasonally varying threshold (e.g., 90th percentile)
                 seas   climatological seasonal cycle
    """

    # check smooth percentile window width is odd
    if smoothPercentileWidth%2 == 0:
        raise XmhwException("smoothPercentileWidth should be odd")
    # check that time dimension (tdim) is present
    if tdim not in temp.dims: 
        raise XmhwException(f"{tdim} dimension not present, default is 'time' or pass as tdim='time_dimension_name'")

    # Set climatology period, if unset use full range of available data
    if all(climatologyPeriod):
        tslice = {tdim: slice(f'{climatologyPeriod[0]}-01-01', f'{climatologyPeriod[1]}-12-31')}
        temp = temp.sel(**tslice)
    # save original attributes in a dictionary to be assigned to final datset
    ds_attrs = {}
    ds_attrs['ts'] = temp.attrs
    #ds_attrs[tdim+'encoding'] = temp.encoding
    for c in temp.dims:
        ds_attrs[c] = temp[c].attrs
    # return an array stacked on all dimensions excluded time
    # Land cells are removed
    # new dimensions are (time,cell)
    ts = land_check(temp, tdim=tdim, removeNans=anynans)

    # check if the calendar attribute is present in series time dimension
    # if not try to guess length of year
    year_days = get_calendar(ts[tdim])
    if year_days == 365.25:
        ts = add_doy(ts, tdim=tdim)
    else:
        XMHW.Exception(f"Module is not yet set to work with a calendar different from gregorian, standard, proleptic_gregorian. NB We treat all these calendars in the same way in the assumption that the timeseries starts after 1582")

    # Flip ts time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Pad missing values for all consecutive missing blocks of length <= maxPadLength
    # NB this is not happening by default and there could be issues if nan values are present in the timeseries
    if maxPadLength:
        ts = ts.interpolate_na(dim=tdim, max_gap=maxPadLength)

    # Apply window_roll to each cell.
    # Window_roll first finds for each day of the year all the ts values that falls in
    # a window +/-windowHalfWidth, then concatenates them in a new timeseries
    climls = []
    for c in ts.cell:
        climls.append( calc_clim(ts.sel(cell=c), windowHalfWidth,
                       pctile, smoothPercentile, smoothPercentileWidth,
                       Ly, tdim, skipna=skipna) )
    results =dask.compute(climls)
    # Save in dataset
    ds = xr.Dataset() 
    thresh_results = [r[0] for r in results[0]]
    ds['thresh'] = xr.concat(thresh_results, dim=ts.cell)
    ds.thresh.name = 'threshold'
    seas_results = [r[1] for r in results[0]]
    ds['seas'] = xr.concat(seas_results, dim=ts.cell)
    ds.seas.name = 'seasonal'

    # unstack cell dimension and add attributes to ds
    ds = ds.unstack('cell')
    ds = annotate_ds(ds, ds_attrs, 'clim')
    # add all parameters used to global attributes 
    params = f"""Threshold calculated using:
    {pctile} percentile;
    climatology period is {ts[0,0][tdim].dt.year.values}-{ts[-1,0][tdim].dt.year.values}'; 
    window half width used for percentile is {windowHalfWidth}"""
    if smoothPercentile:
        params = params + f";  width of moving average window to smooth percentile is {smoothPercentileWidth}"
    ds.attrs['xmhw_parameters'] = params 
    return ds


def calc_clim(ts, windowHalfWidth=5, pctile=90, smoothPercentile=True,
                smoothPercentileWidth=31, Ly=False, tdim='time', skipna=False):
    """ Calculate threshold for one cell grid at the time
    """
    twindow = window_roll(ts, windowHalfWidth, tdim)

    # rechunk twindow otherwise it is passed to dask_percentile as a numpy array 
    twindow = twindow.chunk({'z': -1})
    
     # Calculate threshold and seasonal climatology across years
    thresh_climYear = calculate_thresh(twindow, pctile, skipna)
    seas_climYear = calculate_seas(twindow, skipna) 

    if smoothPercentile:
        thresh_climYear = smooth_clim(thresh_climYear, smoothPercentileWidth, Ly)
        seas_climYear = smooth_clim(seas_climYear, smoothPercentileWidth, Ly)

    return thresh_climYear, seas_climYear


def smooth_clim(clim, smoothPercentileWidth, Ly):

    # If the length of year is < 365/366 (e.g. a 360 day year from a Climate Model)
    if Ly:
        valid = ~np.isnan(clim)
        clim.where(~valid, runavg(clim, smoothPercentileWidth))
        # >= 365-day year
    else:
        clim = runavg(clim, smoothPercentileWidth)
    return clim 


def detect(temp, th, se, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=None, 
           coldSpells=False, tdim='time', intermediate=False, anynans=False): 
    """

    Applies the Hobday et al. (2016) marine heat wave definition to an input time
    series of temperature. Outputs properties of all detected marine heat waves.

    Parameters
    ----------
    temp: xarray DataArray
        Temperature array
    th: xarray DataArray
        Seasonally varying threshold (e.g., 90th percentile)
    se: xarray DataArray
        Climatological seasonal cycle
    minDuration: int
        Minimum duration in days for acceptance detected MHWs (default 5)
    joinAcrossGaps: bool 
       Switch indicating whether to join MHWs separated by a short gap (default True)
    maxGap: int
       Maximum length of gap in days allowed for the joining of MHWs (default 2)
    maxPadLength: int
        Specifies the maximum length in days over which to interpolate NaNs in input temp time series.
        i.e., any consecutive blocks of NaNs with length greater than maxPadLength will be left as NaN.
        If None it does not interpolate (default None).
    coldSpells: bool
        Specifies if the code should detect cold events instead of heat events (default False)
    tdim: str
        Name of time dimension. (DEFAULT='time')
    intermediate: bool
        If True also output dataset with sst, climatologies and detected events along time axis (default False)
    
    Returns
    -------

    mhw: xarray Dataset
        Detected marine heat waves (MHWs). Has new 'events' dimension
    intermediate: xarray Dataset
        dataset with sst and climatologies along time axis - Optional only if intermediate is True
    """
  
   
    
    # check maxGap < minDuration 
    if maxGap >= minDuration:
        raise XmhwException("Maximum gap between mhw events should be smaller than event minimum duration")
    # if time dimension different from time, rename it
    temp = temp.rename({tdim: 'time'})
    # save original attributes in a dictionary to be assigned to final dataset
    ds_attrs = {}
    ds_attrs['ts'] = temp.attrs
    #ds_attrs[tdim+'encoding'] = temp.encoding
    for c in temp.coords:
        ds_attrs[c] = temp[c].attrs

    # return an array stacked on all dimensions excluding time
    # Land cells are removed
    # new dimensions are (time, cell)
    ts = land_check(temp, removeNans=anynans)
    del temp
    th = land_check(th, tdim='doy', removeNans=anynans)
    se = land_check(se, tdim='doy', removeNans=anynans)
    # assign doy 
    ts = add_doy(ts)

    # Linearly interpolate to replace all consecutive missing blocks of length <= maxPadLength
    if maxPadLength:
        ts = ts.interpolate_na(dim=tdim, max_gap=maxPadLength)
    # Flip temp time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Build a pandas series with the positional indexes as values
    # [0,1,2,3,4,5,6,7,8,9,10,..]
    idxarr = pd.Series(data=np.arange(len(ts.time)), index=ts.time.values)
    mhwls = []
    for c in ts.cell:
        mhwls.append(  define_events(ts.sel(cell=c), th.sel(cell=c), se.sel(cell=c), idxarr,
                     minDuration, joinAcrossGaps, maxGap, intermediate))
    results = dask.compute(mhwls)
    mhw_results = [r[0] for r in results[0]]
    mhw = xr.concat(mhw_results, dim=ts.cell)
    mhw = mhw.unstack('cell')
    if intermediate:
        inter_results = [r[1] for r in results[0]]
        mhw_inter = xr.concat(inter_results, dim=ts.cell).unstack('cell')
        mhw_inter = mhw_inter.rename({'index': 'time'})
        mhw_inter = mhw_inter.squeeze(drop=True)
    # if point dimension was added in land_check remove
    mhw = mhw.squeeze(drop=True)

    # Flip climatology and intensities in case of cold spell detection
    if coldSpells:
        mhw = flip_cold(mhw)
    
    mhw  = annotate_ds(mhw, ds_attrs, 'mhw')
    # add all parameters used to global attributes 
    params = f"""MHW detected using:
    {minDuration} days of minimum duration;
        where original timeseries had missing values interpolation was used to fill gaps;"""
    if  maxPadLength:
        params = params + f"; if gaps were more than {maxPadLength} days long, they were left as NaNs"
    if coldSpells:
        params = params + f"; cold events were detected instead of heat events"
    if joinAcrossGaps:
        params = params + f";  events separated by {maxGap} or less days were joined"
    mhw.attrs['xmhw_parameters'] = params 
    if intermediate:
        mhw_inter.squeeze(drop=True)
        return mhw, mhw_inter 
    return mhw 

