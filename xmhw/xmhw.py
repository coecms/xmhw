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
from .identify import (land_check, add_doy, get_calendar, define_events,
                       runavg, window_roll, calculate_thresh,
                       calculate_seas, annotate_ds)
from .features import flip_cold
from .exception import XmhwException


def threshold(temp, tdim='time', climatologyPeriod=[None,None], pctile=90,
              windowHalfWidth=5, smoothPercentile=True,
              smoothPercentileWidth=31, maxPadLength=None, coldSpells=False,
              tstep=False, anynans=False, skipna=False):
    """Calculate threshold and mean climatology (day-of-year).

    Parameters
    ----------
    temp: xarray DataArray
        Temperature timeseries array
    tdim: str, optional
        Name of time dimension (default 'time')
    climatologyPeriod: list(int), optional
        Period over which climatology is calculated, specified as list
        of start and end years. Default is to use the full time series.
    pctile: int, optional 
        Threshold percentile used to detect events (default=90)
    windowHalfWidth: int, optional
        Half width of window about day-of-year used for the pooling of
        values and calculation of threshold percentile (default=5)
    smoothPercentile: bool, optional 
        If True smooth the threshold percentile timeseries with a
        moving average (default is True)
    smoothPercentileWidth: int, optional
        Width of moving average window for smoothing threshold in days,
        has to be odd number (default=31)
    maxPadLength: int, optional
        Specifies the maximum length (days) over which to interpolate
        NaNs in input temp time series. i.e., any consecutive blocks of
        NaNs with length greater than maxPadLength will be left as
        NaN. If None it does not interpolate (default is None).
    coldSpells: bool, optional
        If True the code detects cold events instead of heat events
        (default is False)
    tstep: bool, optional
        If True the timeseries timestep is used as base for 'doy' unit 
        To use with any but 365/366 days year daily files
        (default is False)
    anynans: bool, optional
        Defines in land_check which cells will be dropped, if False
        only ones with all NaNs values, if True all cells with even
        1 NaN along time dimension will be dropped (default is False)
    skipna: bool, optional 
        If True percentile and mean function will use skipna=True.
        Using skipna option is much slower (default is False)

    Returns
    -------
    clim : xarray Dataset
        includes thresh climatological threshold 
                 seas   climatological mean
    """

    # Check smooth percentile window width is odd
    # and that time dimension (tdim) is present
    if smoothPercentileWidth%2 == 0:
        raise XmhwException("smoothPercentileWidth should be odd")
    if tdim not in temp.dims: 
        raise XmhwException(f"{tdim} dimension not present, default" +
                    "is 'time' or pass as tdim='time_dimension_name'")

    # Set climatology period, if unset use full range of available data
    if all(climatologyPeriod):
        tslice = {tdim: slice(f'{climatologyPeriod[0]}-01-01',
                              f'{climatologyPeriod[1]}-12-31')}
        temp = temp.sel(**tslice)
    # Save original attributes in dictionary to assign to final dataset
    ds_attrs = {}
    ds_attrs['ts'] = temp.attrs
    # ds_attrs[tdim+'encoding'] = temp.encoding
    for c in temp.dims:
        ds_attrs[c] = temp[c].attrs
    # Returns an array stacked on all dimensions excluded time
    # Land cells are removed and new dimensions are (time,cell)
    ts = land_check(temp, tdim=tdim, anynans=anynans)

    # check if the calendar attribute is present in time dimension
    # if not try to guess length of year
    year_days = get_calendar(ts[tdim])
    if year_days == 360.0:
        tstep = True
    ts = add_doy(ts, tdim=tdim, keep_tstep=tstep)
    #else:
    #    XMHW.Exception("Module is not yet set to work with a calendar "
    #        + "different from gregorian, standard, proleptic_gregorian."
    #        + "NB We treat all these calendars in the same way in the "
    #        + "assumption that the timeseries starts after 1582")

    # Flip ts time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Linear interpolation of all consecutive missing blocks
    # of length <= maxPadLength
    # NB by default maxPadLength is None and there is no interpolation
    if maxPadLength:
        ts = ts.interpolate_na(dim=tdim, max_gap=maxPadLength)

    # Loop over each cell to calculate climatologies, main functions
    # are delayed, so loop is automatically run in parallel
    climls = []
    for c in ts.cell:
        climls.append( calc_clim(ts.sel(cell=c), tdim, pctile, 
                       windowHalfWidth, smoothPercentile,
                       smoothPercentileWidth, tstep, skipna) )
    results =dask.compute(climls)

    # Concatenate results and save as dataset
    ds = xr.Dataset() 
    thresh_results = [r[0] for r in results[0]]
    ds['thresh'] = xr.concat(thresh_results, dim=ts.cell)
    ds.thresh.name = 'threshold'
    seas_results = [r[1] for r in results[0]]
    ds['seas'] = xr.concat(seas_results, dim=ts.cell)
    ds.seas.name = 'seasonal'
    ds = ds.unstack('cell')

    # add previously saved attributes to ds
    ds = annotate_ds(ds, ds_attrs, 'clim')
    # add all parameters used to global attributes 
    dum = [ts[0,0][tdim].dt.year.values, ts[-1,0][tdim].dt.year.values] 
    params = f"""Threshold calculated using:
    {pctile} percentile;
    climatology period is {dum[0]}-{dum[0]}'; 
    window half width used for percentile is {windowHalfWidth}"""
    if skipna:
        params = params + f"""; NaNs where skipped in percentile and mean
        calculations"""
    if smoothPercentile:
        params = params + f"""; width of moving average window to 
                 smooth percentile is {smoothPercentileWidth}"""
    if anynans:
        params = params + f"""; any grid point with even only 1 NaN along time
        axis has been removed from calculation"""
    ds.attrs['xmhw_parameters'] = params 
    return ds


def calc_clim(ts, tdim, pctile, windowHalfWidth, smoothPercentile,
              smoothPercentileWidth, tstep, skipna):
    """Calculate climatologies.

    Parameters
    ----------
    ts: xarray DataArray
        Temperature timeseries array
    tdim: str
        Name of time dimension
    pctile: int
        Threshold percentile used to detect events
    windowHalfWidth: int
        Half width of window about day-of-year used for the pooling of
        values and calculation of threshold percentile
    smoothPercentile: bool
        If True smooth the threshold percentile timeseries with a
        moving average
    smoothPercentileWidth: int
        Width of moving average window for smoothing threshold in days,
        has to be odd number
    tstep: bool
        If True the timeseries timestep is used as base for 'doy' unit 
        To use with any but 365/366 days year daily files
    skipna: bool
        If True percentile and mean function will use skipna=True.
        Using skipna option is much slower

    Returns
    -------
    thresh_climYear: xarray DataArray
        Climatological threshold for the grid cell
    seas_climYear: xarray DataArray
        Climatological mean for the grid cell 
    """

    twindow = window_roll(ts, windowHalfWidth, tdim)
    # Rechunk twindow so all timeseries is in 1 chunk 
    twindow = twindow.chunk({'z': -1})
    
    # Calculate threshold and seasonal climatology across years
    thresh_climYear = calculate_thresh(twindow, pctile, skipna, tstep)
    seas_climYear = calculate_seas(twindow, skipna, tstep) 

    # If smooth option on smooth both climatologies
    if smoothPercentile:
        thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
        seas_climYear = runavg(seas_climYear, smoothPercentileWidth)

    return thresh_climYear, seas_climYear


def detect(temp, th, se, tdim='time', minDuration=5, joinGaps=True,
           maxGap=2, maxPadLength=None, coldSpells=False,
           intermediate=False, anynans=False): 
    """Applies the Hobday et al. (2016) marine heat wave definition to
    a temperature timeseries. Returns properties of all detected MHWs.

    Parameters
    ----------
    temp: xarray DataArray
        Temperature timeseries array
    th: xarray DataArray
        Climatological threshold (e.g., 90th percentile)
    se: xarray DataArray
        Climatological mean
    tdim: str, optional
        Name of time dimension (default='time')
    minDuration: int, optional
        Minimum duration (days) to accept detected MHWs (default=5)
    joinGaps: bool, optional
       If True join MHWs separated by a short gap (default is True)
    maxGap: int, optional
        Maximum limit of gap length (days) between events (default=2)
    maxPadLength: int, optional
        Specifies the maximum length (days) over which to interpolate
        NaNs in input temp time series. i.e., any consecutive blocks of
        NaNs with length greater than maxPadLength will be left as
        NaN. If None it does not interpolate (default is None).
    coldSpells: bool, optional
        If True the code detects cold events instead of heat events
        (default is False)
    intermediate: bool, optional
        If True return also dataset with input data, detected events
        and some events properties along time axis (default is False)
    anynans: bool, optional
        Defines in land_check which cells will be dropped, if False
        only ones with all NaNs values, if True all cells with even
        1 NaN along time dimension will be dropped (default is False)
    
    Returns
    -------
    mhw: xarray Dataset
        Detected marine heat waves (MHWs). Has new 'events' dimension
    mhw_inter: xarray Dataset, optional
        Dataset with input data, detected events and some events
        properties along time axis. If intermediate is False is None
    """
    
    # check maxGap < minDuration 
    if maxGap >= minDuration:
        raise XmhwException("Maximum gap between mhw events should" +
                            " be smaller than event minimum duration")
    # if time dimension different from time, rename it
    temp = temp.rename({tdim: 'time'})
    # save original attributes in a dictionary to assign to final dataset
    ds_attrs = {}
    ds_attrs['ts'] = temp.attrs
    #ds_attrs[tdim+'encoding'] = temp.encoding
    for c in temp.coords:
        ds_attrs[c] = temp[c].attrs

    # Returns an array stacked on all dimensions excluded time, doy
    # Land cells are removed and new dimensions are (time,cell)
    ts = land_check(temp, anynans=anynans)
    del temp
    th = land_check(th, tdim='doy', anynans=anynans)
    se = land_check(se, tdim='doy', anynans=anynans)
    # assign doy 
    ts = add_doy(ts, tdim=tdim, keep_tstep=tstep)

    # Linear interpolation of all consecutive missing blocks
    # of length <= maxPadLength
    # NB by default maxPadLength is None and there is no interpolation
    if maxPadLength:
        ts = ts.interpolate_na(dim=tdim, max_gap=maxPadLength)
    # Flip temp time series if detecting cold spells
    if coldSpells:
        ts = -1.*ts

    # Build a pandas series with the positional indexes as values
    # [0,1,2,3,4,5,6,7,8,9,10,..]
    idxarr = pd.Series(data=np.arange(len(ts.time)), index=ts.time.values)

    # Loop over each cell to detect MHW events, define_events()
    # is delayed, so loop is automatically run in parallel
    mhwls = []
    for c in ts.cell:
        mhwls.append(define_events(ts.sel(cell=c), th.sel(cell=c),
                     se.sel(cell=c), idxarr, minDuration,
                     joinGaps, maxGap, intermediate))
    results = dask.compute(mhwls)

    # Concatenate results and save as dataset
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
    
    # add previously saved attributes to ds
    mhw  = annotate_ds(mhw, ds_attrs, 'mhw')
    # add all parameters used to global attributes 
    params = f"MHW detected using: {minDuration} days of minimum duration"
    if joinGaps:
        params = (params + 
             f"; events separated by {maxGap} or less days were joined")
    if coldSpells:
        params = (params + 
             f"; cold events were detected instead of heat events")
    if maxPadLength:
        params = params + f"""; where original timeseries had missing values
        interpolation was used to fill them. Gaps > {maxPadLength} days
         long were left as NaNs;"""
    if anynans:
        params = params + f"""; any grid point with even only 1 NaN along time
        axis has been removed from calculation"""
    mhw.attrs['xmhw_parameters'] = params 
    if intermediate:
        return mhw, mhw_inter 
    return mhw

