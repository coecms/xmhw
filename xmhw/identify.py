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
import sys
from .exception import XmhwException


def add_doy(ts, tdim="time"):
    """ Add day of the year as 366 days year as coordinate to timeseries
        Input: 
           timeseries
           dimension name of the time axis, default "time"
        Return: array with doy - day of the year array added as extra coordinate for timeseries
    """
    # get the time axis
    t = ts[tdim]
    # get original dayofyear
    doy_original = t.dt.dayofyear
    # select all days from 1st of March onwards
    march_or_later = t.dt.month >= 3
    # select non leap years
    not_leap_year = ~t.dt.is_leap_year
    # add extra day if not leap year and march or later
    doy = doy_original + (not_leap_year & march_or_later)
    # rechunk and return new doy as coordinate of the "t" input variable
    ts.coords['doy'] = doy.chunk({tdim: -1})
    return ts


def feb29(ts, dim='doy'):
    """
    This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.
    """
    #return (ts.where(ts.doy.isin([59,60,61]),drop=True).mean(dim=dim).values)
    return (ts.where(ts.doy.isin([59,61]),drop=True).mean(dim=dim).values)


def runavg(ts, w):
    """Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:
      ts            Time series [1D xarray array]
      w             Integer length (must be odd) of running average window

    Return:
      ts_smooth     Smoothed time series
    """
    if w%2 == 0:
        raise XmhwException("Running average window should be odd")
    return ts.pad(doy=(w-1)//2, mode='wrap').rolling(doy=w, center=True).mean().dropna(dim='doy')


def window_roll(ts, w, tdim): 
    """Return all values falling in -w/+w window from each value in array"""
    
    width = 2*w+1
    dtime = {tdim: width}
    trolled = ts.rolling(**dtime, center=True).construct('wdim')
    return trolled.stack(z=('wdim', tdim))#.reset_index('z')


def dask_percentile(array, axis, q):
    """ Use dask to calculate percentile
    """
    array = array.rechunk({axis: -1})
    #array = array.chunk({axis: -1})
    return array.map_blocks(
        np.nanpercentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)


def join_gaps(ds, maxGap, tdim='time'):
    """Find gaps between mhws which are less equal to maxGap and join adjacent mhw into one event
       Input:
             ds.start - array of mhw start indexes
             ds.end - array of mhw end indexes
             maxGap - all gaps <= maxGap are removed
    """
    # calculate gaps by subtracting index of end of each mhw from start of successive mhw
    # select as True all gaps > maxGap, these are the values we'll be keeping
    s = ds.start.dropna(dim=tdim)
    e = ds.end.dropna(dim=tdim)
    
    if len(s[tdim]) > 1 :
        pairs = set(zip(s.squeeze().values,e.squeeze().values))
        tplus1 = {tdim: 1}
        tminus1 = {tdim: -1}
        eshift = e.shift(**tplus1).load()
        # by setting first value to -(maxGap+1) then gaps[0] will always be True
        # in this way we avoid a comparison with Nan and retain first start
        eshift[0] = -(maxGap + 1)
        gaps = ((s - eshift) > maxGap + 1)
        
    # shift back gaps series
        gaps_shifted = gaps.shift(**tminus1)
    # use "gaps" to select start indexes to keep
        s = s.where(gaps, drop=True)
    # use "gaps_shifted" to select end indexes and duration to keep
        e = e.where(gaps_shifted, drop=True)
        joined = set()
        if len(s) < len(ds.start):
            if len(s) > 1:
                joined = set(zip(s.squeeze().values,e.squeeze().values)) - pairs
            # if only one element can't use zip 
            elif s.shape == (1,):
                joined = set([(s.values[0],e.values[0])]) - pairs
            # need to add this to make it work with pytest
            elif s.shape == (1,1):
                joined = set([(s[0].values[0], e[0].values[0])]) - pairs
    # reindex so we have a complete time axis
        ds['start'] = s.reindex_like(ds.start)
        ds['end'] = e.reindex_like(ds.end)
    # update events which were joined
        ds['events'] = join_events(ds.events, joined)
    else:
        return ds
    return ds


def mhw_filter(exceed, minDuration=5, joinGaps=True, maxGap=2, tdim='time'):
    """ Filter events of consecutive days above threshold which are longer then minDuration
        ts - timeseries
        exceed_bool - boolean array with True values where ts >= threshold value for same dayofyear
    """
    # exceed: [F,F,F,F,T,T,T,T,T,F,F,...]
    # Build an array with the positional indexes as values
    # [0,1,2,3,4,5,6,7,8,9,10,..]
    # this could be calculated only once for all lat/lon points 
    a = np.arange(len(exceed[tdim]))
    arange = xr.ones_like(exceed) * xr.DataArray(a, coords=[exceed[tdim]], dims=[tdim])
    # Build another array where the the 1st index of a succession of Trues is propagated (is actually the index before a True?)
    # while False points retain their positional indexes
    # events = [0,1,2,3,3,3,3,3,3,9,10,...]
    events = (arange.where(~exceed).ffill(dim=tdim)).fillna(0)
    # by removing the 2nd array from the 1st we get 1/2/3/4 ... counter for each mhw and 0 elsewhere
    # events_map = [0,0,0,0,1,2,3,4,5,0,0,...]
    events_map = arange - events
    # removing the series shifted by 1 place to the right from itself we're left with only the last day of the mhw having a negative counter
    # this is also indicative of the duration of the event, the series is then shifted back one place to the left and the boundaries nan are replaced with zeros 
    # shifted = [nan,0,0,0,1,1,1,1,-5,0,0,...]
    tplus1 = {tdim: 1}
    tminus1 = {tdim: -1}
    shifted = (events_map - events_map.shift(**tplus1)).shift(**tminus1)

    shifted.load()
    shifted[-1,:] = -events_map[-1,:]
    # select only cells where shifted is less equal to the -minDuration,
    duration = events_map.where(shifted <= -minDuration)
    # from arange select where mhw duration, this will the index of last day of mhw  
    end = arange.where( ~np.isnan(duration))
    # removing duration from end index gives starting index
    start = (end - duration + 1)

    # add 1 to events so each event is represented by its starting index
    events = events + 1
    # Selected mhw will be represented by indexes where "events" has values included in mhw_start_idx list
    # and where "events_map" is not 0
    sel_events = events.where(events.isin(start) & (events_map != 0))
    sel_events.name = 'events'

    # if joinAcross Gaps call join_gaps function, this will update start, end and mappings of events
    ds = xr.Dataset({'start': start, 'end': end, 'events': sel_events}).chunk({tdim:-1,'cell':1})
    if joinGaps:
        ds = ds.groupby('cell').map(join_gaps, args=[maxGap], tdim=tdim)
    # transpose dataset so order of coordinates is the same as other arrays
        ds = ds.transpose(tdim, 'cell')

    # set this aside for the moment
    # recreate start arrays with start of events on correct time and not on end time
    #st_idx = start.dropna(dim=tdim).astype(int).values 
    #start2 = xr.full_like(start, np.nan)
    #start2[st_idx] = st_idx
    #starts[0,st_idx] = st_idx
    return  ds


def land_check(temp, tdim='time'):
    """ Stack lat/lon on new dimension cell and remove for land points
        Input:
        temp - sst timeseries on 3D grid
        Return
        ts - modified timeseries with stacked lat/lon and land points removed  
    """
    dims = list(temp.dims)
    # add an extra fake dimensions if array 1-dimensional so a 'cell' dimension can still be created
    dims.remove(tdim)
    if len(dims) == 0:
        temp = temp.expand_dims({'point': [0.]})
        dims = ['point']
    ts = temp.stack(cell=(dims))
    # drop cells that have all nan values along time
    ts = ts.dropna(dim='cell',how='all')
    # if ts.cell.shape is 0 then all points are land, quit
    if ts.cell.shape == (0,):
        raise XmhwException('All points of grid are either land or NaN')
    return ts


def join_events(events, joined):
    """ Set right value for joined events """
    if len(joined) > 0:
        events.load()
        for s,e in joined:
            events[int(s):int(e)+1] = s
    return events
