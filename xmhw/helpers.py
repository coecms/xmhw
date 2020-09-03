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
from .exception import XmhwException


def get_doy(t):
    """ Generate vector for day of the year as 366 days year
        Input: timeseries.time axis
        Return: doy - day of the year array to use as extra coordinate for timeseries
    """
    # get original dayofyear
    doy_original = t.dt.dayofyear
    # select all days from 1st of March onwards
    march_or_later = t.dt.month >= 3
    # select non leap years
    not_leap_year = ~t.dt.is_leap_year
    # add extra day if not leap year and march or later
    doy = doy_original + (not_leap_year & march_or_later)
    # rechunk and return new doy
    return doy.chunk({'time': -1})


def feb29(ts):
    """
    This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.
    """
    #return (ts.where(ts.doy.isin([59,60,61]),drop=True).mean(dim='doy').values)
    return (ts.where(ts.doy.isin([59,61]),drop=True).mean(dim='doy').values)


def runavg(ts, w):
    """Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:
      ts            Time series [1D xarray array]
      w             Integer length (must be odd) of running average window

    Return:
      ts_smooth     Smoothed time series
    """

    return ts.pad(doy=(w-1)//2, mode='wrap').rolling(doy=w, center=True).mean().dropna(dim='doy')


def window_roll(ts, w): 
    """Return all values falling in -w/+w window from each value in array"""
    
    width = 2*w+1
    trolled = ts.rolling(time=width, center=True).construct('wdim')
    return trolled.stack(z=('wdim', 'time'))#.reset_index('z')


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


def join_gaps(ds, maxGap):
    """Find gaps between mhws which are less equal to maxGap and join adjacent mhw into one event
       Input:
             ds.start - array of mhw start indexes
             ds.end - array of mhw end indexes
             maxGap - all gaps <= maxGap arre removed
    """
    # calculate gaps by subtracting index of end of each mhw from start of successive mhw
    # select as True all gaps > maxGap, these are the values we'll be keeping
    
    start = ds.start.dropna(dim='time')
    end = ds.end.dropna(dim='time')
    
    if len(start.time) >0 :
        gaps = ((start - end.shift(time=1)) > maxGap)
    # load first value so be set to True instead of NaN
        gaps.load()
        gaps[0] = True 
        
    # shift back gaps series
        gaps_shifted = gaps.shift(time=-1)
    # use "gaps" to select start indexes to keep
        start = start.where(gaps, drop=True)
    # use "gaps_shifted" to select end indexes and duration to keep
        end = end.where(gaps_shifted, drop=True)
    # reindex so we have a complete time axis
        newst = start.reindex_like(ds.start)
        newen = end.reindex_like(ds.end)
        newst.name = 'start'
        newen.name = 'end'
    else:
        return ds
    return xr.merge([newst, newen], compat='override')


def mhw_filter(exceed, minDuration, joinGaps, maxGap):
    """ Filter events of consecutive days above threshold which ar elonger then minDuration
        ts - timeseries
        exceed_bool - boolean array with True values where ts >= threshold value for same dayofyear
    """
    # exceed: [F,F,F,F,T,T,T,T,T,F,F,...]
    # Build an array with the positional indexes as values
    # [0,1,2,3,4,5,6,7,8,9,10,..]
    # this could be calculated only once for all lat/lon points 
    a = np.arange(len(exceed.time))
    arange = xr.ones_like(exceed) * xr.DataArray(a, coords=[exceed.time], dims=['time'])
    # Build another array where the the 1st index of a succession of Trues is propagated (is actually the index before a True?)
    # while False points retain their positional indexes
    # events = [0,1,2,3,3,3,3,3,3,9,10,...]
    events = (arange.where(~exceed).ffill(dim='time')).fillna(0)
    # by removing the 2nd array from the 1st we get 1/2/3/4 ... counter for each mhw and 0 elsewhere
    # events_map = [0,0,0,0,1,2,3,4,5,0,0,...]
    events_map = arange - events
    # removing the series shifted by 1 place to the right from itself we're left with only the last day of the mhw having a negative counter
    # this is also indicative of the duration of the event, the series is then shifted back one place to the left and the boundaries nan are replaced with zeros 
    # shifted = [nan,0,0,0,1,1,1,1,-5,0,0,...]
    shifted = (events_map - events_map.shift(time=1)).shift(time=-1)

    shifted.load()
    shifted[-1,:,:] = -events_map[-1,:,:]
    # select only cells where shifted is less equal to the -minDuration,
    duration = events_map.where(shifted <= -minDuration)
    # from arange select where mhw duration, this will the index of last day of mhw  
    mhw_end = arange.where( ~xr.ufuncs.isnan(duration))
    # removing duration from end index gives starting index
    mhw_start = (mhw_end - duration + 1)

    # if joinAcross Gaps call join_gaps function, this will update start, end and mappings of events
    if joinGaps:
        ds = xr.Dataset({'start': mhw_start, 'end': mhw_end}).chunk({'time':-1,'lat':1,'lon':1})
        ds = xr.map_blocks(join_gaps, ds, args=[maxGap], template=ds).compute()
        mhw_start = ds.start
        mhw_end = ds.end

    # add 1 to events so each event is represent by is starting index
    events = events + 1
    # Selected mhw will be represented by indexes where "events" has values included in mhw_star_idx list
    # and where "events_map" is not 0
    
    sel_events = events.where(events.isin(mhw_start) & (events_map != 0))
    sel_events.name = 'events'
    mhw_start.name = 'start'
    mhw_end.name = 'end'

    return  mhw_start, mhw_end, sel_events

def land_check(temp):
    """ Stack lat/lon on new dimension cell and remove for land points
        Input:
        temp - sst timeseries on 3D grid
        Return
        ts - modified timeseries with stacked lat/lon and land points removed  
    """
    ts = temp.stack(cell=('lat','lon'))
    # drop cells that have all nan values along time
    ts = ts.dropna(dim='cell',how='all')
    # if ts.cell.shape is 0 then all points are land, quit
    if ts.cell.shape == (0,):
        raise XmhwException('All points of grid are either land or NaN')
    return ts

