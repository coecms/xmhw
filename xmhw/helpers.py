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


def add_doy(ts,dim="time"):
    """ Add day of the year as 366 days year as coordinate to timeseries
        Input: 
           timeseries
           dimension name of the time axis, default "time"
        Return: array with doy - day of the year array added as extra coordinate for timeseries
    """
    # get the time axis
    t=ts[dim]
    # get original dayofyear
    doy_original = t.dt.dayofyear
    # select all days from 1st of March onwards
    march_or_later = t.dt.month >= 3
    # select non leap years
    not_leap_year = ~t.dt.is_leap_year
    # add extra day if not leap year and march or later
    doy = doy_original + (not_leap_year & march_or_later)
    # rechunk and return new doy as coordinate of the "t" input variable
    ts.coords['doy'] = doy.chunk({'time': -1})
    return ts


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


def join_gaps(start, end, maxGap, tdim='time'):
    """Find gaps between mhws which are less equal to maxGap and join adjacent mhw into one event
       Input:
             s - array of mhw start indexes
             e - array of mhw end indexes
             maxGap - all gaps <= maxGap are removed
    """
    # calculate gaps by subtracting index of end of each mhw from start of successive mhw
    # select as True all gaps > maxGap, these are the values we'll be keeping
    
    s = start.dropna(dim=tdim)
    e = end.dropna(dim=tdim)
    pairs = set(zip(s.squeeze().values,e.squeeze().values))
    
    if len(s[tdim]) >0 :
        gaps = ((s - e.shift(time=1)) > maxGap + 1)
    # load first value so be set to True instead of NaN
        gaps.load()
        gaps[0] = True 
        
    # shift back gaps series
        gaps_shifted = gaps.shift(time=-1)
    # use "gaps" to select start indexes to keep
        s = s.where(gaps, drop=True)
    # use "gaps_shifted" to select end indexes and duration to keep
        e = e.where(gaps_shifted, drop=True)
        joined = set(zip(s.squeeze().values,e.squeeze().values)) - pairs
    # reindex so we have a complete time axis
        st = s.reindex_like(start)
        en = e.reindex_like(end)
    else:
        return s, e, set()
    return st, en, joined


def mhw_filter(exceed, minDuration=5, joinGaps=True, maxGap=2):
    """ Filter events of consecutive days above threshold which are longer then minDuration
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
    shifted[-1,:] = -events_map[-1,:]
    # select only cells where shifted is less equal to the -minDuration,
    duration = events_map.where(shifted <= -minDuration)
    # from arange select where mhw duration, this will the index of last day of mhw  
    end = arange.where( ~xr.ufuncs.isnan(duration))
    # removing duration from end index gives starting index
    start = (end - duration + 1)

    # add 1 to events so each event is represented by its starting index
    events = events + 1
    # Selected mhw will be represented by indexes where "events" has values included in mhw_start_idx list
    # and where "events_map" is not 0
    sel_events = events.where(events.isin(start) & (events_map != 0))
    sel_events.name = 'events'

    # if joinAcross Gaps call join_gaps function, this will update start, end and mappings of events
    if joinGaps:
        start, end, joined = join_gaps(start, end, maxGap, tdim='time')
        for s,e in joined:
           sel_events[int(s):int(e)+1] = s

    # set this aside for the moment
    # recreate start arrays with start of events on correct time and not on end time
    #st_idx = start.dropna(dim='time').astype(int).values 
    #start2 = xr.full_like(start, np.nan)
    #start2[st_idx] = st_idx
    #starts[0,st_idx] = st_idx
    start.name = 'start'
    end.name = 'end'

    return  start, end, sel_events

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

def mhw_ds(start, end, events, ts, clim):
    """ Create xarray dataset to hold mhw properties
    """
    # create event coordinate and arrays for start/end indexes and dates
    #event = xr.ones_like(start) * np.arange(len(start.time)) +1
    #event = xr.DataArray(data=start, dims=['event'], coords=[event])
    #index_start = xr.DataArray(data=start, dims=['event'], coords=[event])
    #index_end = xr.DataArray(data=end, dims=['event'], coords=[event])
    #duration = end - start + 1 
#PA possibly I need this if end defined on wrong timestamp!! Need to check
    #date_start = ts.time.isel(time=start.values)
    #date_end = ts.time.isel(time=end.values)
    # create dataset
    mhw = xr.merge([events, start, end])
    # assign event coordinate to dataset
    mhw.assign_coords({'event': events})

    # get temp, climatologies values for events
    ismhw = ~xr.ufuncs.isnan(events)
    mhw_temp = ts.where(ismhw)#.transpose('lat','lon','time')
    #temp_mhw.coords['event'] = events
    # put doy as first dimension so it correspond to events time dimension
    #seas = clim['seas'].transpose('doy','lat','lon')
    #thresh = clim['thresh'].transpose('doy','lat','lon')
    mhw_seas = xr.where(ismhw, seas.sel(doy=ismhw.doy.values).values, np.nan)
    mhw_thresh = xr.where(ismhw, thresh.sel(doy=ismhw.doy.values).values, np.nan)
    mhw_relSeas = mhw_temp - mhw_seas
    mhw_relSeas.assign_coords({'event': events})
    mhw_relSeas['event'] = events
    mhw_relThresh = mhw_temp - mhw_thresh
    mhw_relThresh['event'] = events
    mhw_relThreshNorm = (mhw_temp - mhw_thresh) / (mhw_thresh - mhw_seas)
    mhw_relThreshNorm['event'] = events
    mhw_abs = mhw_temp
    # Find anomaly peak for events 
    #relSeas_group = mhw_relSeas.groupby('event', restore_coord_dims=True) #, squeeze=False) 
    print(group_argmax(mhw_relSeas[:,1,2]))
    sys.exit()
    #mhw['index_peak'] = relSeas_group.reduce(np.argmax,dim='time')
    mhw_relSeas = mhw_relSeas.chunk({'time':-1, 'lat':1, 'lon':1})
    mhw['index_peak'] = xr.map_blocks(group_argmax, mhw_relSeas,
                       template=mhw_relSeas).compute()

    #mhw['index_peak'] = xr.apply_ufunc(np.argmax, relSeas_group, axis=0)
    print(mhw.index_peak)
    mhw['intensity_max'] = relSeas_group.max('time') 
    print(mhw.intensity_max)
    mhw['intensity_mean'] = relSeas_group.mean('time') 
    #mhw['intensity_var'] = relSeas_group.reduce(sqrt_var) 
    mhw['intensity_cumulative'] = relSeas_group.sum('time')
    return mhw

def moreandmore():
    relThresh_group = mhw_relThresh.groupby('start') 
    print('index_peak')
    print(index_peak)
    intensity_max_relThresh = mhw_relThresh.where(index_peak).dropna('time')
    print(intensity_max_relThresh)
    print(intensity_max_relThresh[0:5,1,1].doy.values)
    intensity_mean_relThresh = relThresh_group.mean()
    #intensity_var_relThresh = relThresh_group.reduce(sqrt_var) 
    #intensity_var_relThresh = 
    intensity_cumulative_relThresh = relThresh_group.sum()
    abs_group = mhw_abs.groupby('start')
    intensity_max_abs = mhw_abs.where(index_peak).dropna('time')
    intensity_mean_abs = abs_group.mean()
    #intensity_var_abs = abs_group.reduce(sqrt_var) 
    #intensity_var_abs = intensity_mean
    intensity_cumulative_abs = abs_group.sum()

    # define categories
    categories = np.array(['Moderate', 'Strong', 'Severe', 'Extreme'])
    # Fix categories
    relThreshNorm_group = mhw_relThreshNorm.groupby('start') 
    #index_peakCat = relThreshNorm_group.argmax()
    cats = xr.ufuncs.floor(1. + mhw_relThreshNorm)
    #category = categories[ cats[index_peakCat].apply(cat_min) ]
    duration_moderate = np.sum(cats == 1.)
    duration_strong = np.sum(cats == 2.)
    duration_severe = np.sum(cats == 3.)
    duration_extreme = np.sum(cats >= 4.)

    # define stats
    
    #mhw = xr.Dataset(data_vars=
    mhw = {'date_start': date_start,
                      'date_end': date_end,
                      'duration': duration,
                      'index_start': index_start,
                      'index_end': index_end,
                      'index_peak': index_peak,
                      #'intensity_peak': intensity_peak,
                      'intensity_max': intensity_max,
                      'intensity_mean': intensity_mean,
                      'intensity_var': intensity_var,
                      'intensity_cumulative': intensity_cumulative,
                      'intensity_max_relThresh': intensity_max_relThresh,
                      'intensity_mean_relThresh': intensity_mean_relThresh,
                      #'intensity_var_relThresh': intensity_var_relThresh,
                      'intensity_cumulative_relThresh': intensity_cumulative_relThresh,
                      'intensity_max_abs': intensity_max_abs,
                      'intensity_mean_abs': intensity_mean_abs,
                      #'intensity_var_abs': intensity_var_abs,
                      'intensity_cumulative_abs': intensity_cumulative_abs,
                      #'index_peakCat' : index_peakCat,
                      #'cats': cats
                               }#)
    return mhw

