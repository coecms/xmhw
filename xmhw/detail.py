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


def mhw_ds(ds, ts, thresh, seas, tdim='time'):
    """ Calculate and add to dataset mhw properties
    """
    #date_start = ts.time.isel(time=start.values)
    #date_end = ts.time.isel(time=end.values)
    # assign event coordinate to dataset
    ds = ds.assign_coords({'event': ds.events})
    #  Would calling this new dimension 'time' regardless of tdim create issues?
    #ds['event'].assign_coords({'time': ds[tdim]})
    ds['event'].assign_coords({tdim: ds[tdim]})

    # get temp, climatologies values for events
    ismhw = ~np.isnan(ds.events)
    mhw_temp = ts.where(ismhw)
    #temp_mhw.coords['event'] = events
    mhw_seas = xr.where(ismhw, seas.sel(doy=ismhw.doy.values).values, np.nan)
    mhw_thresh = xr.where(ismhw, thresh.sel(doy=ismhw.doy.values).values, np.nan)
    ds['ts'] = ts
    # get difference between ts and seasonal average, needed to calculate onset and decline rates later
    #ds['anom'] = (ts - seas.sel(doy=ts.doy)).pad({tdim:(1,2)}, constant_values=0.)
    ds['anom'] = (ts - seas.sel(doy=ts.doy))
    print(ds.anom)
    ds['seas'] = mhw_seas
    ds['thresh'] = mhw_thresh
    relSeas = mhw_temp - mhw_seas
    relSeas['event'] = ds.events
    relThresh = mhw_temp - mhw_thresh
    relThresh['event'] = ds.events
    relThreshNorm = (mhw_temp - mhw_thresh) / (mhw_thresh - mhw_seas)
    relThreshNorm['event'] = ds.events
    # this is temporary so I can test better, in theory we don't need to save these values
    ds['relThresh'] = relThresh
    ds['relSeas'] = relSeas
    ds['relThreshNorm'] = relThreshNorm
    # if I remove this then I need to find a way to pass this series to onset/decline
    ds['mabs'] = mhw_temp

    # Save start and end and duration for each event
    start = ds['start']
    end = ds['end']
    #ds = ds.drop_vars(['start', 'end']) 
    ds['end_idx'] = end.groupby('cell').map(group_function, args=[np.nanmax], dim='event')
    ds['start_idx'] = start.groupby('cell').map(group_function, args=[np.nanmax], dim='event')
    # Find anomaly peak for events 
    relSeas = relSeas.chunk({tdim:-1, 'cell':1})
    relSeas_group = relSeas.groupby('cell')  
    # this operation changes also the 'event' dimension reducing it to the actual number of events across all cells
    relSeas_argmax = relSeas_group.map(group_function, args=[np.argmax], dim='event')
    ds['index_peak'] = relSeas_argmax.event + relSeas_argmax
    ds['intensity_max'] = relSeas_group.map(group_function, args=[np.nanmax], dim='event')
    ds['intensity_mean'] = relSeas_group.map(group_function, args=[np.nanmean], dim='event') 
    var = relSeas_group.map(group_function, args=[np.var], dim='event') 
    ds['intensity_var'] = np.sqrt(var) 
    ds['intensity_cumulative'] = relSeas_group.map(group_function, args=[np.sum], dim='event')
    # stats for 
    relThresh = relThresh.chunk({tdim:-1, 'cell':1})
    relThresh_group = relThresh.groupby('cell') 
    dsdict = {'peak': ds.index_peak, 'relT': relThresh, 'mabs': ds.mabs}
    ds2 = xr.Dataset(dsdict).groupby('cell').map(get_peak, 
             args=[[k for k in dsdict.keys() if k != 'peak']])
    ds['intensity_max_relThresh'] = ds2.relT
    ds['intensity_max_abs'] = ds2.mabs
    var = relThresh_group.map(group_function, args=[np.var], dim='event') 
    ds['intensity_var_relThresh'] = np.sqrt(var) 
    ds['intensity_cumulative_relThresh'] = relThresh_group.map(group_function, args=[np.sum], dim='event')
    # abs stats
    abs_group = mhw_temp.groupby('cell')
    ds['intensity_mean_abs'] = abs_group.map(group_function, args=[np.nanmean], dim='event') 
    var = abs_group.map(group_function, args=[np.var], dim='event') 
    ds['intensity_var_abs'] = np.sqrt(var) 
    ds['intensity_cumulative'] = abs_group.map(group_function, args=[np.sum], dim='event')
    # Add categories to dataset
    ds = categories(ds, relThreshNorm)
    ds = ds.groupby('cell').map(onset_decline)
    return ds 

def categories(ds, relThreshNorm):
    # define categories
    categories = {1: 'Moderate', 2: 'Strong', 3: 'Severe', 4: 'Extreme'}
    # Fix categories
    relThreshNorm_group = relThreshNorm.groupby('cell') 
    index_peakCat = relThreshNorm_group.map(group_function, args=[np.argmax], dim='event')
    cats = np.floor(1. + relThreshNorm)
    cat_index = cats.groupby('cell').map(group_function, args=[index_cat], dim='event')
    ds['category'] = xr.zeros_like(cat_index).astype(str)
    for k,v in categories.items():
        ds['category'] = xr.where(cat_index == k, v, ds['category'])

    # calculate duration of each category
    ds['duration_moderate'] = cats.groupby('cell').map(group_function, args=[cat_duration],farg=1, dim='event')
    ds['duration_strong'] = cats.groupby('cell').map(group_function, args=[cat_duration],farg=2, dim='event')
    ds['duration_severe'] = cats.groupby('cell').map(group_function, args=[cat_duration],farg=3, dim='event')
    ds['duration_extreme'] = cats.groupby('cell').map(group_function, args=[cat_duration],farg=4, dim='event')
    # define stats
    return ds

def group_function(array, func, farg=None, dim='event'):
    """ Run function on array after groupby on event dimension """
    if farg:
        return array.groupby(dim).reduce(func, arg=farg)
    return array.groupby(dim).reduce(func)

def group_map(array, func, farg=None, dim='event'):
    """ Run function on array after groupby on event dimension """
    if farg:
        return array.groupby(dim).map(func, arg=farg)
    return array.groupby(dim).map(func)

def get_peak(ds, variables, dim='event'):
    """ Return relThresh and relThreshNorm  where index_peak
    """
    peak = ds.peak.dropna(dim=dim).load()
    ds2 = xr.Dataset()
    for v in variables:
        val = [ds[v][np.int(x)] for x in peak.values]
        ds2[v] = peak.copy()
        ds2[v][:] = val
    return ds2.reindex_like(ds)

def index_cat(array, axis):
    """ Get array maximum and return minimum between peak and 4 , minus 1
        to index category
    """
    peak = np.max(array)
    return np.min([peak, 4])

def cat_duration(array, axis, arg=1):
    """ Return sum for input category (cat)
    """
    return np.sum(array == arg) 


def get_rate(relSeas_peak, relSeas_edge, period):
    """ Calculate onset/decline rate of event
    """
    return (relSeas_peak - relSeas_edge) / period

def get_edge(relSeas, anom, idx, edge, step):
    """ Return the relative start or end of mhw to calculate respectively onset and decline 
        for onset edge = 0 step = 1, relSeas=relSeas[0]
        for decline edge = len(ts)-1 and step = -1, relSeas=relSeas[-1]
    """
    tdim = anom.dims[0]
    anomsh = anom.shift(**{tdim: step})#[idx]#+step]
    x = xr.where(idx == edge, relSeas[edge], anomsh[idx])
    return 0.5*(relSeas[idx] + x)

def get_period(start, end, peak, tsend):
    """ Return the onset/decline period for a mhw 
    """
    x = xr.where(peak == 0, 1, peak)
    onset_period = xr.where(start == 0, x, x + 0.5)
    y = xr.where(peak == tsend, 1, end - start - peak)
    decline_period = xr.where(end == tsend, y, y + 0.5)
    return onset_period, decline_period

def onset_decline(ds):
    """ Calculate rate of onset and decline for each MHW
    """
    start = ds.start_idx.dropna(dim='event').astype(int)
    if len(start) == 0:
        ds['rate_onset'] = ds.start_idx
        ds['rate_decline'] = ds.end_idx
        return ds

    end = ds.end_idx.dropna(dim='event').astype(int)
    peak = ds.index_peak.where(start).dropna(dim='event')
    tslen = len(ds.anom)
    onset_period, decline_period = get_period(start, end, peak, tslen)
    # create numpy array with first an last value of relSeas as first and last and ts-seas in between
    relSeas_start = get_edge(ds.relSeas,ds.anom, start, 0, 1) 
    relSeas_end = get_edge(ds.relSeas, ds.anom, end, tslen-1, -1) 
    relSeas_peak = ds.relSeas[peak.astype(int)]
    onset_rate =  get_rate(relSeas_peak, relSeas_start, onset_period)
    decline_rate =  get_rate(relSeas_peak, relSeas_end, decline_period)
    ds['rate_onset'] = onset_rate.reindex_like(ds.start_idx)
    ds['rate_decline'] = decline_rate.reindex_like(ds.end_idx)
    return ds
