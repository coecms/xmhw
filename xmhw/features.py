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
import pandas as pd
import sys
import time
import dask
from .exception import XmhwException


#@dask.delayed(nout=1)
def mhw_ds(ds, ts, thresh, seas, tdim='time'):
    """Calculate and add to dataset mhw properties
       First add the necessary timeseries to dataset
       Then groupby 'cell' and map the call_mhw_features function
       The call_mhw_feature function groupby 'event' and reduce dataset on same dimension,
       while executin mhw_features where all characteristic of one mhw event are calculated
    """
    # assign event coordinate to dataset
    ds = ds.assign_coords({'event': ds.events})
    #  Would calling this new dimension 'time' regardless of tdim create issues?
    ds['event'].assign_coords({tdim: ds[tdim]})

    # get temp, climatologies values for events
    ismhw = ~np.isnan(ds.events)
    mhw_temp = ts.where(ismhw)
    mhw_seas = xr.where(ismhw, seas.sel(doy=ismhw.doy.values).values, np.nan)
    mhw_thresh = xr.where(ismhw, thresh.sel(doy=ismhw.doy.values).values, np.nan)

    # get difference between ts and seasonal average, needed to calculate onset and decline rates later
    anom = (ts - seas.sel(doy=ts.doy))
    ds['anom_plus'] = anom.shift(**{tdim: 1})
    ds['anom_minus'] = anom.shift(**{tdim: 1})
    # Adding ts, seas, thresh to dataset is only for debugging
    ds['ts'] = ts
    ds['seas'] = mhw_seas
    ds['thresh'] = mhw_thresh
    relSeas = mhw_temp - mhw_seas
    relSeas['event'] = ds.events
    relThresh = mhw_temp - mhw_thresh
    relThresh['event'] = ds.events
    relThreshNorm = (mhw_temp - mhw_thresh) / (mhw_thresh - mhw_seas)
    relThreshNorm['event'] = ds.events
    # Adding these timeseries to dataset to passthem to main mapped function 
    ds['relThresh'] = relThresh
    ds['relSeas'] = relSeas
    ds['relThreshNorm'] = relThreshNorm
    # adding this so i can use it in groupby !!
    ds['cats'] = np.floor(1. + ds.relThreshNorm)
    ds['duration_moderate'] = ds.cats.where(ds.cats == 1)
    ds['duration_strong'] = ds.cats.where(ds.cats == 2)
    ds['duration_severe'] = ds.cats.where(ds.cats == 3)
    ds['duration_extreme'] = ds.cats.where(ds.cats == 4)
    # if I remove this then I need to find a way to pass this series to onset/decline
    ds['mabs'] = mhw_temp

    #From here on work grouping by cell
    # Passing last index of timeseries to calculate rate of onset and decline
    #ds =ds.groupby('cell').map(call_mhw_features, args=[tdim, len(ds.mabs)-1])
    # unify chunks inc ase variables have different chunks along cell
    ds = ds.unify_chunks()
    dstemp = ds.groupby('cell').map(call_template)
    dstemp = dstemp.chunk({'event': -1, 'cell': 1})
    ds = ds.map_blocks(call_groupby, args=[tdim, len(ds.mabs)-1, dstemp.event.values], template=dstemp)
    return ds


def call_groupby(ds, tdim, last, allevs):
    return ds.groupby('cell').map(call_mhw_features_working, args=[tdim, last, allevs])


def call_mhw_features_working(dscell, tdim, last, allevs):
    # convert xarray dataet for 1 cell to dataframe
    dftime = dscell.to_dataframe()
    dftime['time'] = dftime.index
    # call groupby event and calculate some of the mhw properties
    df = agg_df(dftime)
    # calculate the rest of the mhw properties
    df = moreop(df, dftime.relThresh, dftime.mabs)
    # calculate onset_decline
    df = onset_decline(df, last)    
    # convert back to xarray dataset and reindex so all cells have same event axis
    ds = xr.Dataset.from_dataframe(df, sparse=False)
    all_evs = xr.DataArray(allevs, dims=['event'], coords=[allevs])
    ds  = ds.reindex_like(all_evs)
    return ds


def agg_df(df):
    """Define and aggregation dictionary to avoid apply
    """
    return df.groupby('event').agg(
            event = ('event', 'first'),
            index_start = ('start', 'nunique'),
            index_end = ('end', 'nunique'),
            time_start = ('time', 'first'),
            time_end = ('time', 'last'),
            relS_imax = ('relSeas', np.argmax),
            # the following are needed for onset_decline
            relS_first = ('relSeas', 'first'),
            relS_last = ('relSeas', 'last'),
            anom_first = ('anom_plus', 'first'),
            anom_last = ('anom_minus', 'last'),
            # till here
            intensity_max = ('relSeas', 'max'),
            intensity_mean = ('relSeas', 'mean'),
            intensity_cumulative = ('relSeas', 'sum'),
            relS_var = ('relSeas', 'var'),
            relT_var = ('relThresh', 'var'), 
            intensity_cumulative_relThresh = ('relThresh', 'sum'),
            intensity_mean_abs = ('mabs', 'mean'),
            mabs_var = ('mabs', 'var'), 
            intensity_cumulative_abs = ('mabs', 'sum'),
            max_cat = ('cats', 'max'), 
            duration_moderate = ('duration_moderate', 'sum'),
            duration_strong = ('duration_strong', 'sum'),
            duration_severe = ('duration_severe', 'sum'),
            duration_extreme = ('duration_extreme', 'sum') )
            # intensity_max can be sued as relSeas(index_peak) in onset_decline

def moreop(df, relT, mabs):
    df['index_peak'] = df.event + df.relS_imax
    df['intensity_var'] = np.sqrt(df.relS_var) 
    df['intensity_max_relThresh'] = relT.iloc[df.index_peak.values]
    df['intensity_max_abs'] = mabs.iloc[df.index_peak.values]
    df['intensity_var_relThresh'] = np.sqrt(df.relT_var) 
    df['intensity_var_abs'] = np.sqrt(df.mabs_var) 
    df['category'] = np.minimum(df.max_cat, 4)
    return df.drop(['relS_imax', 'relS_var', 'relT_var', 'max_cat', 'mabs_var'], axis=1)


def call_mhw_features(dsgroup, tdim, last, allevs):
    features = [dask.delayed(mhw_features)(v, tdim, last) for k,v in dsgroup.groupby('event')]
    results = dask.compute(features)
    dsobj = xr.concat(results[0], dim='event')
    all_evs = xr.DataArray(allevs, dims=['event'], coords=[allevs])
    dsobj  = dsobj.reindex_like(all_evs)
    return dsobj




def group_function(array, func, farg=None, dim='event'):
    """ Run function on array after groupby on event dimension """
    if farg:
        return array.groupby(dim).reduce(func, arg=farg)
    return array.groupby(dim).reduce(func)


def get_rate(relSeas_peak, relSeas_edge, period):
    """ Calculate onset/decline rate of event
    """
    return (relSeas_peak - relSeas_edge) / period


def get_edge(relS, anom, idx, edge):
    """ Return the relative start or end of mhw to calculate respectively onset and decline 
        for onset edge = 0, anom=anom.shift('time'=1), relSeas=relSeas[0]
        for decline edge = len(ts)-1, anom=anom.shift('time'= -1), relSeas=relSeas[-1]
    """
    x = relS.where(idx == edge, anom)
    return 0.5*(relS + x)


def get_period(start, end, peak, tsend):
    """ Return the onset/decline period for a mhw
    """
    esp = end - start - peak
    x = peak.where(peak == 0, 1)
    onset_period = x.where(start != 0, x + 0.5)
    y = esp.where(peak != tsend, 1)
    decline_period = y.where(end != tsend, y + 0.5)
    return onset_period, decline_period


def onset_decline(df, last):
    """ Calculate rate of onset and decline for each MHW
    """
    onset_period, decline_period = get_period(df.index_start, df.index_end, df.index_peak, last)
    relSeas_start = get_edge(df.relS_first, df.anom_first, df.index_start, 0)
    relSeas_end = get_edge(df.relS_last, df.anom_last, df.index_end, last)
    df['rate_onset'] =  get_rate(df.intensity_max, relSeas_start, onset_period)
    df['rate_decline'] =  get_rate(df.intensity_max, relSeas_end, decline_period)
    return df.drop(['anom_first', 'anom_last', 'relS_last', 'relS_first'], axis=1) 


def flip_cold(ds):
    """Flip mhw intensities if cold spell
    """
    for varname in ds.keys():
        if 'intensity' in varname and '_var' not in varname:
            ds[varname] = -1*ds[varname]
    return ds


def ds_template(ds):
    """Create dataset template for map_blocks
    """
    mhw = ds.copy()
    for var in ['index_end', 'index_start', 'index_peak', 'intensity_max',
                'intensity_mean', 'intensity_var', 'intensity_cumulative',
                'intensity_max_abs', 'intensity_max_relThresh',
                'intensity_cumulative_relThresh', 'intensity_var_relThresh',
                'intensity_cumulative_abs', 'intensity_mean_abs',
                'intensity_var_abs', 'rate_onset', 'rate_decline']:
        mhw[var] = np.nan
        mhw['category'] = np.nan
    for var in ['duration_moderate', 'duration_strong',
                'duration_severe', 'duration_extreme']:
        mhw[var] = 0
    mhw['time_start'] = mhw['time'][0]
    mhw['time_end'] = mhw['time'][-1]
    mhw = mhw.drop_vars(['start','end','anom_plus', 'anom_minus', 'seas', 'ts',
           'thresh', 'events', 'relThresh', 'relSeas', 'relThreshNorm', 'mabs'])
    return mhw.drop_dims(['time'])


def call_template(dsgroup):
    return dsgroup.groupby('event').map(ds_template)
