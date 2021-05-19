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


def mhw_df(df):
    """Calculate and add to dataframe mhw properties for one grid cell
       First add the necessary timeseries to dataframe
       Then groupby 'cell' and map the call_mhw_features function
       The call_mhw_feature function groupby 'event' and reduce dataset on same dimension,
       while executin mhw_features where all characteristic of one mhw event are calculated
    """

    # get temp, climatologies values for events
    ismhw = df.events.notna()

    mhw_temp = df.ts.where(ismhw)
    mhw_seas = df.seas.where(ismhw) #np.where(ismhw, df.seas, np.nan)
    mhw_thresh = df.thresh.where(ismhw)# np.where(ismhw, df.thresh, np.nan)

    # get difference between ts and seasonal average, needed to calculate onset and decline rates later
    anom = (df.ts - df.seas)
    df['anom_plus'] = anom.shift(+1)
    df['anom_minus'] = anom.shift(-1)
    # Adding ts, seas, thresh to dataset is only for debugging
    df['time'] = df.index
    df['seas'] = mhw_seas
    df['thresh'] = mhw_thresh
    t_seas = mhw_temp - mhw_seas
    t_thresh = mhw_temp - mhw_thresh
    thresh_seas = mhw_thresh - mhw_seas
    df['relSeas'] = t_seas
    df['relThresh'] = t_thresh
    df['relThreshNorm'] = t_thresh / thresh_seas
    # add severity
    df['severity'] = t_seas / -(thresh_seas)
    # adding this so i can use it in groupby !!
    df['cats'] = np.floor(1. + df.relThreshNorm)
    df['duration_moderate'] = df.cats == 1.
    df['duration_strong'] = df.cats == 2.
    df['duration_severe'] = df.cats == 3.
    df['duration_extreme'] = df.cats >= 4.
    # if I remove this then I need to find a way to pass this series to onset/decline
    df['mabs'] = mhw_temp
    return df


def mhw_features(dftime, last):
    # call groupby event and calculate some of the mhw properties
    df = agg_df(dftime)
    # calculate the rest of the mhw properties
    df = properties(df, dftime.relThresh, dftime.mabs)
    # calculate onset_decline
    df = onset_decline(df, last)    
    return df

def unique_dropna(s):
    """Apply dropna to series before calling unique so it will return only non NaN values
    """
    return s.dropna().unique()

def agg_df(df):
    """Define and aggregation dictionary to avoid apply
    """
    return df.groupby('events').agg(
            event = ('events', 'first'),
            index_start = ('start', unique_dropna),
            index_end = ('end', unique_dropna),
            time_start = ('time', 'first'),
            time_end = ('time', 'last'),
            relS_imax = ('relSeas', np.argmax),
            # adding this to save the time which is the actual timeframe index, instead of the timeseries index 
            time_peak = ('relSeas', 'idxmax'),
            # the following are needed for onset_decline
            relS_first = ('relSeas', 'first'),
            relS_last = ('relSeas', 'last'),
            anom_first = ('anom_plus', 'first'),
            anom_last = ('anom_minus', 'last'),
            # till here
            intensity_max = ('relSeas', 'max'),
            intensity_mean = ('relSeas', 'mean'),
            intensity_cumulative = ('relSeas', 'sum'),
            severity_max = ('severity', 'max'),
            severity_mean = ('severity', 'mean'),
            severity_cumulative = ('severity', 'sum'),
            severity_var = ('severity', 'var'),
            relS_var = ('relSeas', 'var'),
            relT_var = ('relThresh', 'var'), 
            intensity_mean_relThresh = ('relThresh', 'mean'),
            intensity_cumulative_relThresh = ('relThresh', 'sum'),
            intensity_mean_abs = ('mabs', 'mean'),
            mabs_var = ('mabs', 'var'), 
            intensity_cumulative_abs = ('mabs', 'sum'),
            cats_max = ('cats', 'max'), 
            duration_moderate = ('duration_moderate', 'sum'),
            duration_strong = ('duration_strong', 'sum'),
            duration_severe = ('duration_severe', 'sum'),
            duration_extreme = ('duration_extreme', 'sum') )
            # intensity_max can be used as relSeas(index_peak) in onset_decline

def properties(df, relT, mabs):
    df['index_peak'] = df.event + df.relS_imax
    df['intensity_var'] = np.sqrt(df.relS_var) 
    df['severity_var'] = np.sqrt(df.severity_var) 
    df['intensity_max_relThresh'] = relT.iloc[df.index_peak.values]
    df['intensity_max_abs'] = mabs.iloc[df.index_peak.values]
    df['intensity_var_relThresh'] = np.sqrt(df.relT_var) 
    df['intensity_var_abs'] = np.sqrt(df.mabs_var) 
    df['category'] = np.minimum(df.cats_max, 4)
    df['duration'] = df.index_end - df.index_start + 1
    return df.drop(['relS_imax', 'relS_var', 'relT_var', 'cats_max', 'mabs_var'], axis=1)


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
