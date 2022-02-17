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


import numpy as np
import dask
from .exception import XmhwException


def mhw_df(df):
    """Prepare dataframe before aggregating series in mhw_features().

    Parameters
    ----------
    df: pandas Dataframe
        Includes series for: events, start, end, ts, thresh, seas, bthresh

    Returns
    -------
    df: pandas Dataframe
        Ready to be passed to mhw_features, with added categories,
        severity and relative norms  
    """

    # get temp, climatologies values for events only
    ismhw = df.events.notna()
    mhw_temp = df.ts.where(ismhw)
    mhw_seas = df.seas.where(ismhw)
    mhw_thresh = df.thresh.where(ismhw)

    # difference ts and seas, needed to calculate onset and decline rates
    anom = (df.ts - df.seas)
    df['anom_plus'] = anom.shift(+1)
    df['anom_minus'] = anom.shift(-1)
    # Adding ts, seas, thresh to dataframe so intermediate results and
    # climatologies can be saved together
    df['time'] = df.index
    df['seas'] = mhw_seas
    df['thresh'] = mhw_thresh
    t_seas = mhw_temp - mhw_seas
    t_thresh = mhw_temp - mhw_thresh
    thresh_seas = mhw_thresh - mhw_seas
    df['relSeas'] = t_seas
    df['relThresh'] = t_thresh
    df['relThreshNorm'] = t_thresh / thresh_seas
    # calculate severity
    df['severity'] = t_seas / -(thresh_seas)
    # calculate categories
    df['severity'] = t_seas / -(thresh_seas)
    df['cats'] = np.floor(1. + df.relThreshNorm)
    df['duration_moderate'] = df.cats == 1.
    df['duration_strong'] = df.cats == 2.
    df['duration_severe'] = df.cats == 3.
    df['duration_extreme'] = df.cats >= 4.
    # also needed to calculate onset decline rates
    df['mabs'] = mhw_temp
    return df


def mhw_features(dftime, last):
    """Calculate mhw properties, grouping by each event.
    
    Parameters
    ----------
    dftime: pandas Dataframe
        Includes series for: events, start, end, ts, thresh, seas, bthresh
    last: int
        Index of last element of series to calculate rates

    Returns
    -------
    dfmhw: pandas Dataframe
        Includes MHW characteristics along time index
    """

    # calculate some of the mhw properties aggregating by events
    df = agg_df(dftime)
    # calculate the rest of the mhw properties
    df = properties(df, dftime.relThresh, dftime.mabs)
    # calculate onset decline rates
    df = onset_decline(df, last)    
    return df


def agg_df(df):
    """Groupby events and apply different functions depending on attribute.

    Parameters
    ----------
    dftime: pandas Dataframe
        Includes MHW characteristics along time index
    last: int
        Index of last element of series to calculate rates

    Returns
    -------
    dfout: pandas Dataframe
        Includes most MHW properties by events
    """

    # using an aggregation dictionary to avoid apply.
    dfout = df.groupby('events').agg(
            event = ('events', 'first'),
            index_start = ('start', 'first'),
            index_end = ('end', 'first'),
            time_start = ('time', 'first'),
            time_end = ('time', 'last'),
            relS_imax = ('relSeas', np.argmax),
            # time as dataframe index, instead
            # of the timeseries index 
            time_peak = ('relSeas', 'idxmax'),
            # the following are needed for onset_decline
            # anom_plus is (sst -seas) shifted 1 day ahead 
            # anom_minus is (sst -seas) shifted 1 day back 
            relS_first = ('relSeas', 'first'),
            relS_last = ('relSeas', 'last'),
            anom_first = ('anom_plus', 'first'),
            anom_last = ('anom_minus', 'last'),
            # intensity_max can be used as relSeas(index_peak)
            # in onset_decline
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
    return dfout


def properties(df, relT, mabs):
    """Calculate the rest of MHW properties that cannot be returned
    directly by the groupby aggregations.

    Parameters
    ----------
    df: pandas Dataframe
        Includes most MHW properties by events
    relT: pandas Series
        Difference between ts and threshold where there is an event
    mabs: pandas Series 
        Absolute value of temperature along time index

    Returns
    -------
    df: pandas Dataframe
        As input but with more MHW properties added 
    """

    df['index_peak'] = df.event + df.relS_imax
    df['intensity_var'] = np.sqrt(df.relS_var) 
    df['severity_var'] = np.sqrt(df.severity_var) 
    df['intensity_max_relThresh'] = relT[df.time_peak].values
    df['intensity_max_abs'] = mabs[df.time_peak].values
    df['intensity_var_relThresh'] = np.sqrt(df.relT_var) 
    df['intensity_var_abs'] = np.sqrt(df.mabs_var) 
    df['category'] = np.minimum(df.cats_max, 4)
    df['duration'] = df.index_end - df.index_start + 1
    df = df.drop(['relS_imax', 'relS_var', 'relT_var', 'cats_max',
                  'mabs_var'], axis=1)
    return df


def get_rate(relSeas_peak, relSeas_edge, period):
    """ Calculate onset/decline rate of event
    """
    return (relSeas_peak - relSeas_edge) / period


def get_edge(relS, anom, idx, edge):
    """Returns the relative start or end of MHWs 

    Parameters
    ----------
    relS: pandas Series
        Difference between ts and seas where there is an MHW event
        Onset -> relS[0],  decline -> relS[-1]
    anom: pandas Series
        Difference between ts and seas along time axis 
        Shifted by +1/-1 along time axis for onset/decline
    idx: int
        Edge index is 0 for onset and lst index of ts for decline

    Returns
    -------
    mhw_edge: pandas Series
        Value of onset/decline edge
    """
    x = relS.where(idx == edge, anom)
    mhw_edge = 0.5*(relS + x)
    return mhw_edge 


def get_period(start, end, peak, tsend):
    """Return the onset/decline period for a mhw.

    For onset if event starts on 1st day of timeseries, then:
       if peak on 1 st day of event, onset period -> 1 day
       else -> period=peak.
    In any other case period = peak + 0.5

    For decline if event ends on last day of timeseries, then:
       if peak on last day of event, onset period -> 1 day 
       else -> period=(end - start - peak).
    In any other case period = (end - start -peak) + 0.5
        
    Parameters
    ----------
    start: pandas Series
        Index of start of event along time axis
    end: pandas Series
        Index of end of event along time axis
    peak: pandas Series
        Index of peak of event respect the event itself
        -> index_peak - index_start
    tsend: int
        Index of last element of series

    Returns
    -------
    onset_period: pandas Series
        Period of onset of MHWs
    decline_period: pandas Series
        Period of decline of MHWs
    """

    esp = end - start - peak
    x = peak.where(peak != 0, 1)
    onset_period = x.where(start == 0, x + 0.5)
    y = esp.where(peak != tsend, 1)
    decline_period = y.where(end == tsend, y + 0.5)
    return onset_period, decline_period


def onset_decline(df, last):
    """Calculate rate of onset and decline for each MHW.

    Parameters
    ----------
    df: pandas Dataframe
        Includes most MHW properties by events
    last: int
        Index of last element of series to calculate rates

    Returns
    -------
    df: pandas Dataframe
        As input but with rates of onset an decline added 
    """

    # calculate first the onset/decline period and edge 
    rel_index_peak = df.index_peak - df.index_start
    onset_period, decline_period = get_period(df.index_start,
                                   df.index_end, rel_index_peak, last)
    relSeas_start = get_edge(df.relS_first, df.anom_first, 
                             df.index_start, 0)
    relSeas_end = get_edge(df.relS_last, df.anom_last, df.index_end,
                           last)
    # calculate rates  
    df['rate_onset'] =  get_rate(df.intensity_max, relSeas_start,
                                 onset_period)
    df['rate_decline'] =  get_rate(df.intensity_max, relSeas_end,
                                 decline_period)
    df = df.drop(['anom_first', 'anom_last', 'relS_last',
                  'relS_first'], axis=1) 
    return df


def flip_cold(ds):
    """Flip mhw intensities if cold spell.

    Parameters
    ----------
    ds: xarray Dataset
        Includes MHW properties

    Returns
    -------
    ds: xarray Dataset
        Includes MHW properties with flipped values for intensities
    """

    for varname in ds.keys():
        if 'intensity' in varname and '_var' not in varname:
            ds[varname] = -1*ds[varname]
    return ds
