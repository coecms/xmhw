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
import dask
from datetime import date
from .exception import XmhwException
from .features import mhw_df, mhw_features


def add_doy(ts, tdim="time"):
    """Add coordinate 'doy' day of the year for a 366 days year.

    Parameters
    ----------
    ts: xarray DataArray
        Timeseries array
    tdim: str, optional
        Name of time dimension (default='time')

    Returns
    -------
    ts: xarray DataArray
        Timeseries array with extra 'doy' coordinate
    """

    # get original dayofyear
    # create filters: from 1st of March onwards and non leap years
    # add extra day if not leap year and march or later
    t = ts[tdim]
    doy_original = t.dt.dayofyear
    march_or_later = t.dt.month >= 3
    not_leap_year = ~t.dt.is_leap_year
    doy = doy_original + (not_leap_year & march_or_later)
    # rechunk and return new doy as coordinate of the "ts" input variable
    ts.coords['doy'] = doy.chunk({tdim: -1})
    return ts


def get_calendar(time):
    """Retrieve calendar information and return number of days in year.

    NB still working on ...

    Parameters
    ----------
    time: xarray DataArray
        Time dimension array

    Returns
    -------
    ndays_year: float
        Number of days in a year of timeseries
    """
    # define a dictionary mapping calendar to ndays_year
    # check if calendar is part of the time dimension attributes
    # my assumptions here are Julian can be ignored at best from 1901 onwards we could add 13 days and consider it gregorian
    # gregorian, standard, proleptic_gregorian are all the same ,as differences happens in the distant past
    # for these we ant to use add_doy
    # for 360/ 365 /366 we need different approach, they all can stay as they are but I should then use the original day ofyear admititng this is calculated differently and consistently each time
    ndays = {'standard': 365.25, 'gregorian': 365.25,
             'proleptic_gregorian': 365.25, 'all_leap': 366,
             'noleap': 365, '365_day': 365, '360_day': 360,
             'julian': 365.25}
    if 'calendar' in time.encoding.keys():
        calendar = time.encoding['calendar']
    elif 'calendar' in time.attrs.keys():
        calendar = time.attrs['calendar']
    else:
        calendar = getattr(time.values[0], 'calendar', '')
    if calendar == '':
        #calendar = infer_calendar(tdim)
        pass
    # if calendar was retrieved by variable attributes is possible it
    # was wrongly defined 
    if calendar in ['360', '365', '366']:
        calendar = f'{calendar}_day'
    elif calendar == 'leap':
        calendar = 'standard'
    if calendar not in ndays.keys():
        print('calendar not in keys')
        ndays_year = 365.25 # just to return something valid now
    else:
        ndays_year = ndays[calendar]
    return ndays_year


def feb29(ts, dim='doy'):
    """Calculate values for 29Feb by averaging days 28,29 Feb and 1 Mar
    
    Original code ignores values for 29 Feb, and uses only 28 Feb and
    1 Mar values. To replicate comment/uncomment return options.

    Returns
    -------
        Interpolated values for Feb29
    """
    return (ts.where(ts.doy.isin([59,60,61]),drop=True).mean(
            dim=dim, skipna=True).values)
    #return (ts.where(ts.doy.isin([59,61]),drop=True).mean(
    #         dim=dim, skipna=True).values)


@dask.delayed(nout=1)
def runavg(ts, w):
    """Performs a running average of a time series using a uniform
    window of width w.

    This function assumes that the input time series is periodic.

    Parameters
    ----------
    ts: xarray DataArray
        Temperature timeseries array
    w: int
        Width of window, it should be odd

    Returns
    -------
    ts_avg: xarray DataArray
        Averaged timeseries
    """
    if w%2 == 0:
        raise XmhwException("Running average window should be odd")
    ts_avg = (ts.pad(doy=(w-1)//2, mode='wrap')
                .rolling(doy=w, center=True)
                .mean().dropna(dim='doy'))
    return ts_avg


def window_roll(ts, w, tdim): 
    """Return all values falling in -w/+w window from each day-of-year
    and build new timeseries.

    Parameters
    ----------
    ts: xarray DataArray
        Temperature timeseries array
    w: int
        Half width of window
    tdim: str
        Name of time dimension

    Returns
    -------
    twindow: xarray DataArray
        Stacked array timeseries with new 'z' dimension representing
        a window of width 2*w+1
    """
    
    width = 2*w+1
    dtime = {tdim: width}
    trolled = ts.rolling(**dtime, center=True).construct('wdim')
    troll = trolled.stack(z=('wdim', tdim))
    twindow = troll.dropna(dim='z')
    return twindow


@dask.delayed(nout=1)
def calculate_thresh(twindow, pctile, skipna):
    """Calculate threshold for one cell grid at the time

    Parameters
    ----------
    twindow: xarray DataArray
        Stacked array timeseries with new 'z' dimension representing
        a window of width 2*w+1
    pctile: int
        Threshold percentile used to detect events
    skipna: bool 
        If True percentile and mean function will use skipna=True.
        Using skipna option is much slower

    Returns
    -------
    thresh_climYear: xarray DataArray
        Climatological threshold
    """

    thresh_climYear = (twindow
                       .groupby('doy')
                       .quantile(pctile/100., dim='z', skipna=skipna))
    # calculate value for 29 Feb from mean of 28-29 feb and 1 Mar
    thresh_climYear = thresh_climYear.where(thresh_climYear.doy!=60, 
                                            feb29(thresh_climYear))
    thresh_climYear = thresh_climYear.chunk({'doy': -1})
    return thresh_climYear


@dask.delayed(nout=1)
def calculate_seas(twindow, skipna):
    """ Calculate mean climatology for one cell grid at the time

    Parameters
    ----------
    twindow: xarray DataArray
        Stacked array timeseries with new 'z' dimension representing
        a window of width 2*w+1
    skipna: bool 
        If True percentile and mean function will use skipna=True.
        Using skipna option is much slower

    Returns
    -------
    seas_climYear: xarray DataArray
        Climatological mean
    """
    seas_climYear = (twindow
                       .groupby('doy')
                       .mean(dim='z', skipna=skipna))
    # calculate value for 29 Feb from mean of 28-29 feb and 1 Mar
    seas_climYear = seas_climYear.where(seas_climYear.doy!=60,
                                        feb29(seas_climYear))
    seas_climYear = seas_climYear.chunk({'doy': -1})
    return seas_climYear


def join_gaps(st, end, events, maxGap):
    """Find gaps between mhw events which are <= maxGap and join
    events into one event

    Parameters
    ----------
    st: pandas series
        MHW start indexes
    end: pandas series
        MHW end indexes
    events: pandas series
        MHW events
    maxGap: int
        Maximum limit of gap length (days) between events 

    Returns
    -------
    joined: pandas Dataframe
        Includes series for joined events, start and end indexes 
    """

    # remove NaNs
    # build (st,end) pairs
    # shift end series to align with st series: first value is set to
    # maxGap+1 so start of first event is always kept 
    # subtract shifted end series form start one to get gaps' lengths
    # and select as True all gaps > maxGap
    # use gaps series as selected events start indexes
    # shift back gaps series, select as True all gaps > maxGap
    # set last value to True to retain end of last event
    # use gaps shifted series as selected events end indexes
    # used new and starting pairs to detect events to join
    # reindex so we have a complete time axis
    s = st.dropna()
    e = end.dropna()
    if len(s) > 1:
        pairs = set(zip(s.values,e.values))
        eshift = e.shift(1)
        eshift = eshift.fillna(value=-(maxGap+1))
        gaps = ((s - eshift) > maxGap + 1)
        gaps_shifted = gaps.shift(-1)
        gaps_shifted = gaps_shifted.fillna(value=True)
        s = s.where(gaps).dropna()
        e = e.where(gaps_shifted).dropna()
        if len(s) < len(st.dropna()):
            joined = set(zip(s.values,e.values)) - pairs
            events = join_events(events, joined)
        st = s.reindex_like(st)
        end = e.reindex_like(end)
    else:
        pass
    joined = pd.concat([st.rename('start'), end.rename('end'), events],
                       axis=1) 
    return  joined 


@dask.delayed(nout=2)
def define_events(ts, th, se, idxarr,  minDuration, joinGaps,
                  maxGap, intermediate):
    """Finds all MHW events of duration >= minDuration and calculate
    their properties.

    If joinGaps is True than joins any event that is separated by 
    a number of days <= maxGap

    Parameters
    ----------
    ts: pandas Series
        Temperature timeseries array
    th: pandas Series
        Climatological threshold
    se: pandas Series
        Climatological mean
    idxarr: pandas Series
        Index array 
    minDuration: int
        Minimum duration (days) to accept of detected MHWs
    joinGaps: bool
       If True join MHWs separated by a short gap
    maxGap: int
        Maximum limit of gap length (days) between events
    intermediate: bool
        If True return also dataset with input data, detected events
        and some events properties along time axis

    Returns
    -------
    mhw: xarray Dataset
        Dataset including detected events and their properties
    mhw_inter: xarray Dataset, optional
        Dataset with input data, detected events and some events
        properties along time axis. If intermediate is False is None
    """

    # reindex thresh and seas along time index
    thresh = th.sel(doy=ts.doy)
    seas = se.sel(doy=ts.doy)

    # Find MHWs as exceedances above the threshold
    # Time series of "True" when threshold is exceeded
    bthresh = ts > thresh
    ds = xr.Dataset({'ts': ts, 'seas': seas, 'thresh': thresh,
                     'bthresh': bthresh})

    # Convert xarray dataset to pandas dataframe, as groupby operations
    # are faster in pandas
    df = ds.to_dataframe()
    del ds

    # detect events 
    dfev = mhw_filter(df.bthresh, idxarr, minDuration, joinGaps,
                      maxGap)

    # Prepare dataframe to get features before groupby operation
    df = mhw_df(pd.concat([df,dfev], axis=1))
    del dfev

    # Calculate mhw properties, for each event using groupby 
    dfmhw = mhw_features(df, len(idxarr)-1)

    # Convert back to xarray dataset 
    mhw = xr.Dataset.from_dataframe(dfmhw, sparse=False)
    del dfmhw
    mhw_inter = None
    if intermediate:
        df = df.drop(columns=['doy', 'cell', 'time', 'start', 'end',
                              'anom_plus', 'anom_minus'])
        mhw_inter = xr.Dataset.from_dataframe(df, sparse=False)
    del df
    return mhw, mhw_inter 


def mhw_filter(bthresh, idxarr, minDuration, joinGaps, maxGap=2):
    """Filter events of consecutive days above threshold which are
    longer then minDuration.

    Parameters
    ----------
    bthresh: boolean pandas Series
        True values where ts >= threshold for same day-of-year
    idxarr: pandas Series
        Index array 
    minDuration: int
        Minimum duration (days) to accept detected MHWs
    joinGaps: bool
       If True join MHWs separated by a short gap
    maxGap: int, optional
        Maximum limit of gap length (days) between events (default=2)

    Returns
    -------
    df: pandas Dataframe
        Includes series for events and their start and end indexes
    """
    # Build another array where the last index before the start of 
    # a succession of Trues is propagated while False points retain
    # their positional indexes
    # events = [0,1,2,3,3,3,3,3,3,9,10,...]
    events = (idxarr.where(~bthresh).ffill()).fillna(0)

    # by subtracting events from idxarr we get 1/2/3/4 ... counter
    # for each mhw and 0 elsewhere
    # events_map = [0,0,0,0,1,2,3,4,5,0,0,...]
    events_map = idxarr - events

    # subtracting events_map shifted by 1 place to the right from
    # itself, the last day of the mhw will have a negative value
    # this is also indicative of the duration of the event.
    # The series is then shifted back one place to the left and
    # the boundaries nan are replaced with zeros 
    # shifted = [nan,0,0,0,1,1,1,1,-5,0,0,...]
    shifted = (events_map - events_map.shift(+1)).shift(-1)
    shifted = shifted.where(~np.isnan(shifted), -events_map)

    # select only cells where shifted is less equal to the -minDuration,
    duration = events_map.where(shifted <= -minDuration)
    # from idxarr select where mhw duration is not NaN,
    # this will the index of last day of mhw  
    end = idxarr.where( ~np.isnan(duration))
    # removing duration from end index gives starting index
    st = (end - duration + 1)

    # add 1 to events so each event is represented by its starting index
    events = events + 1

    # Selected mhw will be represented by indexes where "events" has
    # values included in st series and where "events_map" is not 0
    sel_events = events.where(events.isin(st) & (events_map != 0))
    sel_events.name = 'events'

    # if joinGaps call join_gaps function
    if joinGaps:
        df = join_gaps(st, end, sel_events, maxGap)
    else:
        df = pd.concat([st.rename('start'), end.rename('end'),
                        sel_events], axis=1)
    return  df


def land_check(temp, tdim='time', anynans=False):
    """Return new array with all dimensions but time stacked and 
    land points removed.
    
    Parameters
    ----------
    temp: xarray DataArray
        input timeseries
    tdim: str, optional
        Name of time dimension (default='time')
    anynans: bool, optional
        Defines in land_check which cells will be dropped, if False
        only ones with all NaNs values, if True all cells with even
        1 NaN along time dimension will be dropped (default is False)

    Returns
    -------
    ts: xarray DataArray
        Modified timeseries with stacked cell dimension and land points
        removed  
    """

    dims = list(temp.dims)
    # Add an extra fake dimensions if array 1-dimensional
    # so a 'cell' dimension can still be created
    dims.remove(tdim)
    if len(dims) == 0:
        temp = temp.expand_dims({'point': [0.]})
        dims = ['point']
    for d in dims:
        if len(temp[d]) == 0:
            raise XmhwException(f'Dimension {d} has 0 lenght, exiting')
    ts = temp.stack(cell=(dims))
    # drop cells that have all/any nan values along time
    how = 'all'
    if anynans:
        how = 'any'
    ts = ts.dropna(dim='cell',how=how)
    # if ts.cell.shape is 0 then all points are land, quit
    if ts.cell.shape == (0,):
        raise XmhwException('All points of grid are either land or NaN')
    return ts


def join_events(events, joined):
    """Update 'event' series values for joined events"""
    for s,e in joined:
        events.iloc[int(s):int(e)+1] = s
    return events


def annotate_ds(ds, ds_attrs, kind):
    """Add input timeseries attributes to output dataset.
    
    Units for MHW properties are based on original timeseries units
    if units are not present in the file degree_C is assumed
    Units for dimensions are also from the original file

    kind: str
        Can be 'clim' or 'mhw' depending on dataset
    Returns
    -------
    """

    github = 'https://github.com/coecms/xmhw'
    try:
        uts = ds_attrs['temp'].units
        if any(s in uts for s in ['Celsius','celsius']):
            uts = 'degree_C'
    except:
        uts = 'degree_C'
    #set coordinates attributes
    for c in ds.coords:
        if c == 'doy':
            ds[c]['units'] = '1'
            ds[c]['long_name'] = 'Day of the year'
        elif c == 'events':
            ds[c]['units'] = '1'
            ds[c]['long_name'] = 'MHW event identifier: starting index' 
        elif c == 'point':
            continue
        else:
            try:
                for k,v in ds_attrs[c].items():
                    ds[c].attrs[k] = v
            except:
                XmhwException("Could not retrieve original attributes "
                    + f"for {c}, add attributes manually to dataset")
    # set global attributes
    if kind == 'clim':
        # set global attributes
        ds.attrs['source'] = f"xmhw code: {github}"
        # potentially add reference to input data from original dataset
        ds.attrs['title'] = (f"Seasonal climatology and threshold " +
             "calculated to detect marine heatwaves following the " +
             " Hobday et al. (2016) definition")
        ds.attrs['history'] = (
            f"{date.today()}: calculated using xmhw code {github}")
        ds.thresh.attrs['units'] = uts 
        ds.seas.attrs['units'] = uts 
        #ds.threshold.lon.attrs['units'] = 'degree_north' 
       #field.longitude.attrs['units'] = 'degree_east'
    else:
        ds.event.attrs['units'] = "1" 
        ds.event.attrs['long_name'] = "MHW event identifier: starting index" 
        ds.duration.attrs['long_name'] = "MHW duration in number of days" 
        ds.duration.attrs['units'] = '1' 
        ds.intensity_max.attrs['long_name'] = (
            "MHW maximum (peak) intensity relative to seasonal climatology") 
        ds.intensity_max.attrs['units'] = uts 
        ds.intensity_mean.attrs['long_name'] = (
            "MHW mean intensity relative to seasonal climatology") 
        ds.intensity_mean.attrs['units'] = uts 
        ds.intensity_var.attrs['long_name'] = (
            "MHW intensity variability relative to seasonal climatology") 
        ds.intensity_var.attrs['units'] = uts 
        ds.intensity_cumulative.attrs['long_name'] = (
            "MHW cumulative intensity relative to seasonal climatology") 
        ds.intensity_cumulative.attrs['units'] = f"{uts} day" 
        ds.severity_max.attrs['long_name'] = (
            "MHW maximum (peak) severity relative to seasonal climatology") 
        ds.severity_max.attrs['units'] = uts 
        ds.severity_mean.attrs['long_name'] = (
            "MHW mean severity relative to seasonal climatology") 
        ds.severity_mean.attrs['units'] = uts 
        ds.severity_var.attrs['long_name'] = (
            "MHW severity variability relative to seasonal climatology") 
        ds.severity_var.attrs['units'] = uts 
        ds.severity_cumulative.attrs['long_name'] = (
            "MHW cumulative severity relative to seasonal climatology") 
        ds.severity_cumulative.attrs['units'] = f"{uts} day" 
        ds.rate_onset.attrs['long_name'] = "MHW onset rate" 
        ds.rate_onset.attrs['units'] = f"{uts} day-1" 
        ds.rate_decline.attrs['long_name'] = "MHW decline rate" 
        ds.rate_decline.attrs['units'] = f"{uts} day-1" 
        ds.intensity_max_relThresh.attrs['long_name'] = (
            "MHW maximum (peak) intensity relative to threshold") 
        ds.intensity_max_relThresh.attrs['units'] = uts 
        ds.intensity_mean_relThresh.attrs['long_name'] = (
            "MHW mean intensity relative to threshold") 
        ds.intensity_mean_relThresh.attrs['units'] = uts 
        ds.intensity_var_relThresh.attrs['long_name'] = (
            "MHW intensity variability relative to threshold") 
        ds.intensity_var_relThresh.attrs['units'] = uts 
        ds.intensity_cumulative_relThresh.attrs['long_name'] = (
            "MHW cumulative intensity relative to threshold") 
        ds.intensity_cumulative_relThresh.attrs['units'] = f"{uts} day" 
        ds.intensity_max_abs.attrs['long_name'] = (
            "MHW maximum (peak) intensity absolute magnitude") 
        ds.intensity_max_abs.attrs['units'] = uts 
        ds.intensity_mean_abs.attrs['long_name'] = (
            "MHW mean intensity absolute magnitude") 
        ds.intensity_mean_abs.attrs['units'] = uts 
        ds.intensity_var_abs.attrs['long_name'] = (
            "MHW intensity variability abosulute magnitude")
        ds.intensity_var_abs.attrs['units'] = uts 
        ds.intensity_cumulative_abs.attrs['long_name'] = (
            "MHW cumulative intensity absolute magnitude") 
        ds.intensity_cumulative_abs.attrs['units'] = f"{uts} day" 
        # should be treated as flags from CF point of view?
        ds.category.attrs['long_name'] = ("MHW category based on peak "
            + "intensity: 1: Moderate, 2: Strong, 3: Severe or 4: Extreme") 
        ds.duration_moderate.attrs['long_name'] = (
            "Number of days falling in category Moderate") 
        ds.duration_moderate.attrs['units'] = "1" 
        ds.duration_strong.attrs['long_name'] = (
            "Number of days falling in category Strong") 
        ds.duration_strong.attrs['units'] = "1" 
        ds.duration_severe.attrs['long_name'] = (
            "Number of days falling in category Severe") 
        ds.duration_severe.attrs['units'] = "1" 
        ds.duration_extreme.attrs['long_name'] = (
            "Number of days falling in category Extreme") 
        ds.duration_extreme.attrs['units'] = "1" 
        # set global attributes
        ds.attrs['source'] = f"xmhw code: {github}"
        ds.attrs['title'] = (f"Marine heatwave events identified " +
            f"applying the Hobday et al. (2016) marine heat wave definition")
        ds.attrs['history'] = (
            f"{date.today()}: calculated using xmhw code {github}")
    return ds
