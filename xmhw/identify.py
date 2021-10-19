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
import dask
from datetime import date
from .exception import XmhwException
from .features import mhw_df, mhw_features


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

def get_calendar(tdim):
    """Retrieve calendar information or try to guess number of days in a year
       Input:
           tdim - time dimension associated with input timeseries - xarray.DataArray
       Return:
           ndays_year - number of days in a year of timeseries - float

    """
    # define a dictionary mapping calendar to ndays_year
    # check if calendar is part of the time dimension attributes
    # my assumptions here are Julian can be ignored at best from 1901 onwards we could add 13 days and consider it gregorian
    # gregorian, standard, proleptic_gregorian are all the same ,as differences happens in the distant past
    # for these we ant to use add_doy
    # for 360/ 365 /366 we need different approach, they all can stay as they are but I should then use the original day ofyear admititng this is calculated differently and consistently each time
    ndays = {'standard': 365.25, 'gregorian': 365.25, 'proleptic_gregorian': 365.25, 'all_leap': 366,
            'noleap': 365, '365_day': 365, '360_day': 360, 'julian': 365.25}
    if 'calendar' in tdim.encoding.keys():
        calendar = tdim.encoding['calendar']
    elif 'calendar' in tdim.attrs.keys():
        calendar = tdim.attrs['calendar']
    else:
        calendar = getattr(tdim.values[0], 'calendar', '')
    if calendar == '':
        #calendar = infer_calendar(tdim)
        pass
    # if calendar was retrieved by variable attributes is possible it was wrongly defined 
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
    """
    This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.
    """
    return (ts.where(ts.doy.isin([59,60,61]),drop=True).mean(dim=dim, skipna=True).values)
    #return (ts.where(ts.doy.isin([59,61]),drop=True).mean(dim=dim, skipna=True).values)


@dask.delayed(nout=1)
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
    """Return all values falling in -w/+w window from each value in array
       remove nans before returning stacked array
    """
    
    width = 2*w+1
    dtime = {tdim: width}
    trolled = ts.rolling(**dtime, center=True).construct('wdim')
    troll = trolled.stack(z=('wdim', tdim))
    return troll.dropna(dim='z')


@dask.delayed(nout=1)
def calculate_thresh(twindow, pctile, skipna):
    """ Calculate threshold for one cell grid at the time
    """
    thresh_climYear = (twindow
                       .groupby('doy')
                       .quantile(pctile/100., dim='z', skipna=skipna))
    # calculate value for 29 Feb from mean of 28-29 feb and 1 Mar
    thresh_climYear = thresh_climYear.where(thresh_climYear.doy!=60, feb29(thresh_climYear))
    thresh_climYear = thresh_climYear.chunk({'doy': -1})
    return thresh_climYear


@dask.delayed(nout=1)
def calculate_seas(twindow, skipna):
    """ Calculate mean climatology for one cell grid at the time
    """
    seas_climYear = (twindow
                       .groupby('doy')
                       .mean(dim='z', skipna=skipna))
    # calculate value for 29 Feb from mean of 28-29 feb and 1 Mar
    seas_climYear = seas_climYear.where(seas_climYear.doy!=60, feb29(seas_climYear))
    seas_climYear = seas_climYear.chunk({'doy': -1})
    return seas_climYear


def join_gaps(st, end, events, maxGap):
    """Find gaps between mhws which are less equal to maxGap and join adjacent mhw into one event
       Input:
             st - series of mhw start indexes
             end - series of mhw end indexes
             events - series of mhw events
             maxGap - all gaps <= maxGap are removed
    """
    # calculate gaps by subtracting index of end of each mhw from start of successive mhw
    # select as True all gaps > maxGap, these are the values we'll be keeping
    s = st.dropna()
    e = end.dropna()
    if len(s) > 1:
        pairs = set(zip(s.values,e.values))
    
        eshift = e.shift(1)
        # by setting first value to -(maxGap+1) then gaps[0] will always be True
        # in this way we avoid a comparison with Nan and retain first start
        #eshift = eshift.where(~np.isnan(eshift), -(maxGap + 1))
        eshift = eshift.fillna(value=-(maxGap+1))
        gaps = ((s - eshift) > maxGap + 1)
        
    # shift back gaps series
        gaps_shifted = gaps.shift(-1)
        gaps_shifted = gaps_shifted.fillna(value=True)
    # use "gaps" to select start indexes to keep
        s = s.where(gaps).dropna()
    # use "gaps_shifted" to select end indexes and duration to keep
        e = e.where(gaps_shifted).dropna()
        if len(s) < len(st.dropna()):
            joined = set(zip(s.values,e.values)) - pairs
            events = join_events(events, joined)
    # reindex so we have a complete time axis
        st = s.reindex_like(st)
        end = e.reindex_like(end)
    # update events which were joined
    else:
        pass
    return pd.concat([st.rename('start'), end.rename('end'), events], axis=1) 


@dask.delayed(nout=2)
#def define_events(ds, idxarr,  minDuration, joinAcrossGaps, maxGap, intermediate):
def define_events(ts, th, se, idxarr,  minDuration, joinAcrossGaps, maxGap, intermediate):
    """Find all MHW events of duration >= minDuration
       if joinAcrossGaps is True than joins any event that is separated by a number of days <= maxGap
    """
    # reindex thresh and seas along time index
    thresh = th.sel(doy=ts.doy)
    seas = se.sel(doy=ts.doy)

    # Find MHWs as exceedances above the threshold
    # Time series of "True" when threshold is exceeded, "False" otherwise
    bthresh = ts > thresh
    ds = xr.Dataset({'ts': ts, 'seas': seas, 'thresh': thresh, 'bthresh': bthresh})

    # Convert xarray dataset to pandas dataframe, as groupby operation are faster in pandas
    df = ds.to_dataframe()
    del ds
    # detect events 
    dfev = mhw_filter(df.bthresh, idxarr, minDuration, joinAcrossGaps, maxGap)
    # Prepare dataframe to get features before groupby operation
    df = mhw_df(pd.concat([df,dfev], axis=1))
    del dfev
    # Calculate mhw properties, for each event using groupby 
    dfmhw = mhw_features(df, len(idxarr)-1)

    # convert back to xarray dataset and reindex (?) so all cells have same event axis
    mhw = xr.Dataset.from_dataframe(dfmhw, sparse=False)
    del dfmhw
    mhw_inter = None
    if intermediate:
        df = df.drop(columns=['doy', 'cell', 'time', 'start', 'end', 'anom_plus', 'anom_minus'])
        mhw_inter = xr.Dataset.from_dataframe(df, sparse=False)
    del df
    return mhw, mhw_inter 


def mhw_filter(bthresh, idxarr, minDuration=5, joinGaps=True, maxGap=2):
    """ Filter events of consecutive days above threshold which are longer then minDuration
        bthresh - boolean series with True values where ts >= threshold value for same dayofyear
        a = series of same length with indexes maybe I can sue bthresh.index?  
    """
    # Build another array where the last index before the start of a succession of Trues is propagated
    # while False points retain their positional indexes
    # events = [0,1,2,3,3,3,3,3,3,9,10,...]
    events = (idxarr.where(~bthresh).ffill()).fillna(0)
    # by removing the 2nd array from the 1st we get 1/2/3/4 ... counter for each mhw and 0 elsewhere
    # events_map = [0,0,0,0,1,2,3,4,5,0,0,...]
    events_map = idxarr - events
    # removing the series shifted by 1 place to the right from itself we're left with only the last day of the mhw having a negative counter
    # this is also indicative of the duration of the event, the series is then shifted back one place to the left and the boundaries nan are replaced with zeros 
    # shifted = [nan,0,0,0,1,1,1,1,-5,0,0,...]
    shifted = (events_map - events_map.shift(+1)).shift(-1)

    #shifted.iloc[-1] = -events_map.iloc[-1]
    shifted = shifted.where(~np.isnan(shifted), -events_map)
    # select only cells where shifted is less equal to the -minDuration,
    duration = events_map.where(shifted <= -minDuration)
    # from idxarr select where mhw duration, this will the index of last day of mhw  
    end = idxarr.where( ~np.isnan(duration))
    # removing duration from end index gives starting index
    st = (end - duration + 1)

    # add 1 to events so each event is represented by its starting index
    events = events + 1
    # Selected mhw will be represented by indexes where "events" has values included in st list
    # and where "events_map" is not 0
    sel_events = events.where(events.isin(st) & (events_map != 0))
    sel_events.name = 'events'


    # if joinAcross Gaps call join_gaps function, this will update start, end and mappings of events
    if joinGaps:
        df = join_gaps(st, end, sel_events, maxGap)
    else:
        df = pd.concat([st.rename('start'), end.rename('end'), sel_events], axis=1)
    return  df


def land_check(temp, tdim='time', removeNans=False):
    """ Stack lat/lon on new dimension cell and remove for land points
        Input:
        temp - sst timeseries on multi-dimensional grid
        Return
        ts - modified timeseries with stacked lat/lon and land points removed  
    """
    dims = list(temp.dims)
    # add an extra fake dimensions if array 1-dimensional so a 'cell' dimension can still be created
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
    if removeNans:
        how = 'any'
    ts = ts.dropna(dim='cell',how=how)
    # if ts.cell.shape is 0 then all points are land, quit
    if ts.cell.shape == (0,):
        raise XmhwException('All points of grid are either land or NaN')
    return ts


def join_events(events, joined):
    """ Set right value for joined events """
    for s,e in joined:
        events.iloc[int(s):int(e)+1] = s
    return events

def annotate_ds(ds, ds_attrs, kind):
    """ Add attributes to output dataset, kind refer to clim or mhw
        Units for MHW properties are based on original timeseries units
        if units are not present in the file degree_C is assumed
        Units for dimensions are also from the original file
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
                XmhwException(f"Could not retrieve original attributes for {c}, add attributes manually to final dataset")
    # set global attributes
    if kind == 'clim':
        # set global attributes
        ds.attrs['source'] = f'xmhw code: {github}'
        # potentially add reference to input data from original dataset
        ds.attrs['title'] = f'Seasonal climatology and threshold calculated to detect marine heatwaves following the Hobday et al. (2016) definition'
        ds.attrs['history'] = f'{date.today()}: calculated using xmhw code {github}'
        ds.thresh.attrs['units'] = uts 
        ds.seas.attrs['units'] = uts 
        #ds.threshold.lon.attrs['units'] = 'degree_north' 
       #field.longitude.attrs['units'] = 'degree_east'
    else:
        ds.event.attrs['units'] = '1' 
        ds.event.attrs['long_name'] = 'MHW event identifier: starting index' 
        ds.duration.attrs['long_name'] = 'MHW duration in number of days' 
        ds.duration.attrs['units'] = '1' 
        ds.intensity_max.attrs['long_name'] = 'MHW maximum (peak) intensity relative to seasonal climatology' 
        ds.intensity_max.attrs['units'] = uts 
        ds.intensity_mean.attrs['long_name'] = 'MHW mean intensity relative to seasonal climatology' 
        ds.intensity_mean.attrs['units'] = uts 
        ds.intensity_var.attrs['long_name'] = 'MHW intensity variability relative to seasonal climatology' 
        ds.intensity_var.attrs['units'] = uts 
        ds.intensity_cumulative.attrs['long_name'] = 'MHW cumulative intensity relative to seasonal climatology' 
        ds.intensity_cumulative.attrs['units'] = f'{uts} day' 
        ds.severity_max.attrs['long_name'] = 'MHW maximum (peak) severity relative to seasonal climatology' 
        ds.severity_max.attrs['units'] = uts 
        ds.severity_mean.attrs['long_name'] = 'MHW mean severity relative to seasonal climatology' 
        ds.severity_mean.attrs['units'] = uts 
        ds.severity_var.attrs['long_name'] = 'MHW severity variability relative to seasonal climatology' 
        ds.severity_var.attrs['units'] = uts 
        ds.severity_cumulative.attrs['long_name'] = 'MHW cumulative severity relative to seasonal climatology' 
        ds.severity_cumulative.attrs['units'] = f'{uts} day' 
        ds.rate_onset.attrs['long_name'] = 'MHW onset rate' 
        ds.rate_onset.attrs['units'] = f'{uts} day-1' 
        ds.rate_decline.attrs['long_name'] = 'MHW decline rate' 
        ds.rate_decline.attrs['units'] = f'{uts} day-1' 
        ds.intensity_max_relThresh.attrs['long_name'] = 'MHW maximum (peak) intensity relative to threshold' 
        ds.intensity_max_relThresh.attrs['units'] = uts 
        ds.intensity_mean_relThresh.attrs['long_name'] = 'MHW mean intensity relative to threshold' 
        ds.intensity_mean_relThresh.attrs['units'] = uts 
        ds.intensity_var_relThresh.attrs['long_name'] = 'MHW intensity variability relative to threshold' 
        ds.intensity_var_relThresh.attrs['units'] = uts 
        ds.intensity_cumulative_relThresh.attrs['long_name'] = 'MHW cumulative intensity relative to threshold' 
        ds.intensity_cumulative_relThresh.attrs['units'] = f'{uts} day' 
        ds.intensity_max_abs.attrs['long_name'] = 'MHW maximum (peak) intensity absolute magnitude' 
        ds.intensity_max_abs.attrs['units'] = uts 
        ds.intensity_mean_abs.attrs['long_name'] = 'MHW mean intensity absolute magnitude' 
        ds.intensity_mean_abs.attrs['units'] = uts 
        ds.intensity_var_abs.attrs['long_name'] = 'MHW intensity variability abosulute magnitude' 
        ds.intensity_var_abs.attrs['units'] = uts 
        ds.intensity_cumulative_abs.attrs['long_name'] = 'MHW cumulative intensity absolute magnitude' 
        ds.intensity_cumulative_abs.attrs['units'] = f'{uts} day' 
        # should be treated as flags from CF point of view?
        ds.category.attrs['long_name'] = 'MHW category based on peak intensity: 1: Moderate, 2: Strong, 3: Severe or 4: Extreme' 
        ds.duration_moderate.attrs['long_name'] = 'Number of days falling in category Moderate' 
        ds.duration_moderate.attrs['units'] = '1' 
        ds.duration_strong.attrs['long_name'] = 'Number of days falling in category Strong' 
        ds.duration_strong.attrs['units'] = '1' 
        ds.duration_severe.attrs['long_name'] = 'Number of days falling in category Severe' 
        ds.duration_severe.attrs['units'] = '1' 
        ds.duration_extreme.attrs['long_name'] = 'Number of days falling in category Extreme' 
        ds.duration_extreme.attrs['units'] = '1' 
        # set global attributes
        ds.attrs['source'] = f'xmhw code: {github}'
        ds.attrs['title'] = f'Marine heatwave events identified applying the Hobday et al. (2016) marine heat wave definition'
        ds.attrs['history'] = f'{date.today()}: calculated using xmhw code {github}'
    return ds

