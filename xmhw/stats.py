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
import sys
import time
from xmhw.identify import land_check
from .exception import XmhwException


def block_average(mhw, dstime=None, blockLength=1, mtime='time_start',
                  removeMissing=False, split=False):
    '''
    Inputs:
      
      mhw     marine heatwave events xarray dataset 


    Options:

      dstime            xarray dataset with 'time' dimension containing at least original sst,
                        and climatologies as optional, default is None
                        if present and is array or dataset with only 1 variable script assumes this
                        is temp and sw_temp is set to True
                        if present and dataset with 'thresh' and 'seas' also sw_cats is set to True
                        Best option is to pass the intermediate dataset you get from detect
      mtime             timestep to use to assign mhw to bins, default is 'time_start' to follow 
                        Eric code, use mtime='index_peak' to use mhw peak instead
      blockLength       Size of block (in years) over which to calculate the
                        averaged MHW properties. Must be an integer greater than
                        or equal to 1 (DEFAULT = 1 [year])
      removeMissing     Boolean switch indicating whether to remove (set = NaN)
                        statistics for any blocks in which there were missing
                        temperature values (Default = False)
                        required if removeMissing = TRUE)
      indexes           Indexes generated as intermediate products to calculate the MHW events
                        characteristics. Default is None but it is required if split is True
                        as output by marineHeatWaves.detect
      split             Split events crossing a block boundary into two events and calculate new stats for them
                        (Default = False, requires indexes if True)
                        NB total_days and categories are assigned based on actual year already


                        If both clim and temp are provided, this will output annual counts
                        of moderate, strong, severe, and extreme days.

    Notes:

      This function assumes that the input time vector consists of continuous daily values. Note that
      in the case of time ranges which start and end part-way through the calendar year, the block
      averages at the endpoints, for which there is less than a block length of data, will need to be
      interpreted with care.

    '''
    # Check if dstime is present and how many variables are included
    sw_temp = False
    sw_cats = False
    # currently I'm try to accept any name for tempearute as long as it is passed on its own and only temp thresh seas if they are all
    # passed as a dataset
    # might make more sense to deal with this separately and allow to indicate the names?
    if dstime is not None:
        sw_temp = True
        # if dstime is xarray convert it to dataset
        if 'dataarray' in str(type(dstime)):
            dstime = xr.Dataset({'temp': dstime})
        # if there is only 1 variable assume is sst sw_temp is True and sw_cats stays False 
        variables = list(dstime.keys())
        if len(variables) == 1:
            temp_name = variables[0]
            dstime['temp'] = dstime[temp_name].rename('temp')
        # if there are 3 or more variables make sure thresh and seas are included
        if len(variables) >= 3 and  all(x in variables for x in ['temp', 'thresh', 'seas']):
            sw_cats = True


    # Check if all the necessary variables are present
    if removeMissing and not sw_temp:
        print(f'To remove missing values you need to pass the original temperature timeseries')
        return None
    # if split option divide events starting in one year and ending in successive in two and recalculate averages??
    # or (much eaiser if events cross a year/block assign them to block that includes most days of detected event
    if split:
        evs = find_across(mhw)
        if len(evs.events) > 0:
            altered = split_event(evs)
    # create bins based on blockLength to used with groupby_bins
    # NB if the last bin has less than blockLength years, it won't be included.
    # So I'm using last-year+blockLength+1 to make sure we get a bin for last year/s included
    bins=range(1982,2020+blockLength+1,blockLength)

    # calculate with aggregation function
    blockls = []
    # remove land and stack on cell
    mhw = land_check(mhw, tdim='events') 
    # this defines years array to use to groupby arrays
    tgroup = mhw[mtime].isel(cell=0).dt.year
    for c in mhw.cell:
        blockls.append(call_groupby(mhw.sel(cell=c), tgroup, bins))
    results = dask.compute(blockls)
    block = xr.concat(results[0], dim=mhw.cell)



    # if we have sst and/or climatologies we add more stats along time axis
    if sw_temp:
        if sw_cats:
            print("Both sst and climatologies are available, calculating sst and category stats") 
            dstime['cats'] = np.floor(1 + (dstime['temp'] - dstime['thresh']) / (dstime['thresh'] - dstime['seas'])).astype(int)
            dstime = dstime.drop_vars([v for v in dstime.keys() if v not in ['temp','cats']])
            if 'cell' not in dstime.coords:
                dstime['cats'] = land_check(dstime['cats'])
                dstime['temp'] = land_check(dstime['temp'])
            mode = 'cats'
        else:
            mode = 'temp'
            if 'cell' not in dstime.coords:
                dstime['temp'] = land_check(dstime['temp'])
        tgroup = dstime.isel(cell=0).time.dt.year
        statsls = []
        print(f'should be here')
        for c in dstime.cell:
            statsls.append(call_groupby(dstime.sel(cell=c), tgroup, bins, mode=mode))
            results = dask.compute(statsls)
        tstats = xr.concat(results[0], dim=dstime.cell)
        print(f"tstats, {tstats}")
        block = xr.merge([block, tstats])

    return block.unstack('cell')


@dask.delayed
def call_groupby(ds, tgroup, bins, mode='mhw'):
    """Call groupby on mhw results cell by cell

       Input:
       ds = mhw dataset for 1 grid point - xarray Dataset
       tgroup = years of mhw time variable to use to assign events to blocks - xarray array 
       bins = intervals to use to define blocks
       mode = define which agg function to call, default is 'mhw' - string
    """
    # convert mhw Dataset to Dataframe
    df = ds.to_dataframe()
    # groupby mtime and aggregate variables
    fagg = 'agg_' + mode
    dfblock = globals()[fagg](df, tgroup, bins)
    # convert Catgorical index to normal index
    dfblock.index = dfblock.index.to_list()
    dfblock.index.name = 'years'
    # convert back to xarray dataset
    block = xr.Dataset.from_dataframe(dfblock, sparse=False)
    del df, dfblock
    return block


def agg_mhw(df, tgroup, bins):
    """Apply groupby on dataframe after defining an aggregation dictionary to avoid apply groupby several times

       Input:
       df = mhw dataset for 1 grid point - pandas Dataframe
       mtime = name of mhw time variable to use to assign events to blocks - string
       Return:
       pandas Dataframe resulting from grouping by mtime and aggregating variables following dictionary
    """
    # first use pandas.cut to separate datFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    return df.groupby(dfbins).agg(
            ecount = ('event', 'count'),
            duration = ('duration', 'mean'),
            intensity_max = ('intensity_max', 'mean'),
            intensity_max_max = ('intensity_max', 'max'),
            intensity_mean = ('intensity_mean', 'mean'),
            intensity_cumulative = ('intensity_cumulative', 'mean'),
            total_icum = ('intensity_cumulative', 'sum'),
            intensity_mean_relThresh = ('intensity_mean_relThresh', 'mean'),
            intensity_cumulative_relThresh = ('intensity_cumulative_relThresh', 'mean'),
            severity_mean = ('severity_mean', 'mean'),
            severity_cumulative = ('severity_cumulative', 'mean'),
            intensity_mean_abs = ('intensity_mean', 'mean'),
            intensity_cumulative_abs = ('intensity_cumulative', 'mean'),
            rate_onset = ('rate_onset', 'mean'),
            rate_decline = ('rate_decline', 'mean'))


def cat_days(series, cat=1):
    """ Return count of days where category == cat 
    """
    return series[series == cat].count()


def agg_cats(df, tgroup, bins):
    """Apply groupby on dataframe after defining an aggregation dictionary to avoid apply groupby several times

       Input:
       df = cats for 1 grid point - pandas Dataframe
       mtime = name of mhw time variable to use to assign events to blocks - string
       Return:
       pandas Dataframe resulting from grouping by mtime and aggregating variables following dictionary
    """
    # first use pandas.cut to separate datFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    return df.groupby(dfbins).agg(
            temp_mean = ('temp', 'mean'),
            temp_max = ('temp', 'max'),
            temp_min = ('temp', 'min'),
            moderate_days = ('cats', lambda x: cat_days(x,1)),
            strong_days = ('cats', lambda x: cat_days(x,2)),
            severe_days = ('cats', lambda x: cat_days(x,3)),
            extreme_days = ('cats', lambda x: cat_days(x,4)))


def agg_temp(df, tgroup, bins):
    """Apply groupby on dataframe after defining an aggregation dictionary to avoid apply groupby several times

       Input:
       df = cats for 1 grid point - pandas Dataframe
       mtime = name of mhw time variable to use to assign events to blocks - string
       Return:
       pandas Dataframe resulting from grouping by mtime and aggregating variables following dictionary
    """
    # first use pandas.cut to separate datFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    return df.groupby(dfbins).agg(
            temp_mean = ('temp', 'mean'),
            temp_max = ('temp', 'max'),
            temp_min = ('temp', 'min'))


def find_across(mhw):
    """Find all events that span across a year"""
    return mhw.where(mhw['time_start'].dt.year != mhw['time_end'].dt.year).dropna(dim='events')


def split_event(mhw_ev):
    "If event span across two years you might wan tto split it into two and calculate separate stats"
    return mhw_ev
