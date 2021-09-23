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


def block_average(mhw, dstime=None, period=None, blockLength=1, mtime='time_start',
                  removeMissing=False, split=False):
    """Calculate statistics like averages, mean and maximum on blocks of years,
    length of blocks is in years (1 by default). Each event is assigned to a
    year based on one of time variables: start (default), end or peak. 
    Inputs:
      
      mhw     marine heatwave events xarray dataset 

    Options:

      dstime            xarray dataset with 'time' dimension containing at least original sst,
                        and climatologies as optional, default is None
                        if present and is array or dataset with only 1 variable script assumes this
                        is sst and sw_temp is set to True
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

   """ 
    # Check if dstime is present and how many variables are included
    # if dstime has only and extra coordinate on top of time than calling
    # land_check is not necessary for dstime
    # and the mhw 'cell' coordinate should be renamed to be consistent
    sw_temp = False
    sw_cats = False
    # currently I'm try to accept any name for tempearute as long as it is passed on its own and only temp thresh seas if they are all
    # passed as a dataset
    # might make more sense to deal with this separately and allow to indicate the names?
    # put this ins eparate function and add option to check if cats is alredy present, then we do not need clims
    if dstime is not None:
        dstime, sw_cats, sw_temp = check_variables(dstime)
        dstime, stack_coord = check_coordinates(dstime)
        period = [dstime.time.dt.year[0].values, dstime.time.dt.year[-1].values]

    # Check if all the necessary variables are present
    if removeMissing and not sw_temp:
       raise XmhwException("To remove missing values you need to pass " \
             "the original temperature timeseries")
    if not period and not sw_temp:
       raise XmhwException(f"As the original timeseries is not available, the" \
             " timeseries period as [start_year, end_year] has to be passed")
    # if split option divide events starting in one year and ending in successive in two and recalculate averages??
    # or (much eaiser if events cross a year/block assign them to block that includes most days of detected event
    if split:
        evs = find_across(mhw)
        if len(evs.events) > 0:
            altered = split_event(evs)
    # create bins based on blockLength to used with groupby_bins
    # NB if the last bin has less than blockLength years, it won't be included.
    # So I'm using last-year+blockLength+1 to make sure we get a bin for last year/s included
    bins=range( period[0], period[1]+blockLength+1,blockLength)

    # calculate with aggregation function
    blockls = []
    # remove land and stack on cell
    mhw = land_check(mhw, tdim='events') 
    # this defines years array to use to groupby arrays
    tgroup = mhw[mtime].isel(cell=0).dt.year
    for c in mhw.cell:
        blockls.append(call_groupby(mhw.sel(cell=c), tgroup, bins))
    results = dask.compute(blockls)
    block = xr.concat(results[0], dim=mhw.cell).unstack('cell')

    # if we have sst and/or climatologies we add more stats along time axis
    if sw_temp:
        if sw_cats:
            print("Both sst and climatologies are available, " \
                  "calculating sst and category stats") 
            mode = 'cats'
        else:
            mode = 'sst'
        tgroup = dstime.isel({stack_coord: 0}).time.dt.year
        statsls = []
        for c in dstime[stack_coord]:
            cell_stats = call_groupby(dstime.sel({stack_coord: c}),
                                      tgroup, bins, mode=mode)
            statsls.append(cell_stats)
            results = dask.compute(statsls)
        tstats = xr.concat(results[0], dim=dstime[stack_coord])
        if stack_coord == 'cell':
            tstats = tstats.unstack(stack_coord)
        block = xr.merge([block, tstats])

    return block


def check_variables(dstime):
    """Check dstime variables and coordinates to make sure it will work and to determine
       what stats can be calculated.

       Input:
           dstime
       Return:

    """
    # if the user passed dstime the assumption is at least sst is present
    # if dstime is DataArray convert it to dataset
    sw_temp = True
    sw_cats = False
    if 'dataarray' in str(type(dstime)):
        dstime = xr.Dataset({'sst': dstime})

    # check variable/s names , if only variable assume it even is not 'sst'
    # else if there is more than 1  variable:
    # 1 - check if cats is present, if yes set sw_cats to True
    # 2 - else if cats is not present, check if 'sst', 'thresh' and 'seas' are
    #     all included and calculate cats from them
    # 3 - else if we cannot calculate cats make sure 'sst' is present if not
    #     set sw_temp to False and print warning

    variables = list(dstime.keys())
    if len(variables) == 1:
        sst_name = variables[0]
        dstime['sst'] = dstime[sst_name].rename('sst')
    elif 'cats' in variables:
        sw_cats = True
    elif all(x in variables for x in ['sst', 'thresh', 'seas']):
        sw_cats = True
        dstime['cats'] = np.floor(1 + (dstime['sst'] - dstime['thresh']) / (dstime['thresh'] - dstime['seas'])).astype(int)
    elif 'sst' not in variables:
            sw_temp = False
            print("We cannot identify which variable is the sst timeseries as it is not named 'sst'")
    dstime = dstime.drop_vars([v for v in dstime.keys() if v not in ['sst','cats']])
    return dstime, sw_cats, sw_temp 


def check_coordinates(dstime):
    # First check if we need to stack coordinates before passing data to loop
    # We assume that if ther eis only another coordinate apart from time there is no need to run a land_check
    # If there are more than 1 coordinates left than we need to stach the arrays before passing them to loop
    dscoords = list(dstime.coords)
    dscoords.remove('time')
    if len(dscoords) == 1:
        stack_coord = dscoords[0]
    else:
        stack_coord = 'cell'
        dstime = land_check(dstime)
        #if sw_cats:
        #    dstime['cats'] = land_check(dstime['cats'])
    return dstime, stack_coord


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
    # add total number of actual MHW days per block
    if sw_cats:
        dfblock['total_days'] = (dfblock['moderate_days'] +
                            dfblock['strong_days'] + 
                            dfblock['severe_days'] + 
                            dfblock['extreme_days'])
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
            sst_mean = ('sst', 'mean'),
            sst_max = ('sst', 'max'),
            sst_min = ('sst', 'min'),
            moderate_days = ('cats', lambda x: cat_days(x,1)),
            strong_days = ('cats', lambda x: cat_days(x,2)),
            severe_days = ('cats', lambda x: cat_days(x,3)),
            extreme_days = ('cats', lambda x: cat_days(x,4)))


def agg_sst(df, tgroup, bins):
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
            sst_mean = ('sst', 'mean'),
            sst_max = ('sst', 'max'),
            sst_min = ('sst', 'min'))


def find_across(mhw):
    """Find all events that span across a year."""
    return mhw.where(mhw['time_start'].dt.year != mhw['time_end'].dt.year).dropna(dim='events')


def split_event(mhw_ev):
    """If event span across two years you might want to split it into two 
    and calculate separate stats.
    """
    return mhw_ev


def mhw_rank(mhwds):
    """Rank mhw on each properties going from largest to smallest (1,.., n-events)
    Calculate the rank and return periods of marine heatwaves (MHWs) according to
    each metric. Takes as input an xarray dataset of detected MHWs, returned by the detetct function.
    Any date or index variable is ignored.

    Inputs:
      mhw     Marine heat waves (MHWs) xarray dataset 
    Return:
      rank          The rank of each MHW according to each MHW property. A rank of 1 is the
                    largest, 2 is the 2nd largest, etc. Each key (listed below) is a list
                    of length N where N is the number of MHWs.
      returnPeriod  The return period (in years) of each MHW according to each MHW property.
                    The return period signifies, statistically, the recurrence interval for
                    an event at least as large/long as the event in quetion. Each key (listed
                    below) is a list of length N where N is the number of MHWs.

    Notes:
      This function assumes that the MHWs were calculated over a suitably long record that return
      periods make sense. If the record length is a few years or less than this becomes meaningless.

    """
    # should be based on calendar
    days_year = 365.25
    nYears = 14245/days_year
    rank = xr.Dataset() 
    return_period = xr.Dataset() 
    # skip index and time variables
    variables =  [k for k in mhwds.keys() if not any(x in k for x in ['event', 'time', 'index'])]
    for var in variables:
        rank[var] = ranke_variable(mhwds[var])
        return_period[var] = (nYears + 1) / rank[var]
    return rank, return_period


def rank_variable(array):
    """Rank an array assigning values 1,2,3 ... to array elements going from biggest to smallest
     Input:
           array: a data array
     Return:
           rank: a rank order array
    """
    rank_values = len(array) - array.values.argsort().argsort()
    rank = xr.full_like(array, np.nan)
    rank[:] = rank_values
    return rank 
