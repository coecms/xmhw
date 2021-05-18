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


def block_average(mhw, temp=None, clim=None, blockLength=1, mtime='time_start',
                  removeMissing=False, split=False):
    '''
    Inputs:
      
      mhw     marine heatwave events xarray dataset 

      clim    Climatology xarray dataset used to detect marine heatwave events (optional)

      temp    Temperature array used  to calculate climatologies (optional)

    Options:

      mtime                  timestep to use to assign mhw to bins, default is 'time_start' to follow 
                             Eric code, use mtime='index_peak' to use mhw peak instead
      blockLength            Size of block (in years) over which to calculate the
                             averaged MHW properties. Must be an integer greater than
                             or equal to 1 (DEFAULT = 1 [year])
      removeMissing          Boolean switch indicating whether to remove (set = NaN)
                             statistics for any blocks in which there were missing
                             temperature values (Default = False)
      clim                   The temperature climatology (including missing value information)
                             as output by marineHeatWaves.detect
      temp                   Temperature time series. If included mhwBlock will output block
                             averages of mean, max, and min temperature (Default = None but
                             required if removeMissing = TRUE)
      indexes                Indexes generated as intermediate products to calculate the MHW events
                             characteristics. Default is None but it is required if split is True
                             as output by marineHeatWaves.detect
      split                  Split events crossing a block boundary into two events and calculate new stats for them
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
    # Check if all the necessary variables are present
    if removeMissing and not temp:
        print(f'To remove missing values you need to pass the original temperature timeseries')
        return None
    # Check what stats to output
    # if temp included calculate stats for it, if clim also included calculate categories days count
    sw_temp=False
    sw_cats=False
    if temp is not None:
        sw_temp = True
        if clim is not None:
            sw_cats = True
        else:
            sw_cats = False
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

    # check if variable to use to assing mhws to blocks is a time array or not
    #if mhw[mtime].dt is None :
    #    tgroup = mhw[mtime].time.dt.year
    #else:
    tgroup = mhw[mtime].dt.year


    # calculate mean of variables after grouping by year
    block = mhw.groupby_bins(tgroup, bins, right=False).mean()

    # remove averages of indexes, events and category (which need special treatment)
    block = block.drop(['event', 'index_start', 'index_end', 'index_peak'] )

    # Other stats can be calculated one by one
    # count
    block['ecount'] = mhw.events.groupby_bins(tgroup, bins, right=False).count()
    # total_days
    # total_icum
    # calculate maximum of intensity_max
    block['intensity_max_max'] = mhw.intensity_max.groupby_bins(tgroup, bins, right=False).max()
    # calculate maximum of intensity_max
    block['total_icum'] = mhw.intensity_cumulative.groupby_bins(tgroup, bins, right=False).sum()

    # test to see if with big dataset we need to do again a cell by cell calculation!
    # calculate with aggregation function
    blockls = []
    ds = land_check(mhw, tdim='events') 
    for c in ds.cell:
        blockls.append(groupby_mhw(ds.sel(cell=c), tgroup, bins))
    results = dask.compute(blockls)
    block2 = xr.concat(results[0], dim=ds.cell).unstack('cell')


    # if sw_temp
    if sw_temp:
        pass
    if sw_cats:
        pass

    return block, block2


@dask.delayed
def groupby_mhw(ds, tgroup, bins):
    """Call groupby on mhw results cell by cell

       Input:
       ds = mhw dataset for 1 grid point - xarray Dataset
       tgroup = years of mhw time variable to use to assign events to blocks - xarray array 
       bins = intervals to use to define blocks
    """
    # convert mhw Dataset to Dataframe
    df = ds.to_dataframe()
    # groupby mtime and aggregate variables
    dfblock = agg_mhw(df, tgroup, bins)
    # convert Catgorical index to normal index
    dfblock.index = dfblock.index.to_list()
    print(dfblock.index)
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
            moderate_days = ('duration_moderate', 'sum'),
            strong_days = ('duration_strong', 'sum'),
            rate_onset = ('rate_onset', 'mean'),
            rate_decline = ('rate_decline', 'mean'))



def find_across(mhw):
    """Find all events that span across a year"""
    return mhw.where(mhw['time_start'].dt.year != mhw['time_end'].dt.year).dropna(dim='events')

def split_event(mhw_ev):
    "If event span across two years you might wan tto split it into two and calculate separate stats"
    return mhw_ev
