#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 ARC Centre of Excellence for Climate Extremes
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
from xmhw.identify import land_check
from .exception import XmhwException


def block_average(
    mhw,
    dstime=None,
    period=None,
    blockLength=1,
    mtime="time_start",
    removeMissing=False,
    split=False,
):
    """Calculate statistics like averages, mean and maximum on blocks of years.

    This function assumes that the input time vector consists of continuous
    daily values. Note that in the case of time ranges which start and end
    part-way through the calendar year, the block averages at the endpoints,
    for which there is less than a block length of data, will need to be
    interpreted with care.

    Parameters
    ----------
    mhw: xarray Dataset
        Includes MHW properties by events
    dstime: xarray DataArray/Dataset, optional
         Based on intermediate dataset returned by detect(), includes
         original ts and climatologies (optional) along 'time' dimension
         (default is None)
         If present and is array or dataset with only 1 variable function
         assumes this is ts and sw_temp is set to True
         If present and dataset with 'thresh' and 'seas', also sw_cats
         is set to True
    period: pandas Series
        Absolute value of temperature along time index
    blockLength: int, optional
        Size of blocks in years (default=1)
    mtime: str, optional
        Name of mhw time variable to use to assign events to blocks,
        Options are start or peak times (default='time_start')
    removeMissing: bool, optional
        If True remove statistics for any blocks that has NaNs in ts.
        (default is False)
    split: bool, optional
        (default is False)

    Returns
    -------
    df: pandas Dataframe
        As input but with more MHW properties added
        If both clim and temp are provided, this will output annual counts
        of moderate, strong, severe, and extreme days.

      indexes   Indexes generated as intermediate products by detect()
                to calculate the MHW events characteristics.
                Default is None but it is required if split is True
      split     Split events crossing a block boundary into two events and
                calculate new stats for them
                (Default = False, requires indexes if True)
                NB total_days and categories are assigned based on actual year
                already
    """

    # Check if dstime is present and how many variables are included
    # if dstime has only an extra coordinate on top of time than calling
    # land_check is not necessary for dstime
    # and the mhw 'cell' coordinate should be renamed to be consistent
    sw_temp = False
    sw_cats = False
    point = False
    # currently I'm try to accept any name for temperature as long as it
    # is passed on its own and only temp thresh seas if they are all
    # passed as a dataset. Might make more sense to deal with this
    # separately and allow to indicate the names?
    # put this in separate function and add option to check if cats i
    # is already present, then we do not need clims
    if dstime is not None:
        dstime, sw_cats, sw_temp = check_variables(dstime)
        dstime, stack_coord = check_coordinates(dstime)
        if stack_coord is None:
            point = True
        period = [
            dstime.time.dt.year[0].values,
            dstime.time.dt.year[-1].values,
        ]

    # Check if all the necessary variables are present
    if removeMissing and not sw_temp:
        raise XmhwException(
            "To remove missing values you need to pass "
            "the original temperature timeseries"
        )
    if not period and not sw_temp:
        raise XmhwException(
            "As the original timeseries is not available, the"
            " timeseries period as [start_year, end_year] has to be passed"
        )
    # if split option divide events starting in one year and ending
    # in successive in two and recalculate averages??
    # or (much eaiser if events cross a year/block assign them to block
    # that includes most days of detected event
    if split:
        evs = find_across(mhw)
        if len(evs.events) > 0:
            altered = split_event(evs)
    # create bins based on blockLength to used with groupby_bins
    # NB if the last bin has less than blockLength years, it won't be included.
    # So I'm using last-year+blockLength+1 to make sure we get a bin
    # for last year/s included
    bins = range(period[0], period[1] + blockLength + 1, blockLength)

    # calculate with aggregation function
    blockls = []
    # this defines years array to use to groupby arrays
    tgroup = mhw[mtime].dt.year
    if stack_coord == 'point':
        blockls.append(call_groupby(mhw, tgroup, bins))
    else:
        # remove land and stack on cell
        mhw = land_check(mhw, tdim="events")
        for c in mhw.cell:
            blockls.append(call_groupby(mhw.sel(cell=c), tgroup, bins))
    results = dask.compute(blockls)
    if stack_coord == 'point':
        block = blockls[0]
    else:
        block = xr.concat(results[0], dim=mhw.cell).unstack("cell")

    # if we have ts and/or climatologies we add more stats along time axis
    if sw_temp:
        if sw_cats:
            print(
                "Both ts and climatologies are available, "
                "calculating ts and category stats"
            )
            mode = "cats"
        else:
            mode = "ts"
        #tgroup = dstime.isel({stack_coord: 0}).time.dt.year
        tgroup = dstime.time.dt.year
        statsls = []
        if stack_coord == 'point':
            statsls.append( call_groupby(
                    dstime.sel({stack_coord: c}), tgroup, bins, mode=mode)
                )
        else:
            for c in dstime[stack_coord]:
                cell_stats = call_groupby(
                    dstime.sel({stack_coord: c}), tgroup, bins, mode=mode
                )
                statsls.append(cell_stats)
        results = dask.compute(statsls)
        if stack_coord == 'point':
            tstast = results[0][0]
        else:
            tstats = xr.concat(results[0], dim=dstime[stack_coord])
        if stack_coord == "cell":
            tstats = tstats.unstack(stack_coord)
        block = xr.merge([block, tstats])

    return block


def check_variables(dstime):
    """Check dstime variables and coordinates to determine stats to calculate.

    Parameters
    ----------
    dstime: xarray DataArray/Dataset
         includes original ts and climatologies (optional) along 'time' dim

    Returns:
    -------
    dstime: xarray Dataset
        Same as input after removing extra variables if necessary
    sw_cats: bool
        True if categories are present
    sw_temp: bool
        True if temperature is present

    """
    # if the user passed dstime the assumption is at least ts is present
    # if dstime is DataArray convert it to dataset
    sw_temp = True
    sw_cats = False
    if "dataarray" in str(type(dstime)):
        dstime = xr.Dataset({"ts": dstime})

    # check variable/s names , if only variable assume it even is not 'ts'
    # else if there is more than 1  variable:
    # 1 - check if cats is present, if yes set sw_cats to True
    # 2 - else if cats is not present, check if 'ts', 'thresh' and 'seas' are
    #     all included and calculate cats from them
    # 3 - else if we cannot calculate cats make sure 'ts' is present if not
    #     set sw_temp to False and print warning

    variables = list(dstime.keys())
    if len(variables) == 1:
        ts_name = variables[0]
        dstime["ts"] = dstime[ts_name].rename("ts")
    elif "cats" in variables:
        sw_cats = True
    elif all(x in variables for x in ["ts", "thresh", "seas"]):
        sw_cats = True
        dstime["cats"] = np.floor(
            1
            + (dstime["ts"] - dstime["thresh"])
            / (dstime["thresh"] - dstime["seas"])
        ).astype(int)
    if "ts" not in variables:
        sw_temp = False
        print("Cannot identify temperature as it is not named 'ts'")
    dstime = dstime.drop_vars(
        [v for v in dstime.keys() if v not in ["ts", "cats"]]
    )
    return dstime, sw_cats, sw_temp


def check_coordinates(dstime):
    """Check if coordinates are stacked on cell dimension.

    Parameters
    ----------
    dstime: xarray Dataset
         Based on intermediate dataset returned by detect(), includes
         original ts and climatologies (optional) along 'time' dimension


    Returns
    -------
    dstime: xarray Dataset
        Same as input after applying land_check if necessary
    stack_coord: str
        Name of stacked dimension
    """

    # find name of time dimension
    # If there is already a stacked dimension skip land_check
    check = True
    # now that we use a stacked array without creating an index 
    # the stacked coord is a dimension without coordinates
    # and its type is int64 not anymore object
    ds_coords = list(dstime.dims)
    # if only 1 dimension assumes is time and set stack_coords to point
    if len(ds_coords) == 1:
        stack_coord = 'point'
        check = False
    for x in ds_coords:
        dtype = str(dstime[x].dtype)
        if dtype == "int64":
            stack_coord = x
            check = False
        elif "datetime" in dtype:
            tdim = x
            print(f"Assuming {tdim} is time dimension")
    if check:
        dstime = land_check(dstime, tdim=tdim)
        stack_coord = "cell"
    return dstime, stack_coord


@dask.delayed
def call_groupby(ds, tgroup, bins, mode="mhw"):
    """Call groupby on mhw results cell by cell using the specified
    aggregation dictionary.

    Parameters
    ----------
    ds: xarray Dataset
        mhw dataset for 1 grid point
    tgroup: xarray DataArray
        Years of mhw time variable to use to assign events to blocks
    bins: list(int)
        Intervals to use to define blocks
    mode: str, optional
        Defines which agg_<mode> function to call (default is 'mhw')
    """

    # convert mhw Dataset to Dataframe
    df = ds.to_dataframe()
    # groupby mtime and aggregate variables
    fagg = "agg_" + mode
    dfblock = globals()[fagg](df, tgroup, bins)
    # add total number of actual MHW days per block
    if fagg == "agg_cats":
        dfblock["total_days"] = (
            dfblock["moderate_days"]
            + dfblock["strong_days"]
            + dfblock["severe_days"]
            + dfblock["extreme_days"]
        )
    # convert Catgorical index to normal index
    dfblock.index = dfblock.index.to_list()
    dfblock.index.name = "years"
    # convert back to xarray dataset
    block = xr.Dataset.from_dataframe(dfblock, sparse=False)
    return block


def agg_mhw(df, tgroup, bins):
    """Apply groupby on mhw properties dataframe after defining an
    aggregation dictionary.

    Parameters
    ----------
    df: pandas Dataframe
        Dataframe including mhw properties for 1 grid point
    tgroup: xarray DataArray
        Years of mhw time variable to use to assign events to blocks
    bins: list(int)
        Intervals to use to define blocks

    Returns
    -------
    dfgroup: pandas Dataframe
        MHW porperties aggregated by mtime following dictionary
    """

    # first use pandas.cut to separate datFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    dfgroup = df.groupby(dfbins).agg(
        ecount=("event", "count"),
        duration=("duration", "mean"),
        intensity_max=("intensity_max", "mean"),
        intensity_max_max=("intensity_max", "max"),
        intensity_mean=("intensity_mean", "mean"),
        intensity_cumulative=("intensity_cumulative", "mean"),
        total_icum=("intensity_cumulative", "sum"),
        intensity_mean_relThresh=("intensity_mean_relThresh", "mean"),
        intensity_cumulative_relThresh=(
            "intensity_cumulative_relThresh",
            "mean",
        ),
        severity_mean=("severity_mean", "mean"),
        severity_cumulative=("severity_cumulative", "mean"),
        intensity_mean_abs=("intensity_mean", "mean"),
        intensity_cumulative_abs=("intensity_cumulative", "mean"),
        rate_onset=("rate_onset", "mean"),
        rate_decline=("rate_decline", "mean"),
    )
    return dfgroup


def cat_days(series, cat=1):
    """Return count of days where category == cat"""
    return series[series == cat].count()


def agg_cats(df, tgroup, bins):
    """Apply groupby on timeseries and categories dataframe after
    defining an aggregation dictionary.

    Parameters
    ----------
    df: pandas Dataframe
        Dataframe including ts and categories for 1 grid point
    tgroup: xarray DataArray
        Years of mhw time variable to use to assign events to blocks
    bins: list(int)
        Intervals to use to define blocks

    Returns
    -------
    dfgroup: pandas Dataframe
        Timeseries and categories aggregated by mtime following dictionary
    """

    # first use pandas.cut to separate dataFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    dfgroup = df.groupby(dfbins).agg(
        ts_mean=("ts", "mean"),
        ts_max=("ts", "max"),
        ts_min=("ts", "min"),
        moderate_days=("cats", lambda x: cat_days(x, 1)),
        strong_days=("cats", lambda x: cat_days(x, 2)),
        severe_days=("cats", lambda x: cat_days(x, 3)),
        extreme_days=("cats", lambda x: cat_days(x, 4)),
    )
    return dfgroup


def agg_ts(df, tgroup, bins):
    """Apply groupby on ts dataframe after defining an aggregation
    dictionary.

    Parameters
    ----------
    df: pandas Dataframe
        Dataframe including timeseries for 1 grid point
    tgroup: xarray DataArray
        Years of mhw time variable to use to assign events to blocks
    bins: list(int)
        Intervals to use to define blocks

    Returns
    -------
    dfgroup: pandas Dataframe
        Timeseries aggregated by mtime following dictionary
    """

    # first use pandas.cut to separate datFrame in bins
    dfbins = pd.cut(tgroup, bins, right=False)
    dfgroup = df.groupby(dfbins).agg(
        ts_mean=("ts", "mean"), ts_max=("ts", "max"), ts_min=("ts", "min")
    )
    return dfgroup


def find_across(mhw):
    """Find all events that span across a year."""
    mhw_span = mhw.where(
        mhw["time_start"].dt.year != mhw["time_end"].dt.year
    ).dropna(dim="events")
    return mhw_span


def split_event(mhw_ev):
    """If event span across two years you might want to split it into two
    and calculate separate stats.
    """
    return mhw_ev


def mhw_rank(mhwds):
    """Rank mhw on each properties going from largest to smallest
    (1,.., n-events).

    Calculate the rank and return periods of marine heatwaves (MHWs)
    according to each metric.
    Any date or index variable is ignored.

    This function assumes that the MHWs were calculated over a suitably
    long record that return periods make sense. If the record length is
    a few years or less than this becomes meaningless.

    Parameters
    ----------
    mhwds: xarray Dataset
        Includes MHW properties by events

    Returns
    -------
    rank
        The rank of each MHW according to each MHW property. A rank
        of 1 is the largest, 2 is the 2nd largest, etc. Each key
        (listed below) is a list of length N where N is the number of MHWs.
    returnPeriod
        The return period (in years) of each MHW according to each MHW property
        The return period signifies, statistically, the recurrence interval for
        an event at least as large/long as the event in question. Each key
        (listed below) is a list of length N where N is the number of MHWs.
    """

    # should be based on calendar
    days_year = 365.25
    nYears = 14245 / days_year
    rank = xr.Dataset()
    return_period = xr.Dataset()
    # skip index and time variables
    variables = [
        k
        for k in mhwds.keys()
        if not any(x in k for x in ["event", "time", "index"])
    ]
    for var in variables:
        rank[var] = rank_variable(mhwds[var])
        return_period[var] = (nYears + 1) / rank[var]
    return rank, return_period


def rank_variable(array):
    """Rank an array assigning values 1,2,3 ... to array elements going
    from biggest to smallest.

    Parameters
    ----------
    array: xarray DataArray
        input variable

    Returns
    -------
    rank:
        a rank order array
    """
    rank_values = len(array) - array.values.argsort().argsort()
    rank = xr.full_like(array, np.nan)
    rank[:] = rank_values
    return rank
