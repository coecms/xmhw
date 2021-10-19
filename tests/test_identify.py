#!/usr/bin/env python
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

#import pytest

from xmhw.identify import (land_check, add_doy, window_roll, runavg, define_events,
                           mhw_filter, feb29, get_calendar, join_gaps, join_events) 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest
import pandas.testing as pdtest


def test_add_doy(oisst_ts, oisst_doy):
    doy = add_doy(oisst_ts, tdim="time").doy.values 
    nptest.assert_array_equal(doy, oisst_doy) 


def test_feb29(oisst_ts):
    ts = add_doy(oisst_ts, tdim="time")
    # this is testing for feb29 averaging 28 Feb and 1st of march only
    #a =np.array([18.2074995])
    # this is testing for feb29 averaging 28, 29 Feb and 1st of march 
    a =np.array([18.13])
    b = feb29(ts, dim='time')
    nptest.assert_almost_equal(a, b[1,2], decimal=5) 


def test_runavg():
    a = xr.DataArray([1,2,2,4,3,2], dims=['doy'], coords=[np.array([1,2,3,4,5,6])])
    b = runavg(a, 3).compute()
    nptest.assert_almost_equal(b.values, np.array([1.66667, 1.66667, 2.66667, 3., 3., 2.]), decimal=5)
    c = runavg(a, 5).compute()
    nptest.assert_almost_equal(c.values, np.array([2. , 2.2, 2.4, 2.6, 2.4, 2.4]), decimal=5)
    with pytest.raises(XmhwException):
        runavg(a, 2).compute()


def test_window_roll(oisst_ts, tstack):
    ts = oisst_ts.sel(time=slice('2003-01-01','2003-01-03'),lat=-42.625, lon=148.125)
    array = window_roll(ts, 1, 'time')
    #assert array.z.index
    nptest.assert_almost_equal(array.values, tstack, decimal=5)


def test_join_gaps(filter_data):
    bthresh, idxarr, st, end, evs, st2, end2, evs2 = filter_data 
    # testing with maxGap=2 should return identical dataset
    df = join_gaps(st, end, evs, 2)
    pdtest.assert_series_equal(df.start, st)
    pdtest.assert_series_equal(df.end, end)
    pdtest.assert_series_equal(df.events, evs)
    # testing with gap 3 should join two events
    df2 = join_gaps(st, end, evs, 3)
    pdtest.assert_series_equal(df2.start, st2)
    pdtest.assert_series_equal(df2.end, end2)
    pdtest.assert_series_equal(df2.events, evs2)
    # testing only last two events to make sure it works with array len 1
    st[5] = np.nan
    end[5] = np.nan
    evs[1:6] = np.nan
    df3 = join_gaps(st, end, evs, 3)
    pdtest.assert_series_equal(df3.events[10:], evs2[10:])


def test_mhw_filter(filter_data):
    # These tests only check on 1 D to make sure it work on 2 d add extra tests
    bthresh, idxarr, st, end, evs, st2, end2, evs2 = filter_data 
    # test with joinGaps=False
    df = mhw_filter(bthresh, idxarr, 5, False)
    pdtest.assert_series_equal( df.start, st)
    pdtest.assert_series_equal( df.end, end)
    pdtest.assert_series_equal( df.events, evs)
    # test with default joinGaps True and maxGaps=3, join 2nd and 3rd events
    df2 = mhw_filter(bthresh, idxarr, 5, True, 3)
    pdtest.assert_series_equal( df2.start, st2)
    pdtest.assert_series_equal( df2.end, end2)
    pdtest.assert_series_equal( df2.events, evs2)


def test_join_events(join_data):
    evs, evs2, joined = join_data
    evs3 = join_events(evs, joined)
    pdtest.assert_series_equal(evs2, evs3)
    assert True


def test_land_check(oisst_ts, clim_oisst, landgrid):
    newts = land_check(oisst_ts)
    assert newts.shape == (731, 12)
    # test timeseries with with only few nans in a lat/lon cell 
    # both for removeNans=False (default) and removeNans=True
    fewnans = oisst_ts.copy()
    fewnans[245,1,2] = np.nan
    newts = land_check(fewnans, removeNans=True)
    assert newts.shape == (731, 11)
    newts = land_check(fewnans)
    assert newts.shape == (731, 12)
    # test timeseries with different dimension names
    diffdim = oisst_ts.rename({'lat': 'a', 'lon': 'b', 'time': 'c'})
    newts = land_check(diffdim, tdim='c')
    assert newts.shape == (731, 12)
    newts = land_check(clim_oisst.thresh1)
    assert newts.shape == (366, 1)
    # test exception raised when all points are land 
    with pytest.raises(XmhwException):
        land_check(landgrid)
    # test exception raised when one of the dimension to stack has lenght 0 
    with pytest.raises(XmhwException):
        land_check(oisst_ts.sel(lat=slice(-41,-41.5)))

def test_define_events(define_data, mhw_data, inter_data):
    # test define events return two datasets if intermediate is True
    ts, th, se, idxarr = define_data
    mhwds = mhw_data
    interds = inter_data
    res = define_events(ts.isel(cell=0), th.isel(cell=0), se.isel(cell=0),
            idxarr, 5, True, 2, True)
    results = res.compute()
    print('from funct', results[1])
    print('interds', interds)
    xrtest.assert_allclose(results[0], mhwds)
    xrtest.assert_allclose(results[1], interds)

    # test define events return one dataset only if intemediate is False, as default
    res = define_events(ts.isel(cell=0), th.isel(cell=0), se.isel(cell=0),
            idxarr, 5, True, 2, False)
    results = res.compute()
    xrtest.assert_allclose(results[0], mhwds)
    assert results[1] is None


def test_annotate_ds():
    pass

def test_get_calendar(calendars):
    noleap, all_leap, day_365, day_366, gregorian, standard, julian, proleptic, ndays_year = calendars 
    del calendars
    # test retrieving calendar attribute
    cal_list =  locals()
    cal_list.pop('ndays_year')
    for calendar,timerange in cal_list.items():
        var = xr.DataArray(coords={'time': timerange}, dims=['time'])
        assert get_calendar(var.time) == ndays_year[calendar]
    # test guessing number of days per year if calendar not present
    # test working with different calendars
