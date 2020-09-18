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

from xmhw.helpers import (land_check, add_doy, window_roll, mhw_filter, feb29, 
     get_peak, index_cat, cat_duration, group_function, join_gaps, join_events) 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest

def test_add_doy(oisst_ts, oisst_doy):
    doy = add_doy(oisst_ts, tdim="time").doy.values 
    nptest.assert_array_equal(doy, oisst_doy) 

def test_feb29(oisst_ts):
    # this is testing for feb29 averaging 28 Feb and 1st of march I believe it should i nclude 29 Feb too!
    ts = add_doy(oisst_ts, tdim="time")
    a =np.array([18.2074995])
    b = feb29(ts, dim='time')
    nptest.assert_almost_equal(a, b[1,2], decimal=5) 

def test_runavg():
#(ts, w):
    assert True

def test_window_roll(oisst_ts, tstack):
    ts = oisst_ts.sel(time=slice('2003-01-01','2003-01-03'),lat=-42.625, lon=148.125)
    array = window_roll(ts, 1, 'time')
    #assert array.z.index
    nptest.assert_almost_equal(array.values, tstack, decimal=5)

def test_dask_percentile():
#(array, axis, q):
    assert True

def test_join_gaps(mhwfilter):
    exceed, st, en, evs, st2, en2, evs2 = mhwfilter 
    ds = xr.Dataset({'start': st, 'end':en, 'events': evs})
    # testing with maxGap=3 should return identical dataset
    ds2 = join_gaps(ds, 3)
    xrtest.assert_equal(ds.start, ds2.start)
    xrtest.assert_equal(ds.end, ds2.end)
    xrtest.assert_equal(ds.events, ds2.events)
    # testing with gap 2 should join two events
    ds = xr.Dataset({'start': st, 'end':en, 'events': evs})
    ds3 = join_gaps(ds, 2)
    xrtest.assert_equal(st2, ds3.start)
    xrtest.assert_equal(en2, ds3.end)
    xrtest.assert_equal(evs2, ds3.events)
    # testing only last two events to make sure it works with array len 1
    ds.start[5] = np.nan
    ds.end[5] = np.nan
    ds.events[1:6] = np.nan
    ds4 = join_gaps(ds, 2)
    xrtest.assert_equal(evs2[10:], ds4.events[10:])

def test_group_function():
    # This is only testing the option where no extxra argument is passed
    event = np.arange(20)
    event[:3] = 1
    event[3:10] = 2
    event[10:] = 3 
    a = xr.DataArray(np.arange(20), dims=["event"], coords={"event": event})
    amax = xr.DataArray([2.,9.,19.], dims=["event"], coords={"event": [1,2,3]})
    xrtest.assert_equal( group_function(a, np.max), amax)

def test_mhw_filter(mhwfilter):
    # These tests only check on 1 D to make sure it work on 2 d add extra tests
    exceed, st, en, evs, st2, en2, evs2 = mhwfilter 
    # test with joinGaps=False
    ds = mhw_filter(exceed, 5, joinGaps=False)
    xrtest.assert_equal( ds.start, st)
    xrtest.assert_equal( ds.end, en)
    xrtest.assert_equal( ds.events, evs)
    # test with default joinGaps True and maxGaps=2, join 2nd and 3rd events
    ds2 = mhw_filter(exceed, 5)
    xrtest.assert_equal( ds2.start, st2)
    xrtest.assert_equal( ds2.end, en2)
    xrtest.assert_equal( ds2.events, evs2)

def test_index_cat():
    a = np.array([0,2,0,2,3,4])
    b = np.array([0,2,0,2,3,1])
    assert index_cat(a, 0) == 3 
    assert index_cat(b, 0) == 2 

def test_get_peak():
#(array, axis):
    assert True

def test_cat_duration():
    a = np.array([1,2,1,1,3,2,1])
    assert cat_duration(a,0,arg=1) == 4  
    assert cat_duration(a,0,arg=2) == 2  
    assert cat_duration(a,0,arg=3) == 1  
    assert cat_duration(a,0,arg=4) == 0  

def test_mhw_ds():
    assert True

def test_categories():
    assert True

def test_join_events():
    evs = xr.DataArray(np.arange(20))
    evs2 = evs.copy()
    evs2[1:8] = 1
    evs2[12:19] = 12
    joined = set([(1,7),(12,18)])
    evs3 = join_events(evs, joined)
    xrtest.assert_equal(evs2, evs3)
    assert True

def test_land_check(oisst_ts, landgrid):
    # should add test with timeseries with different dimension names
    newts = land_check(oisst_ts)
    assert newts.shape == (731, 12)
    with pytest.raises(XmhwException):
        land_check(landgrid)
