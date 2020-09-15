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

from xmhw.helpers import land_check, add_doy, window_roll, mhw_filter, feb29 
from xmhw.helpers import get_peak, index_cat, cat_duration, group_function 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest

def test_add_doy(oisst_ts, oisst_doy):
    doy = add_doy(oisst_ts,dim="time").doy.values 
    nptest.assert_array_equal(doy, oisst_doy) 

def test_feb29():
#(ts):
    assert True

def test_runavg():
#(ts, w):
    assert True

def test_window_roll(oisst_ts, tstack):
    ts = oisst_ts.sel(time=slice('2003-01-01','2003-01-03'),lat=-42.625, lon=148.125)
    array = window_roll(ts, 1)
    #assert array.z.index
    nptest.assert_almost_equal(array.values, tstack, decimal=5)

def test_dask_percentile():
#(array, axis, q):
    assert True

def test_join_gaps():
#(ds, maxGap):
    assert True

def test_group_function():
    # This is only testing the option where no etxra argument is passed
    event = np.arange(20)
    event[:3] = 1
    event[3:10] = 2
    event[10:] = 3 
    a = xr.DataArray(np.arange(20), dims=["event"], coords={"event": event})
    amax = xr.DataArray([2.,9.,19.], dims=["event"], coords={"event": [1,2,3]})
    xrtest.assert_equal( group_function(a, np.max), amax)

def test_mhw_filter(mhwfilter):
    # These tests only check on 1 D to make sure it work on 2 d add extra tests
    exceed, st, en, evs = mhwfilter
    # test with joinGaps=False
    ds = mhw_filter(exceed, 5, joinGaps=False)
    xrtest.assert_equal( ds.start, st)
    xrtest.assert_equal( ds.end, en)
    xrtest.assert_equal( ds.events, evs)
    # test with default joinGaps True and maxGaps=2, join 2nd and 3rd events
    ds = mhw_filter(exceed, 5)
    st[24] = np.nan
    en[17] = np.nan
    evs[18:25] = 11
    xrtest.assert_equal( ds.start, st)
    xrtest.assert_equal( ds.end, en)
    xrtest.assert_equal( ds.events, evs)

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
    assert cat_duration(a,arg=1) == 4  
    assert cat_duration(a,arg=1) == 2  
    assert cat_duration(a,arg=2) == 1  
    assert cat_duration(a,arg=3) == 0  

def test_mhw_ds():
    assert True

def test_categories():
    assert True

def test_join_events():
#(array):
    assert True

def test_land_check(oisst_ts, landgrid):
    newts = land_check(oisst_ts)
    assert newts.shape == (731, 12)
    with pytest.raises(XmhwException):
        land_check(landgrid)
