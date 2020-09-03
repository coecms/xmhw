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

from xmhw.helpers import land_check, get_doy, window_roll, feb29 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest

def test_get_doy(oisst_ts, oisst_doy):
    doy = get_doy(oisst_ts,dim="time").doy.values 
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

def test_mhw_filter():
#(exceed, minDuration, joinGaps, maxGap):
    assert True

def test_sqrt_var():
#(array, axis):
    assert True

def test_cat_min():
#(array, axis):
    assert True

def test_group_argmax():
#(array):
    assert True

def test_land_check(oisst_ts, landgrid):
    newts = land_check(oisst_ts)
    assert newts.shape == (731, 12)
    with pytest.raises(XmhwException):
        land_check(landgrid)
