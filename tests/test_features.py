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


from xmhw.features import (group_function, mhw_ds, index_cat, cat_duration, categories,
                           get_rate, get_period, get_edge, onset_decline, mhw_features) 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest


def test_group_function():
    # This is only testing the option where no extxra argument is passed
    event = np.arange(20)
    event[:3] = 1
    event[3:10] = 2
    event[10:] = 3 
    a = xr.DataArray(np.arange(20), dims=["event"], coords={"event": event})
    amax = xr.DataArray([2.,9.,19.], dims=["event"], coords={"event": [1,2,3]})
    xrtest.assert_equal( group_function(a, np.max), amax)

def test_index_cat():
    a = np.array([0,2,0,2,3,4,6])
    b = np.array([0,2,0,2,3,1,1])
    assert index_cat(a) == 4 
    assert index_cat(b) == 3 


def test_cat_duration():
    a = np.array([1,2,1,1,3,2,1])
    assert cat_duration(a,arg=1) == 4  
    assert cat_duration(a,arg=2) == 2  
    assert cat_duration(a,arg=3) == 1  
    assert cat_duration(a,arg=4) == 0  


def test_mhw_features():
    # testing to check that with all-nan array calculation skipped and np.nan assigned
    ds = xr.Dataset()
    for var in ['start','end','anom_plus', 'anom_minus', 'seas', 'ts',
            'thresh', 'events', 'relThresh', 'relSeas', 'relThreshNorm', 'mabs']:
        ds[var] = xr.DataArray([np.nan, np.nan, np.nan], dims=['time'], coords=[np.arange(3)])
    ds = mhw_features(ds, 'time', 321)
    assert np.isnan( ds.start_idx.values )


def test_categories():
    ds = xr.Dataset()
    ds['relThreshNorm'] = xr.DataArray([1.2, 0.9, 2.3, 1.5, 0.8, 0.7, 1.6, 2.1])
    ds = categories(ds)
    nptest.assert_array_equal(ds['category'], np.array(['Severe']) )
    #nptest.assert_array_equal(ds['category'], np.array([3.])) 
    nptest.assert_array_equal( ds['duration_moderate'], np.array([3.]) )
    nptest.assert_array_equal( ds['duration_strong'], np.array([3.]) )
    nptest.assert_array_equal( ds['duration_severe'], np.array([2.]) )
    nptest.assert_array_equal( ds['duration_extreme'], np.array([0.]) )


def test_onset_decline(rates):
    # testing that with  all-nan array calculations are skipped and np.nan are assigned
    ds, onset, decline = rates
    ds = onset_decline(ds, 321)
    nptest.assert_almost_equal(ds.rate_decline.values, decline)
    nptest.assert_almost_equal(ds.rate_onset.values, onset)


def test_get_edge():
    # test for idx != edge 
    assert get_edge(2.3, 1.7, 2, 0) == 2.0
    # test for idx == edge
    assert get_edge(2.3, 1.7, 0, 0) == 2.3


def test_get_period():
#    def get_period(start, end, peak, tsend):
    start = xr.DataArray([0, 8, 18])
    end =  xr.DataArray([4, 15, 25])
    # test where start=peak=0
    peak1 =  xr.DataArray([0, 10, 19])
    ons1 = xr.DataArray([1, 10.5, 19.5])
    dec1 = xr.DataArray([4.5, -2.5, -12])
    ons, dec = get_period(start, end, peak1, 25)
    xrtest.assert_equal( ons, ons1 )
    xrtest.assert_equal( dec, dec1 )
    # test where end=peak=len(ts)
    peak2 =  xr.DataArray([3, 15, 25])
    ons2= xr.DataArray([3., 15.5, 25.5])
    dec2 = xr.DataArray([1.5, -7.5, 1.])
    ons, dec = get_period(start, end, peak2, 25)
    xrtest.assert_equal( ons, ons2 )
    xrtest.assert_equal( dec, dec2 )

def test_get_rate():
    edge = xr.DataArray([1. , 1.5, 2.5])
    period = xr.DataArray([1, 10.5, 19.5])
    peak = xr.DataArray([1.4, 2.4, 1.8])
    result =  xr.DataArray([0.4, 0.08571429, -0.03589744])
    xrtest.assert_allclose( result, get_rate(peak, edge, period))


def flip_cold():
    ds = xr.Dataset()
    y = xr.DataArray([1., 2., np.nan], dims=['x'], coords=[np.arange(3)])
    z = xr.DataArray([-1., -2., np.nan], dims=['x'], coords=[np.arange(3)])
    ds['intensity_sum_dummy'] = y
    ds['intensity_var_dummy'] = y 
    ds['dummy'] = y 
    ds2 = flip_cold(ds)
    xrtest.assert_equal(ds2['intensity_sum_dummy'], z) 
    xrtest.assert_equal(ds2['intensity_var_dummy'], y) 
    xrtest.assert_equal(ds2['dummy'], y) 

