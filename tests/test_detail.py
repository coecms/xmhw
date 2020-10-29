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


from xmhw.detail import (group_function, index_cat, cat_duration, categories,
                         get_peak, get_rate, get_period, get_edge, onset_decline) 
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
    assert index_cat(a, 0) == 4 
    assert index_cat(b, 0) == 3 

def test_get_peak():
    evs = np.array([np.nan,1,1,1,1,1,np.nan,np.nan,2,2,2,2,np.nan,3,3,3,
                   np.nan, 4,4,4,4,4, np.nan])
    a = xr.DataArray(np.arange(23), dims=['evs'], coords=[evs])
    b = a + 0.5
    peak = xr.DataArray([3,8,np.nan,18], dims=['ev'], coords=[np.array([1,2,3,4])])
    ds =xr.Dataset({'a': a, 'b':b, 'peak': peak})
    peaks = get_peak(ds, ['a','b'], dim='ev')
    nptest.assert_equal(peaks['a'].values, np.array([3,8,np.nan,18])) 
    nptest.assert_equal(peaks['b'].values, np.array([3.5,8.5,np.nan,18.5])) 

def test_cat_duration():
    a = np.array([1,2,1,1,3,2,1])
    assert cat_duration(a,0,arg=1) == 4  
    assert cat_duration(a,0,arg=2) == 2  
    assert cat_duration(a,0,arg=3) == 1  
    assert cat_duration(a,0,arg=4) == 0  

def test_mhw_ds():
    assert True

def test_categories(dsnorm):
    ds = categories(dsnorm, dsnorm['relThreshNorm'])
    nptest.assert_array_equal(ds['category'][0,:], np.array(['Moderate', '0.0',
           'Strong', '0.0', 'Moderate', 'Moderate', 'Strong', '0.0', '0.0'])) 
    nptest.assert_array_equal( ds['duration_moderate'][0,:], np.array([5., np.nan,
                               2., np.nan,  5.,  5., 16., np.nan, np.nan]))
    nptest.assert_array_equal( ds['duration_strong'][0,:], np.array([0., np.nan,
                               3., np.nan,  0.,  0., 1., np.nan, np.nan]))
    nptest.assert_array_equal( ds['duration_severe'][0,:], np.array([0., np.nan,
                               0., np.nan,  0.,  0., 0., np.nan, np.nan]))
    nptest.assert_array_equal( ds['duration_extreme'][0,:], np.array([0., np.nan,
                               0., np.nan,  0.,  0., 0., np.nan, np.nan]))


def test_onset_decline():
    ds = xr.Dataset()
    ds['start_idx'] = xr.DataArray([np.nan, np.nan, np.nan], dims=['event'], coords=[np.arange(3)])
    ds = onset_decline(ds)
    xrtest.assert_equal(ds.start_idx, ds.rate_decline)
    xrtest.assert_equal(ds.start_idx, ds.rate_onset)

def test_get_edge():
    relSeas = xr.DataArray(np.arange(1,20))
    anom = xr.DataArray(np.arange(21,40))
    # use idx for onset and idx+1 for decline so you test respectively start
    # edge for onset and end edge for decline but avoid start for decline and
    # end for onset since they aren't valid values
    idx = xr.DataArray([0, 5, 17])
    onset = xr.DataArray([1. , 15.5, 27.5])
    decline = xr.DataArray([12.5 , 17.5, 19.0])
    #for onset edge = 0 step = +1, relSeas=relSeas[0]
    xrtest.assert_equal( get_edge(relSeas, anom, idx, 0, 1),
                               onset)
    #for decline edge = len(ts)-1 and step = -1, relSeas=relSeas[-1]
    xrtest.assert_equal( get_edge(relSeas, anom, idx+1, 18, -1),
                               decline)

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

