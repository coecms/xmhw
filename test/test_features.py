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


from xmhw.features import (mhw_df, properties, unique_dropna, agg_df, get_rate,
                           get_period, get_edge, onset_decline, mhw_features, flip_cold) 
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest


def test_mhw_features():
    # testing to check that with all-nan array calculation skipped and np.nan assigned
    df = pd.DataFrame()
    #for var in ['start','end','anom_plus', 'anom_minus', 'seas', 'ts',
    #        'thresh', 'events', 'relThresh', 'relSeas', 'relThreshNorm', 'mabs']:
    #    df[var] = pd.Series([np.nan, np.nan, np.nan], dims=['time'], coords=[np.arange(3)])
    #df = mhw_features(df,  321)
    #assert np.isnan( df.start_idx.values )
    pass


def test_onset_decline(rates_data):
    # testing that with  all-nan array calculations are skipped and np.nan are assigned
    df, onset, decline = rates_data
    df = onset_decline(df, 321)
    nptest.assert_almost_equal(df.rate_decline.values, decline)
    nptest.assert_almost_equal(df.rate_onset.values, onset)


def test_get_edge():
    d = {'c1': [2.3, 2.3], 'c2': [1.7, 1.7], 'c3': [2,0]}
    df = pd.DataFrame(data=d)
    # test for idx != edge on 1st row and idx=edge on 2nd row 
    edges = get_edge(df['c1'], df['c2'], df['c3'], 0) 
    assert edges[0] == 2.0
    assert edges[1] == 2.3


def test_get_period():
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


def test_flip_cold():
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


def test_unique_dropna():
    pass


def test_agg_df():
    pass


def test_properties():
    pass
