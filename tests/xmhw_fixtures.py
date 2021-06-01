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

import pytest
import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime


TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")
# oisst data from 2003 to 2004 included for small region
oisst = os.path.join(TESTS_DATA, "oisst_2003_2004.nc")
# oisst data from 2003 to 2004 included for all land region
land = os.path.join(TESTS_DATA, "land.nc")
# threshold and seasonal avg calculated using Eric Olivier MHW code on two points of OISST region subset for same period 2003-2004
# point1 lat=-42.625, lon=148.125
# point2 lat=-41.625, lon=148.375
oisst_clim = os.path.join(TESTS_DATA,"test_clim_oisst.nc")
oisst_clim_nosmooth = os.path.join(TESTS_DATA,"test_clim_oisst_nosmooth.nc")
relthreshnorm = os.path.join(TESTS_DATA, "relthreshnorm.nc")


@pytest.fixture(scope="module")
def oisst_ts():
    ds = xr.open_dataset(oisst)
    return ds.sst 


@pytest.fixture(scope="module")
def landgrid():
    ds = xr.open_dataset(land)
    return ds.sst 


@pytest.fixture(scope="module")
def clim_oisst():
    ds = xr.open_dataset(oisst_clim)
    return ds 


@pytest.fixture(scope="module")
def clim_oisst_nosmooth():
    ds = xr.open_dataset(oisst_clim_nosmooth)
    return ds 


@pytest.fixture(scope="module")
def dsnorm():
    ds = xr.open_dataset(relthreshnorm)
    return ds.stack(cell=['lat','lon']) 


@pytest.fixture
def oisst_doy():
    a = np.arange(1,367)
    b = np.delete(a,[59])
    return np.concatenate((b,a)) 


@pytest.fixture
def tstack():
    return np.array([ np.nan, 16.99, 17.39, 16.99, 17.39, 17.3 , 17.39, 17.3 , np.nan])


@pytest.fixture
def filter_data():
    a = [0,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0]
    time =  pd.date_range('2001-01-01', periods=len(a))
    array = pd.Series(a, index=time)
    idxarr = pd.Series(data=np.arange(len(a)), index=time)
    bthresh = array==1
    st = pd.Series(index=time, dtype='float64').rename('start')
    end = pd.Series(index=time, dtype='float64').rename('end')
    events = pd.Series(index=time, dtype='float64').rename('events')
    st[5] = 1
    st[16] = 11
    st[24] = 20
    end[5] = 5
    end[16] = 16
    end[24] = 24
    events[1:6] = 1
    events[11:17] = 11
    events[20:25] =20 
    st2 = st.copy()
    end2 = end.copy()
    events2 = events.copy()
    st2[24] = np.nan
    end2[16] = np.nan
    events2[17:25] = 11
    return (bthresh, idxarr, st, end, events, st2, end2, events2) 

@pytest.fixture
def join_data():
    evs = pd.Series(np.arange(20)).rename('events')
    evs2 = evs.copy()
    evs2[1:8] = 1
    evs2[12:19] = 12
    joined = set([(1,7),(12,18)])
    return (evs, evs2, joined)


@pytest.fixture
def rates_data():
    d = { 'index_start': [3.], 'index_end': [10.], 'index_peak': [8.],
            'relS_first': [2.3], 'relS_last': [1.8], 'intensity_max': [3.1], 
            'anom_first': [0.3], 'anom_last': [0.2]}
    df = pd.DataFrame(d)
    return [df, np.array([0.21176470]), np.array([-4.20])]


@pytest.fixture
def define_data():
    # create 1-D dataset to pass to define_events
    time = pd.date_range('2001-01-01', periods=9)
    ds = xr.Dataset()
    ds['ts'] = xr.DataArray(data=[15.6, 17.3, 18.2, 19.5, 19.4, 19.6, 18.1, 17.0, 15.2],
                  dims=['time'], coords={'time': time})
    ds['seas'] = xr.DataArray(data=[15.8, 16.0, 16.2, 16.5, 16.6, 16.4, 16.6, 16.7, 16.4],
                    dims=['time'], coords={'time': time})
    ds['thresh'] = xr.DataArray([16.0, 16.7, 17.6, 17.9, 18.1, 18.2, 17.3, 17.2, 17.0],
                            dims=['time'], coords={'time': time})
    ds['bthresh'] = ds['ts'] > ds['thresh']
    ds = ds.expand_dims(['lat','lon'])
    ds = ds.stack(cell=(['lat','lon']))
    # Build a pandas series with the positional indexes as values
    # [0,1,2,3,4,5,6,7,8,9,10,..]
    idxarr = pd.Series(data=np.arange(9), index=ds.time.values)
    return (ds, idxarr)


@pytest.fixture
def mhw_data():
    # generate a mhw dataset as returned by define_mhw based on define_data
    mhwds = xr.Dataset(coords={'events': [1]})
    vars_dict = {'event': [1.0],
        'index_start': [1.0],
        'index_end': [6.0],
        'time_start': [datetime.datetime(2001, 1, 2, 0, 0)],
        'time_end': [datetime.datetime(2001, 1, 7, 0, 0)],
        'time_peak': [datetime.datetime(2001, 1, 6, 0, 0)],
        'intensity_max': [3.2],
        'intensity_mean': [2.3],
        'intensity_cumulative': [13.8],
        'severity_max': [-1.42857],
        'severity_mean': [-1.86931],
        'severity_cumulative': [-11.215873],
        'severity_var': [0.265495],
        'intensity_mean_relThresh': [1.05],
        'intensity_cumulative_relThresh': [6.30],
        'intensity_mean_abs': [18.6834],
        'intensity_cumulative_abs': [112.1],
        'duration_moderate': [4],
        'duration_strong': [2],
        'duration_severe': [0],
        'duration_extreme': [0],
        'index_peak': [5.0],
        'intensity_var': [0.809938],
        'intensity_max_relThresh': [1.40],
        'intensity_max_abs': [19.6],
        'intensity_var_relThresh': [0.437035],
        'intensity_var_abs': [0.9495613],
        'category': [2.0],
        'duration': [6.0],
        'rate_onset': [0.5888889],
        'rate_decline': [1.5333333]}
    for k,v in vars_dict.items():
        mhwds[k] = xr.DataArray(data=v, dims=['events'], coords={'events': [1]})
    return mhwds


@pytest.fixture
def inter_data():
    # generate intermediate data as returned by define_events with intermediate True and define_data as input
    index = pd.date_range('2001-01-01', periods=9)
    ids = xr.Dataset(coords={'index': index})
    vars_dict = {'ts': [15.6, 17.3, 18.2, 19.5, 19.4, 19.6, 18.1, 17.0, 15.2],
        'seas': [np.nan, 16.0, 16.2, 16.5, 16.6, 16.4, 16.6, np.nan, np.nan],
        'thresh': [np.nan, 16.7, 17.6, 17.9, 18.1, 18.2, 17.3, np.nan, np.nan],
        'events': [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        'relSeas': [np.nan, 1.3, 2.0, 3.0, 2.79999, 3.2, 1.5, np.nan, np.nan],
        'relThresh': [np.nan, 0.6, 0.6, 1.6, 1.3, 1.4, 0.8, np.nan, np.nan],
        'relThreshNorm': [np.nan, 0.85714, 0.4285714, 1.142857, 0.866667, 0.77778, 1.142857, np.nan, np.nan],
        'severity': [np.nan, -1.857143, -1.42857, -2.142857, -1.8666667, -1.77778, -2.142857, np.nan, np.nan],
        'cats': [np.nan, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, np.nan, np.nan],
        'duration_moderate': [False, True, True, False, True, True, False, False, False],
        'duration_strong': [False, False, False, True, False, False, True, False, False],
        'duration_severe': [False]*9,
        'duration_extreme': [False]*9,
        'mabs': [np.nan, 17.3, 18.2, 19.5, 19.4, 19.6, 18.1, np.nan, np.nan]}
    for k,v in vars_dict.items():
        ids[k] = xr.DataArray(data=v, dims=['index'], coords={'index': index})
    return ids
