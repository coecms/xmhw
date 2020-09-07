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
def oisst_doy():
    a = np.arange(1,367)
    b = np.delete(a,[59])
    return np.concatenate((b,a)) 

@pytest.fixture(scope="module")
def tstack():
    return np.array([ np.nan, 16.99, 17.39, 16.99, 17.39, 17.3 , 17.39, 17.3 , np.nan])

@pytest.fixture(scope="module")
def mhwfilter():
    a = [0,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0]
    time =  pd.date_range('2001-01-01', periods=len(a))
    array = xr.DataArray(a, dims=['time'], coords=[time])
    array = array.expand_dims('cell', axis=1)
    exceed = array==1
    st = xr.full_like(array,np.nan, dtype=np.float)
    end = xr.full_like(array,np.nan, dtype=np.float)
    events = xr.full_like(array,np.nan, dtype=np.float)
    st[5] = 1
    st[17] = 11
    st[24] = 20
    end[5] = 5
    end[17] = 17
    end[24] = 24
    events[1:6] = 1
    events[11:18] = 11
    events[20:25] =20 
    return (exceed, st, end, events) 
