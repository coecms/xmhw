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
import py
import os
import xarray as xr


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

