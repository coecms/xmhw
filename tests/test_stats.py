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

import pandas as pd
from xmhw.stats import (block_average, check_variables, cat_days,
                        check_coordinates)
from xmhw.identify import land_check
from xmhw_fixtures import *
from xmhw.exception import XmhwException
import numpy.testing as nptest
import xarray.testing as xrtest
import pandas.testing as pdtest


def test_block_average():
    pass

def test_cat_days():
    cats = pd.Series(data=[1,2,1,1,2,3,1,4,3,2,1,1,2])
    assert cat_days(cats) == 6
    assert cat_days(cats, cat=2) == 4
    assert cat_days(cats, cat=3) == 2
    assert cat_days(cats, cat=4) == 1


def test_check_variables(inter_data):
    # case where ts and cats are both included
    dstime, sw_cats, sw_temp = check_variables(inter_data)
    variables = [x for x in dstime.variables.keys()]
    assert sw_cats is True
    assert sw_temp is True
    assert set(variables) == set(['cats', 'ts', 'index'])
    # case where ts is not included
    notts = inter_data.drop('ts')
    dstime, sw_cats, sw_temp = check_variables(notts)
    variables = [x for x in dstime.variables.keys()]
    assert sw_cats is True
    assert sw_temp is False
    assert set(variables) == set(['cats', 'index'])
    # case where only ts is included as array
    ts = inter_data['ts']
    dstime, sw_cats, sw_temp = check_variables(ts)
    variables = [x for x in dstime.variables.keys()]
    assert sw_cats is False
    assert sw_temp is True
    assert set(variables) == set(['ts', 'index'])
    # case where only ts is included as dataset 
    tsds = xr.Dataset()
    tsds['ts'] = ts
    dstime, sw_cats, sw_temp = check_variables(tsds)
    variables = [x for x in dstime.variables.keys()]
    assert sw_cats is False
    assert sw_temp is True
    assert set(variables) == set(['ts', 'index'])

def test_check_coordinates(inter_data):
    inter_stack = land_check(inter_data, tdim='index')
    outds, coord = check_coordinates(inter_data)
    xrtest.assert_equal(inter_stack, outds) 
    assert coord == 'cell'
    stacked = inter_stack.rename({'cell': 'other'})
    outds, coord = check_coordinates(stacked)
    xrtest.assert_equal(stacked, outds) 
    assert coord == 'other'
