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


from xmhw.mhw import threshold, detect, mhw_filter 
from mhw_fixtures import *

def test_mhw_filter():
#mhw_filter(exceed, minDuration, joinGaps, maxGap):
    assert True

def test_threshold(clim_oisst, oisst)
    clim = threshold(oisst)
    #threshold(temp, climatologyPeriod=[None,None], pctile=90, windowHalfWidth=5, smoothPercentile=True, 
    th1 = clim['thresh'].sel(lat=-42.625, lon=148.125)
    seas1 = clim['seas'].sel(lat=-42.625, lon=148.125)
    th2 = clim['thresh'].sel(lat=-41.625, lon=148.375)
    seas2 = clim['seas'].sel(lat=-41.625, lon=148.375)
    assert 

def test_detect():
    assert True
#detect(temp, clim=None, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=False, 

def test_mhw_ds():
#mhw_ds(start, end, events, ts, clim):
    assert True
