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


from xmhw.xmhw import threshold, detect 
from xmhw_fixtures import *
from numpy import testing as nptest
from xmhw.exception import XmhwException


@pytest.mark.xfail
def test_threshold(clim_oisst, clim_oisst_nosmooth, oisst_ts):
    # test exceptions with wrong arguments
    with pytest.raises(XmhwException):
        clim = threshold(oisst_ts, smoothPercentileWidth=6)
    clim = threshold(oisst_ts, smoothPercentile=False, skipna=True)
    th1 = clim['thresh'].sel(lat=-42.625, lon=148.125)
    seas1 = clim['seas'].sel(lat=-42.625, lon=148.125)
    th2 = clim['thresh'].sel(lat=-41.625, lon=148.375)
    seas2 = clim['seas'].sel(lat=-41.625, lon=148.375)
    #temporarily testing only after mid March so as to avoid the +-2 days from feb29
    nptest.assert_array_almost_equal(clim_oisst_nosmooth.thresh1[60:].values,th1[60:].values) 
    nptest.assert_array_almost_equal(clim_oisst_nosmooth.thresh2[60:].values,th2[60:].values) 
    nptest.assert_array_almost_equal(clim_oisst_nosmooth.seas1[60:].values,seas1[60:].values, decimal=4) 
    nptest.assert_array_almost_equal(clim_oisst_nosmooth.seas2[60:].values,seas2[60:].values, decimal=4) 
    # test default smooth True
    clim = threshold(oisst_ts, skipna=True)
    th1 = clim['thresh'].sel(lat=-42.625, lon=148.125)
    seas1 = clim['seas'].sel(lat=-42.625, lon=148.125)
    th2 = clim['thresh'].sel(lat=-41.625, lon=148.375)
    seas2 = clim['seas'].sel(lat=-41.625, lon=148.375)
    #temporarily testing only after mid March so as to avoid the =-15 days from feb29
    nptest.assert_array_almost_equal(clim_oisst.thresh1[82:].values,th1[82:].values) 
    nptest.assert_array_almost_equal(clim_oisst.thresh2[82:].values,th2[82:].values) 
    nptest.assert_array_almost_equal(clim_oisst.seas1[82:].values,seas1[82:].values, decimal=4) 
    nptest.assert_array_almost_equal(clim_oisst.seas2[82:].values,seas2[82:].values, decimal=4) 
    # add test with 1-dimensional and/or 2-dimensional arrays to make sure it still works 
    # add test with skipna False for this set and one without nans

@pytest.mark.xfail
def test_detect(oisst_ts, clim_oisst):
# detect(temp, thresh, seas, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=None, coldSpells=False, tdim='time')
    # test exceptions with wrong arguments
    with pytest.raises(XmhwException):
        mhw = detect(oisst_ts, clim_oisst.thresh2, clim_oisst.seas2, minDuration=3, maxGap=5)

