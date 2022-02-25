Detect function in detail
-------------------------

The *detect* function identifies all the mhw events and their
characteristics. It corresponds to the second part of the original
detect function and again mimic the logic and most of options of the
original code.

::

       def detect(temp, th, se, tdim='time', minDuration=5, joinGaps=True, 
                  maxGap=2, maxPadLength=None, coldSpells=False,
                  intermediate=False, anynans=False):

Apart from the timeseries, the threshold and the seasonal average, the
other arguments are all optional. As for threshold an option to pass the
name of the time dimension (**tdim**) and the **anynans** argument to
define which grid cells will be removed from calculation, were added. It
is important to used this last consistently with the approach used when
calculating the threshold. The last new argument is **intermediate**,
when set to True, also intermediate results are saved. These include the
original timeseries, climatologies, detected events, categories and some
of the mhw properties but along the full length of the time axis.

Arguments specific to **detect()**: \* minDuration: int, optional
Minimum duration (days) to accept detected MHWs (default=5) \* joinGaps:
bool, optional If True join MHWs separated by a short gap (default is
True) \* maxGap: int, optional Maximum limit of gap length (days)
between events (default=2)

.. code:: ipython3

    mhw, intermediate = detect(sst, clim['thresh'], clim['seas'], intermediate=True)
    intermediate

    xarray.Dataset
    Dimensions:
        time: 14392  lat: 12  lon: 20
    Coordinates:
        time (time) datetime64[ns] 1981-09-01T12:00:00 ...
        lat (lat) float64 -43.88 -43.62 ...
        lon (lon) float64 144.1 144.4 ...
     Data variables:
        ts (time, lat, lon) float32 11.13 11.16 ...
        seas (time, lat, lon) float64 nan nan ...
        thresh (time, lat, lon) float64 nan nan ...
        bthresh (time, lat, lon) object False ...
        events (time, lat, lon) float64 nan nan ...
        relSeas (time, lat, lon) float64 nan nan ...
        relThresh (time, lat, lon) float64 nan nan ...
        relThreshNorm (time, lat, lon) float64 nan nan ...
        severity (time, lat, lon) float64 nan nan ...
        cats (time, lat, lon) float64 nan nan ...
        duration_moderate (time, lat, lon) object False ...
        duration_strong (time, lat, lon) object False ...
        duration_severe (time, lat, lon) object False ...
        duration_extreme (time, lat, lon) object False ... 
        mabs (time, lat, lon) float32 nan nan ...
    Attributes: (0)

