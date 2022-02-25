Detecting heatwaves at different frequencies
--------------------------------------------

The original marine heatwave code assumes the timeseries has daily
frequency. In *xmhw* you can also calculate the climatologies and then
detect the heatwaves based on your timeseries original timestep. So if
you are passing monthly data you can calculate monthly climatologies, if
you pass a timeseries resampled as an average over n-days than n-days
will be your climatology base unit.

.. code:: ipython3

    # First I am loading again the test data and 
    # create a new timeseries by averaging 5-days interval
    sst = ds['sst']
    sst_5days = sst.coarsen(time=5, boundary="trim").mean() 

.. code:: ipython3

    # Now we can calculate the threshold and detect mhw again with the new timeseries.
    # We are using the 'tstep=True' option to tell the code to use the 5days intervals
    # as timestep base
    clim_5days = threshold(sst_5days, smoothPercentileWidth=5, tstep=True)

    ---------------------------------------------------------------------------
    XmhwException                             Traceback (most recent call last)
      /local/w35/pxp581/tmp/ipykernel_2716220/2181158061.py in <module>
        2 # We are using the 'tstep=True' option to tell the code to use the 5days intervals
        3 # as timestep base
        ...
    ---> 62            raise XmhwException("To use original timestep as " +
         63                "climatology base unit, timeseries has to have" +
         64                " complete years")

    XmhwException: To use original timestep as climatology base unit,
                   timeseries has to have complete years

The first attempt produced an exception this is because at the moment
the code cannot handle yet incomplete years. This means that every year
needs to have the same number of timesteps. This timeseries starts in
Sep 1981 and end in Jan 2021. So we have to select only the years in
between. We also have to remove all the 29 of Feb so every year has 365
days that can be equally split in 5 days intervals.

.. code:: ipython3

    sst_yrs = sst.sel(time=slice('1982','2020'))
    sst_365 = sst_yrs.sel(time=~((sst_yrs.time.dt.month == 2) & 
                                    (sst_yrs.time.dt.day == 29)))
    sst_5days = sst_365.coarsen(time=5, boundary="exact").mean()

.. code:: ipython3

    clim_5days = threshold(sst_5days, smoothPercentileWidth=5, tstep=True)
    clim_5days

    xarray.Dataset
    Dimensions:  doy: 73  lat: 12  lon: 20
    Coordinates:
        quantile () float64 0.9
        doy (doy) int64 1 2 3 4 ... 72 73
        lat (lat) float64 -43.88 -43.62 ... -41.12
        lon (lon) float64 144.1 144.4 ... 148.9
    Data variables:
        thresh (doy, lat, lon) float64 dask.array<chunksize=(72, 1, 20), ...
        seas (doy, lat, lon) float64 dask.array<chunksize=(72, 1, 20), ...
    Attributes:
        source: xmhw code: https://github.com/coecms/xmhw
        title: Seasonal climatology and threshold calculated to detect marine
               heatwaves following the  Hobday et al. (2016) definition
        history: 2021-11-19: calculated using xmhw code https://github.com/coecms/xmhw
        xmhw_parameters: Threshold calculated using:
                   90 percentile;
                   climatology period is 1982-1982';
                   window half width used for percentile is 5;
                   width of moving average window to smooth percentile is 5

As you can see we ended with only 73 “doy” steps, as this
day-of-the-year is really a 5 days interval. Note also that I’ve changed
the smoothPercentileWidth to 5 instead of the default 31. All the
default for both the threshold and detect functions are based on a daily
timesteps so if you use a different frequency they need to be adapted to
produce sensible results.

The detect() function will also need to be passed tstep=True to be
consistent.

.. code:: ipython3

    mhw_5days = detect(sst_5days, clim_5days['thresh'], clim_5days['seas'],
                       maxGap=1, tstep=True)
    mhw_5days

    xarray.Dataset
    Dimensions: events: 208  lat: 12  lon: 20
    Coordinates:
        events (events) float64 282.0 284.0 ... 2.818e+03
        lat (lat) float64 -43.88 -43.62 ... -41.12
        lon (lon) float64 144.1 144.4 ... 148.9
    Data variables: (31)
    Attributes:
       source: xmhw code: https://github.com/coecms/xmhw
       title: Marine heatwave events identified applying the Hobday
               et al. (2016) marine heat wave definition
       history: 2021-11-19: calculated using xmhw code https://github.com/coecms/xmhw
       xmhw_parameters: MHW detected using: 
            5 days of minimum duration;
            events separated by 1 or less days were joined

You can also use the same option with monthly, weekly data or any other
interval which is not daily. This is the option to use also with a 360
days year calendar, as the standard behaviour would be to try to get
force the timeseries in a 366 days year, which would cause an error. So
even if ‘tstep’ is False, the code will try to work out the calendar and
if this is a 360 days one it will impose tstep=True.

