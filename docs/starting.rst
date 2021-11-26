Detecting marine heatwave events with xmhw
==========================================

*xmhw* is a xarray based version of the MarineHeatWave code. The main
difference with the original code are the following: \* uses xarray and
dask \* the calculation of climatologies and detection of mhw events are
in separate functions, so they can be called independently \* can handle
multi dimensional grids and detect land points \* NaNs treatment can be
customised \* severity of events added to *detect* function output \*
produce xarray datasets instead of list of dictionaries

Import *threhshold* to calculate the climatologies and *detect* to
detect the mhw events.

.. code:: ipython3

    import xarray as xr
    import dask
    from xmhw.xmhw import threshold, detect

Calculating the climatologies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this demo I am using a small subset of the NOAA OISST timeseries.
You can use whatever seawater temperature dataset you have available,
just select a small region initially to test the code.

.. code:: ipython3

    # open file, read sst and calculate climatologies
    ds =xr.open_dataset('sst_test.nc')
    sst = ds['sst']
    clim = threshold(sst)
    clim


    Dimensions:   (doy: 366, lat: 12, lon: 20)
    Coordinates:
        quantile  float64 0.9
      * doy       (doy) int64 1 2 3 4 5  ... 363 364 365 366
      * lat       (lat) float64 -43.88 -43.62 ...  -41.12
      * lon       (lon) float64 144.1 144.4 ...  148.9
    Data variables:
        thresh    (doy, lat, lon) float64 dask.array&lt;chunksize=(365, 1, 20), ...
        seas      (doy, lat, lon) float64 dask.array&lt;chunksize=(365, 1, 20), ...
    Attributes:
        source:  xmhw code: https://github.com/coecms/xmhw
        title:   Seasonal climatology and threshold calculated to detect marine
                 heatwaves following the  Hobday et al. (2016) definition
        history: 2021-11-19: calculated using xmhw code https://github.com/coecms/xmhw
        xmhw_parameters:  Threshold calculated using:
                 90 percentile;
                 climatology period is 1981-1981';
                 window half width used for percentile is 5; width of moving
                 average window to smooth percentile is 31

As you can see above **clim** is a xarray dataset with two variables: \*
thresh - the percentile threshold \* seas - the climatology mean.

The dimension is **doy** which stands for day of the year, this is based
on a 366 days calendar. Finally the dataset includes a few global
attributes detailing the climatology period, the percenttile used and
other parameters used in the calculation. This can be easily saved to a
file simply by running: > clim.to_netcdf(‘filename’)

Detecting MHW events
~~~~~~~~~~~~~~~~~~~~

Now that we have the climatologies we can run detect

.. code:: ipython3

    mhw = detect(sst, clim['thresh'], clim['seas'])
    mhw

    xarray.Dataset
    Dimensions: 
        events: 2047  lat: 12  lon: 20
    Coordinates:
        events (events) float64 91.0 116.0 ...
        lat (lat) float64 -43.88 -43.62 ...
        lon (lon) float64 144.1 144.4 ...
    Data variables:
        event (events, lat, lon) float64 nan nan ...
        index_start (events, lat, lon) float64 nan nan ...
        index_end (events, lat, lon) float64 nan nan ...
        time_start (events, lat, lon) datetime64[ns] NaT NaT ...
        time_end (events, lat, lon) datetime64[ns] NaT NaT ...
        time_peak (events, lat, lon) datetime64[ns] NaT NaT ...
        intensity_max (events, lat, lon) float64 nan nan ...
        intensity_mean (events, lat, lon) float64 nan nan ...
        intensity_cumulative (events, lat, lon) float64 nan nan ...
        severity_max (events, lat, lon) float64 nan nan ...
        severity_mean (events, lat, lon) float64 nan nan ...
        severity_cumulative (events, lat, lon) float64 nan nan ...
        severity_var (events, lat, lon) float64 nan nan ...
        intensity_mean_relThresh (events, lat, lon) float64 nan nan ...
        intensity_cumulative_relThresh (events, lat, lon) float64 nan nan ...
        intensity_mean_abs (events, lat, lon) float32 nan nan ...
        intensity_cumulative_abs (events, lat, lon) float32 nan nan ... 
        duration_moderate (events, lat, lon) float64 nan nan ...
        duration_strong (events, lat, lon) float64 nan nan ... 
        duration_severe (events, lat, lon) float64 nan nan ... 
        duration_extreme (events, lat, lon) float64 nan nan ...
        index_peak (events, lat, lon) float64 nan nan ...
        intensity_var (events, lat, lon) float64 nan nan ...
        intensity_max_relThresh (events, lat, lon) float64 nan nan ...
        intensity_max_abs (events, lat, lon) float32 nan nan ...
        intensity_var_relThresh (events, lat, lon) float64 nan nan ...
        intensity_var_abs (events, lat, lon) float32 nan nan ...
        category (events, lat, lon) float64 nan nan ...
        duration (events, lat, lon) float64 nan nan ...
        rate_onset (events, lat, lon) float64 nan nan ...
        rate_decline (events, lat, lon) float64 nan nan ...

    Attributes:
        source: xmhw code: https://github.com/coecms/xmhw
        title: Marine heatwave events identified applying the
            Hobday et al. (2016) marine heat wave definition
        history: 2021-11-19: calculated using xmhw code https://github.com/coecms/xmhw
        xmhw_parameters: MHW detected using: 5 days of minimum duration;
                         events separated by 2 or less days were joined

We can see above all the output variables listed and again global
attributes detailing the dataset settings. The dimension **events**
represents the starting point of each event. Let’s select one grid point
to see more in detail its structure.

.. code:: ipython3

    mhw_point = mhw.isel(lat=2, lon=15)
    mhw_point.events

    array([   91.,   116.,   164., ..., 14375., 14379., 14381.])

Printing out the all events array shows that the first detected event
occurs at the 91st timestep of the original timeseries, the last events
starts at timestep 14381. Not all these events will be occuring at the
selected grid point. We can see that having a look at the index_start or
time_start variables. By dropping all the NaN values along the events
dimension, we can see there are 60 mhw events occuring at this grid
point.

.. code:: ipython3

    mhw_point.time_start.dropna(dim='events')

    array(['1985-04-08T12:00:00.000000000', '1988-05-12T12:00:00.000000000',
           '1988-06-10T12:00:00.000000000', '1988-07-17T12:00:00.000000000',
           ...
           dtype='datetime64[ns]')

As for the climatologies dataset, we can save the mhw dataset to a
netcdf file easily.

.. code:: ipython3

    mhw.to_netcdf('mhw_test.nc')

This file has a small grid, so we could save it as it is and still
produce a small file. However, it is worth adding some “encoding” to
save storage, this will be necessary when dealing with bigger grids.
Xarray has automatically used a float64 format for ~20 of the variables.
Converting all the variables to float32 format will save a lot of
storage. This dataset also has a lot of NaNs values, as its structure is
“sparse”, so it is a good idea to save the results in a compressed
format. Encoding allows us to add internal compression and also to
convert the arrays format.

.. code:: ipython3

    # First we create a dictionary representing the settings we want to use
    # then we apply that to all the dataset variables and we use the
    #  resulting dictionary when calling to_netcdf() 
    #
    comp = dict(zlib=True, complevel=5, shuffle=True, dtype='float32')
    encoding = {var: comp for var in mhw.data_vars}
    mhw.to_netcdf('mhw_test_encoded.nc', encoding=encoding)

Checking the sizes of both files

.. code:: ipython3

    !du -sh mhw_test.nc
    !du -sh mhw_test_encoded.nc

    109M	mhw_test.nc
    2.2M	mhw_test_encoded.nc
