Threshold in detail
-------------------

In the previous example the threshold function was called with its
default arguments, so only temperature was needed. As for the original
Marine heatwave code several other parameters can be set:

::

    threshold(temp, tdim='time', climatologyPeriod=[None,None], pctile=90, windowHalfWidth=5,  
              smoothPercentile=True, smoothPercentileWidth=31, maxPadLength=None, 
              coldSpells=False, Ly=False, anynans=False, skipna=False):

Where *temp* is the temperature timeseries, this is the only input
needed. Arguments names are the same as the original MarineHeatWave
code, where possible:

-  **climatologyPeriod**: list(int), optional Period over which
   climatology is calculated, specified as list of start and end years.
   Default is to use the full time series.
-  **pctile**: int, optional Threshold percentile used to detect events
   (default=90)
-  **windowHalfWidth**: int, optional Half width of window about
   day-of-year used for the pooling of values and calculation of
   threshold percentile (default=5)
-  **smoothPercentile**: bool, optional If True smooth the threshold
   percentile timeseries with a moving average (default is True)
-  **smoothPercentileWidth**: int, optional Width of moving average
   window for smoothing threshold in days, has to be odd number
   (default=31)
-  **maxPadLength**: int, optional Specifies the maximum length (days)
   over which to interpolate NaNs in input temp time series. i.e., any
   consecutive blocks of NaNs with length greater than maxPadLength will
   be left as NaN. If None it does not interpolate (default is None).
-  **coldSpells**: bool, optional If True the code detects cold events
   instead of heat events (default is False)
-  **Ly**: bool, optional !! Not yet fully implemented If True the
   length of the year is < 365/366 days (e.g. a 360 day year from a
   climate model). This affects the calculation of the climatology
   (default is False)

New arguments
^^^^^^^^^^^^^

-  **tdim** - optional, to specify the time dimension name, default is
   “time” . NB you do not need to pass the time array as in the original
   as the timeseries is an xarray data array the time dimension is
   included
-  **anynans**: bool, optional Defines in land_check which cells will be
   dropped, if False only ones with all NaNs values, if True all cells
   with even 1 NaN along time dimension will be dropped (default is
   False)
-  **skipna**: bool, optional If True percentile and mean function will
   use skipna=True. Using skipna option is much slower (default is
   False)

More on missing values later.

Example
^^^^^^^

This is just showing how we can call the function changing some of the
default parameters. In this case we are assuming sst time dimension is
called ‘time_0’ and we want a base period from 1 Jan 1984 to 31 Dec
1994.

   clim = threshold(sst, climatologyPeriod=[1984,1994], tdim=‘time_0’)

NB after passing the timeseries as first argument, the order of the
other ones is irrelevant as they are all keywords arguments.

It is important to notice that differently from the original function
which takes a numpy 1D array, because we are using xarray we can pass a
3D array (in fact we could pass any n-dim array) and the code will deal
with it. We selected a 12X20 lat-lon region and of these 135 grid cells
are ocean.

The function return a dataset with the arrays: - **thresh** - for the
threshold timeseries - **seas** - for the seasonal mean

Differently from the original function, here the climatologies are saved
not along the entire timeseries but only along the new **doy**
dimension. Given that xarray keeps the coordinates with the arrays there
is no need to repeat the climatologies along the time axis. We also try
to follow the CF conventions and define appropriate variables attributes
and some global attributes that record the parameters used to calculate
the threshold for provenance.

Handling of dimensions and land points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As so before we are passing the full grid to the function without
worrying about land points, or how many dimensions it has. Before
calculating anything, the code calls the function land_check() (from
xmhw.identify). This function handles the dimensions and land points of
the grid in two steps: - stacks all dimensions but the time dimension in
a new ‘cell’ dimension; - removes all the land points, these are assumed
to have all NaN values along the time axis

In our example ‘cell’ will be composed by stacked (lat,lon) points. The
resulting array will have (time, cell) dimensions, and the cell points
which are land will not be part of it. The climatologies then will be
calculated for each cell point. Finally the results will be unstacked
before returning the final output. NB This approach can occasionally
produce a grid of different size from the original if all the cells at a
specific latitude or longitude are masked as land. In that case the
final grid will be smaller, you can however easily reindex your results
as the original grid. > clim = clim.reindex_like(sst)

Handling of NaNs
~~~~~~~~~~~~~~~~

It is important to understand how the **threshold()** function is
dealing with NaNs. If there are NaNs values in the timeseries that is
passed to the function, this could produce wrong results. You can take
care of NaNs in the timeseries before passing it to threshold or you can
take one of the following approaches: 1) We already saw that
land_check() will remove all the points that have all NaNs values along
the time dimension. You can choose to be more strict and also exclude
any ell points that even just one NaN value. To do so you can set the
**anynans** argument to True. This is a bit of an extreme approach as
especially with observations data it is not unusual to have a few NaNs.
> clim = threshold(sst, anynans=True)

2) set **skipna** to True - this tells the code to skip NaNs when
   calculating averages and/or the percentile. By default the **skipna**
   argument is set to False as using this option can double up the
   execution time. But if you are working on a small grid than it is a
   safer option. > clim = threshold(sst, skipnans=True)

3) use **maxPadLength** this will trigger interpolation for all NaNs
   points, with the exception of consecutive blocks with length greater
   than maxPadLength. > clim = threshold(sst, maxpadlength=5,
   anynans=True)

Used in conjuction with **anynans** as shown above you can use it to
eliminate only the cell points that have bigger gaps.

