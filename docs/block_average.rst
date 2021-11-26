Block average
~~~~~~~~~~~~~

The blockAverage function on the original MHW code is used to calculate
statistics along a block of time. The default is 1 year block. If the
timeseries used starts or ends in the middle of the year then the
results for this two years have to be treated carefully. Most of the
statistics calculated on the block are simple statistics. Given that the
mhw properties are saved now as an array it is simple to calculate them
after grouping by year or “bins”of years on the entire dataset. However,
we added xmhw has a block_average() function to reproduce the same
results.

.. code:: ipython3

    # To call with standard parameters, all is needed is the output of detect function 
    # I am also passing the intermediate results datasets as a way to provide the temperature and
    # climatologies
    from xmhw.stats import block_average
    block = block_average(mhw, dstime=intermediate)
    block
     
    Assuming time is time dimension
    Both ts and climatologies are available, calculating ts and category stats

    xarray.Dataset
    Dimensions:
        years: 41  lat: 12  lon: 20
    Coordinates:
        years (years) object [1981, 1982) ... 
        lat (lat) float64 -43.88 -43.62 ...
        lon (lon) float64 144.1 144.4 ...
    Data variables:
        ecount (years, lat, lon) float64 0.0 0.0 ...
        duration (years, lat, lon) float64 nan nan ...
        intensity_max (years, lat, lon) float64 nan nan ...
        intensity_max_max (years, lat, lon) float64 nan nan ...
        intensity_mean (years, lat, lon) float64 nan nan ...
        intensity_cumulative (years, lat, lon) float64 nan nan ...
        total_icum (years, lat, lon) float64 0.0 0.0 ...
        intensity_mean_relThresh (years, lat, lon) float64 nan nan ...
        intensity_cumulative_relThresh (years, lat, lon) float64 nan nan ...
        severity_mean (years, lat, lon) float64 nan nan ...
        severity_cumulative (years, lat, lon) float64 nan nan ...
        intensity_mean_abs (years, lat, lon) float64 nan nan ...
        intensity_cumulative_abs (years, lat, lon) float64 nan nan ...
        rate_onset (years, lat, lon) float64 nan nan ...
        rate_decline (years, lat, lon) float64 nan nan ...
        ts_mean (years, lat, lon) float32 11.57 11.56 ...
        ts_max (years, lat, lon) float32 14.28 14.32 ...
        ts_min (years, lat, lon) float32 9.51 9.46 ...
        moderate_days (years, lat, lon) float64 0.0 0.0 ...
        strong_days (years, lat, lon) float64 0.0 0.0 ...
        severe_days (years, lat, lon) float64 0.0 0.0 ...
        extreme_days (years, lat, lon) float64 0.0 0.0 ...
        total_days (years, lat, lon) float64 0.0 0.0 ...

Block_average function in detail
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   def block_average(mhw, dstime=None, period=None, blockLength=1, mtime='time_start',
                     removeMissing=False, split=False):

       Parameters
       ----------
       mhw: xarray Dataset 
           Includes MHW properties by events
       dstime: xarray DataArray/Dataset, optional
            Based on intermediate dataset returned by detect(), includes
            original ts and climatologies (optional) along 'time' dimension
            (default is None)
            If present and is array or dataset with only 1 variable script assumes this
            is sst and sw_temp is set to True
            If present and dataset with 'thresh' and 'seas' also sw_cats is set to True
       period: pandas Series
           Absolute value of temperature along time index
       blockLength: int, optional
           Size of blocks in years (default=1)
       mtime: str, optional
           Name of mhw time variable to use to assign events to blocks, 
           Options are start or peak times (default='time_start')
       removeMissing: bool, optional
           If True remove statistics for any blocks that has NaNs in ts.
           Work in progress
           (default is False)
       split: bool, optional
           Work in progress
           (default is False)

We are still working on the mhw_rank() function to calculate rank and
return periods.
