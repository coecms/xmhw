Setting up dask
~~~~~~~~~~~~~~~

Both the threshold and detect functions are set up to use dask delayed
automatically. I found this was a good away to make sure the main
processes would be automatically run in parallel even if you are not
experienced with dask. This approach add some overhead before the actual
calculation starts, when dask is working out the task graph. This is
usually negligible, but with a bigger grid size you might end up with
too many tasks and a less efficinet graph. In that case, and anytime you
are working with limited resources, it is more efficient to split the
grid and run the code separately for each grid section. You can easily
recompose the original grid by concatenating together the resulting
datasets. Even when running the functions on the full grid it is
important to setup proper chunks. The code will make sure that the
timeseries for each grid point are all in the same chunk. This is
important as it is requires by some of the calculation, it also makes
sense as every operation is done cell by cell. > sst =
sst.chunk({‘time’: -1})

This corresponds to have a chunk size (for time dimension) equal to the
length of the timeseries. The code will not do anything for the other
dimensions, so it is a good idea to make sure that once you have chunked
the time dimension, you are left with resonable chunk sizes.

.. code:: ipython3

    # This will tell dask to automatically determine a good chunk size for the other dimensions
    # Assuming that the sst variable has (time, lat, lon) dimensions
    sst = sst.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
    sst.data

   sst.data

shows the chunks shape and their size. As the sst array is quite small
we have only 1 chunk in this case, if this was a big grid we would want
chunks around 200 MB of size. Below is an example running detect on a
big grid by splitting the grid according to chunks. You can split the
grid in different ways and advantage of this methid is that the data
will be all in one chunk. So by managing the chunksize you can optimise
the amount of memory used. Whichever way you are splitting the grid make
sure that is somehow aligned with the dataset chunks.

.. code:: ipython3

    # retrieve chunks information
    # xt/xy will be lists the number of lat/lon points for each chunk
    # As an example
    # [20, 20, 20], [30, 30, 30] 
    # means we have 20 lat and 30 lon points for each chunk for a total of 600 grid cells
    # we set these as out "steps"
    dummy, xt, xy = sst.chunks
    xtstep = xt[0]
    ytstep = yt[0]
    # the length of the list gives has the number of chunks, in the example 3
    xtblocks = len(xt)
    ytblocks = len(yt)
    print(xtstep, ytstep, xtblocks, ytblocks)

.. code:: ipython3

    # create an empty list to store the results
    # loop first across the ytblocks and then xtblocks
    # using xtstep and ytstep to determine the indexes to use with isel
    
    results = []
    for i in range(xtblocks):
        xt_from = i*xtstep
        xt_to = (i+1)*xtstep
        for j in range (ytblocks):
            yt_from = j*ytstep
            yt_to = (j+1)*ytstep
            ts = sst.isel(xt_ocean=slice(xt_from,xt_to),yt_ocean=slice(yt_from,yt_to))
            th = thresh.isel(xt_ocean=slice(xt_from,xt_to),yt_ocean=slice(yt_from,yt_to))
            se = seas.isel(xt_ocean=slice(xt_from,xt_to),yt_ocean=slice(yt_from,yt_to))
            j+=1
            # run function
            results.append(detect(ts,th,se))
            del ts, th, se 
        i+=1

.. code:: ipython3

    # Combine the results into one dataset
    mhw = xr.combine_by_coords(results)
