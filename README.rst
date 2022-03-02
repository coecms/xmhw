=============================
 `xmhw <https://xmhw.readthedocs.io/en/latest>`_
=============================

XMHW - Xarray based Marine HeatWave code -  

.. image:: https://readthedocs.org/projects/xmhw/badge/?version=latest
  :target: https://xmhw.readthedocs.io/en/latest/
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6270280.svg
   :target: https://doi.org/10.5281/zenodo.6270280

.. content-marker-for-sphinx

XMHW identifies marine heatwaves from a timseries of Sea Surface Temperature data and calculates statistics associated to the detected heatwaves. It is based on the `marineHeatWaves code <https://github.com/ecjoliver/marineHeatWaves/>`_ by Eric Olivier 

Functions:

- **threshold**  
- **detect** 
- **block_average**  work in progress!!
- **mhw_rank**       work in progress!!

As this code uses xarray results are xarray datasets.

-------
Install
-------

    You can install the latest version of xmhw directly from conda (coecms channel)

    conda install -c coecms -c conda-forge xmhw

    If you want to install an unstable version or a different branch:

    * git clone https://github.com/coecms/xmhw
    * git checkout <branch-name>   (if installing a a different branch from master)
    * cd xmhw
    * python setup.py install or pip install ./
      use --user with either othe commands if you want to install it in ~/.local

    
---
Use
---
Some examples of how to use the functions and explanations of how the functions work are shown in the readthedocs documentation linked above and the `xmhw_demo.ipynb notebook <https://github.com/coecms/xmhw/blob/master/docs/xmhw_demo.ipynb>_` in the docs folder.

----------------------
Latest version v 0.8.0
----------------------

Main updates:
    * Added a tstep (timestep) option to use the original timestep of the timeseries passed as input as step, rather than assuming data is daily. This allows to calculate climatologies and detect mhw events at different frequency.
    * Added conda support
    * Added readthedocs documentation
     

---------------
Version v 0.7.0
---------------

Main updates:
    * redesigned threshold function, to make it faster
      In particular there is an option to run the averages and percentile calculation without skipping NaNs.
      This makes the calculation faster.
    * Instead of delayed the entire main function calculating both climatologies, the calculation of percentile and seasonal average were further separated so they are now both delayed, so it is the runavg() function used to smooth them. This allows more parallelisation.
    * There are more options to manage the handling of NaNs this are explained in the demo notebook
    * Improved documentation of functions 

Disclaimer!!
------------

This is a work in progress, I tested both threshold and detect and these can reproduced Eric Olivier results with the data I used.
Still I haven't extensively tested this and most unit tests in the code need updating.
In particular the code is potentially not ready for timeseries with a 360 days calendar year

I am currently working on block_average and mhw_rank functions.

~~~~~



