=============================
 `xmhw <https://xmhw.readthedocs.io/en/stable>`_
=============================

XMHW - Xarray based Marine HeatWave code -  

.. image:: https://readthedocs.org/projects/xmhw/badge/?version=latest
  :target: https://xmhw.readthedocs.io/en/stable/
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5652368.svg
   :target: https://doi.org/10.5281/zenodo.5652368

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


    git clone https://github.com/coecms/xmhw
    
    cd xmhw 
    
    python setup.py install --user 
    
    --user option install the module in ~/.local.
---
Use
---
Some examples of how to use the functions and explanations of how the functions work are shown in the xmhw_demo.ipynb notebook in the docs folder  [https://github.com/coecms/xmhw/blob/master/docs/xmhw_demo.ipynb]. 

We will keep on adding more information to this notebook.

--------------------
Latest version v 0.7
--------------------

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



