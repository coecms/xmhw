=============================
 `xmhw <https://xmhw.readthedocs.io/en/stable>`_
=============================

XMHW - Xarray based Marine HeatWave code -  

.. image:: https://readthedocs.org/projects/xmhw/badge/?version=latest
  :target: https://xmhw.readthedocs.io/en/stable/

.. content-marker-for-sphinx

XMHW identifies marine heatwaves from a timseries of Sea Surface Temperature data and calculates statistics associated to the detected heatwaves. It is based on the`marineHeatWaves code <https://github.com/ecjoliver/marineHeatWaves/>`_ by Eric Olivier 

Functions:

- **threshold**  
- **detect** 
- **block_average**  work in progress

Disclaimer!!

This is a work in progress, I tested both threshold and detect and these can reproduced Eric Olivier results with the data I used.
Still I haven't extensively tested this and most unit tests in the code need updating.
In particular the code is potentially not ready for timeseries with a 360 days calendar year

I am still working on block_average.

As this code uses xarray results are xarray datasets.
-------
Install
-------


    git clone
    
    cd xmhw 
    
    python setup.py install --user 
    
    --user option install the module in ~/.local.
---
Use
---

To come ...

~~~~~

