=============================
 `xmhw <https://xmhw.readthedocs.io/en/stable>`_
=============================

XMHW - Xarray based Marine HeatWave code -  

.. image:: https://readthedocs.org/projects/xmhw/badge/?version=latest
  :target: https://xmhw.readthedocs.io/en/stable/

.. content-marker-for-sphinx

XMHW identifies marine heatwaves from a timseries of Sea Surface Temperature data and calculates statistics associated to the detected heatwaves. It is based on the `marineHeatWaves code <https://github.com/ecjoliver/marineHeatWaves/>`_ by Eric Olivier 

Functions:

- **threshold**  
- **detect** 
- **block_average**  work in progress!!
- **mhw_rank**       work in progress!!


-------
Install
-------


    git clone 
    cd xmhw
    python setup.py install ( --user ) if you want to install it in ~/.local
---
Use
---
Some examples of how to use the functions and explanations of how the functions work are shown in the xmhw_demo.ipynb notebook in the docs folder.
We will keep on adding more information to this notebook.

___________________
Latest version v0.7
-------------------
Main updates:
    * redesigned threshold function, to make it faster
      In particular there is an option to run the averages and percentile calculation without skipping NaNs.
      This makes the calculation faster.
    * Instead of delayed the entire main function calculating both climatologies, the calculation of percentile and seasonal average were further separated so they are now both delayed, so it is the runavg() function used to smooth them. This allows more parallelisation.
    * There are more options to manage the handling of NaNs this are explained in the demo notebook
    * Improved documentation of functions 

~~~~~



