# XMHW - Xarray based Marine HeatWave code  


XMHW identifies marine heatwaves from a timeseries of Sea Surface Temperature data and calculates statistics associated to the detected heatwaves. It is based on the`marineHeatWaves code <https://github.com/ecjoliver/marineHeatWaves/>`_ by Eric Olivier 

Functions:

- **threshold**  
- **detect** 
- **block_average**  work in progress


Install
-------

You can install the latest version of xmhw from PyPI::

    python -m pip install xmhw

Or directly from conda (coecms channel)::

    conda install -c coecms -c conda-forge xmhw

To install an unstable version or a different branch::

    git clone https://github.com/coecms/xmhw
    cd xmhw
    git checkout <branch-name>
    python -m pip install .

---------------------
Working on NCI server
---------------------

Xmhw is pre-installed into a Conda environment at NCI. Load it with::

    module use /g/data3/hh5/public/modules
    module load conda/analysis3-unstable

NB You need to be a member of hh5 to load the modules
