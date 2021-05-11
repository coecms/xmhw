=============================
 `xmhw <https://xmhw.readthedocs.io/en/stable>`_
=============================

XMHW - Xarray based Marine HeatWave code -  

.. image:: https://readthedocs.org/projects/xmhw/badge/?version=latest
  :target: https://xmhw.readthedocs.io/en/stable/

.. content-marker-for-sphinx

XMHW identifies marine heatwaves from a timseries of Sea Surface Temperature data and calculates statistics associated to the detected heatwaves. It is based on the`marineHeatWaves code <https://github.com/ecjoliver/marineHeatWaves/>`_ by Eric Olivier 

It has two main functions:

- **threshold**  
- **detect**  raijin projects: 0i10 for replicas 

We are working ona dding the Blockaverage and ... functions also absed on the original MHW

-------
Install
-------

XMHW is pre-installed into a Conda environment at NCI. Load it with::

    module use /g/data3/hh5/public/modules
    module load conda/analysis3-unstable

Or clone the github repository::

    git clone 
    cd xmhw
    python setup.py install ( --user ) if you want to install it in ~/.local
---
Use
---

clef cmip5
~~~~~

Find CMIP5 files matching the constraints::

    clef cmip5 --model BCC-CSM1.1 --variable tas --experiment historical --table day

You can filter CMIP5 by the following terms:
 
 * ensemble/member
 * experiment
 * experiment-family
 * institution
 * model
 * table/cmor_table
 * realm
 * frequency
 * variable
 * cf-standard-name

See ``clef cmip5 --help`` for all available filters and their aliases

   ``--latest`` will check the latest versions of the datasets on the ESGF
website, and will only return matching files

It will return a path for all the files available locally at NCI and a dataset-id for the ones that haven't been downloaded yet.

You can use the flags ``--local`` and ``--missing`` to return respectively only the local paths or the missing dataset-id::

    clef --local cmip5 --model MPI-ESM-LR --variable tas --table day
    clef --missing cmip5 --model MPI-ESM-LR --variable tas --table day

NB these flags come immediately after the command "clef" and before the sub-command "cmip5" or "cmip6". They are also clearly mutually exclusive.
You can repeat arguments more than once:: 

    clef --missing cmip5 --model MPI-ESM-LR -v tas -v tasmax -t day -t Amon

clef cmip6
~~~~~

You can filter CMIP6 by the following terms:
 
 * activity
 * experiment
 * institution
 * source_type 
 * model
 * member
 * table
 * realm
 * frequency
 * variable
 * version

See ``clef cmip6 --help`` for all available filters

-------
Develop
-------

Development install::

    conda env create -f conda/dev-environment.yml
    source activate clef-dev
    pip install -e '.[dev]'

The `dev-environment.yml` file is for speeding up installs and installing
packages unavailable on pypi, `requirements.txt` is the source of truth for
dependencies.

To work on the database tables you may need to start up a test database.

You can start a test database either with Docker::

    docker-compose up # (In a separate terminal)
    psql -h localhost -U postgres -f db/nci.sql
    psql -h localhost -U postgres -f db/tables.sql
    # ... do testing
    docker-compose rm

Or with Vagrant::

    vagrant up
    # ... do testing
    vagrant destroy

Run tests with py.test (they will default to using the test database)::

    py.test

or connect to the production database with::

    py.test --db=postgresql://clef.nci.org.au/postgres

Build the documentation using Sphinx::

    python setup.py build_sphinx
    firefox docs/_build/index.html

New releases are packaged and uploaded to anaconda.org by CircleCI when a new
Github release is made

Documentation is available on ReadTheDocs, both for `stable
<https://clef.readthedocs.io/en/stable/>`_ and `latest
<https://clef.readthedocs.io/en/latest/>`_ versions.
