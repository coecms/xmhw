#!/usr/bin/env python
#
# Uses Python Build Reasonableness https://docs.openstack.org/developer/pbr/
# Add configuration to `setup.cfg`

from setuptools import setup

setup(
        setup_requires=['pbr', 'setuptools'],
        pbr=True,
        )

