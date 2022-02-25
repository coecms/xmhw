#!/bin/bash
pip install coverage pytest-cov
py.test --cov=xmhw --cov-report xml:/tmp/artefacts/tests/pytest/coverage.xml --junit-xml /tmp/artefacts/tests/pytest/results.xml
