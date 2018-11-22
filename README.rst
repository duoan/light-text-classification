=========================
Light Text Classification
=========================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires| |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations| |commits-since| |zenodo|

.. |docs| image:: https://readthedocs.org/projects/light-text-classification/badge/?style=flat
    :target: https://readthedocs.org/projects/light-text-classification
    :alt: Documentation Status


.. |travis| image:: https://travis-ci.org/classtag/light-text-classification.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/classtag/light-text-classification

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/classtag/light-text-classification?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/classtag/light-text-classification

.. |requires| image:: https://requires.io/github/classtag/light-text-classification/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/classtag/light-text-classification/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/classtag/light-text-classification/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/classtag/light-text-classification

.. |version| image:: https://img.shields.io/pypi/v/light-text-classification.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/light-text-classification

.. |commits-since| image:: https://img.shields.io/github/commits-since/classtag/light-text-classification/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/classtag/light-text-classification/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/light-text-classification.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/light-text-classification

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/light-text-classification.svg
    :alt: Supported versions
    :target: https://pypi.org/project/light-text-classification

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/light-text-classification.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/light-text-classification

.. |zenodo| image:: https://zenodo.org/badge/21369/classtag/light-text-classification.svg
    :alt: Release management
    :target: https://zenodo.org/badge/latestdoi/21369/classtag/light-text-classification

.. end-badges

A lightbox library for text classification, collected state-of-art solution from recent years.

* Free software: MIT license

Installation
============

::

    pip install light-text-classification

Documentation
=============


https://light-text-classification.readthedocs.io/en/latest/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 120
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
