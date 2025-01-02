mwspectk.py
===================================

.. toctree::
    Home <self>
    usage

**mwspectk.py** is a Python library that provides tools to analyze
Fourier-transform microwave spectra using data analytical methods
rather than quantum mechanical methods.

| mwspectk.py is designed to be used in one of two ways:
* As a generic library, providing functions and classes for another program
* A library for generating simple and readable scripts for simple tasks

| The abilities of mwspectk.py are not limited to:
* Loading several spectral datatypes into Python
* Peak picking experimental spectra
* Identifying peaks that share the same frequency across different spectra
* Erasing peaks from experimental spectra
* Removing peaks in one spectrum from another
* Performing intensity ratio calculations
* Finding constant difference patterns **(WIP)**

.. note::
    This project will be updated sporadically if I think theres changes
    that are necessary. I'll make sure the code works properly, but I'm
    not the best at code deployment. Most of the time compilation and
    setup will be on the user end.