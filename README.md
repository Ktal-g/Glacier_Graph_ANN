# Glacier_Graph_ANN
Machine model that trains on directional graph data

Note: this is a work in progress, not all resources are given and I haven't had time to make "quality of life" improvements for general use. Please contant me at s1533132@ed.ac.uk if assistance is required.

## Contents:
Feature_inputs.ipynb
  - Creates glacier shp files for flowlines and catchments. 
  - Open from the oggm binder test space: https://docs.oggm.org/en/latest/cloud.html. 
  
glacier_network.py
  - contains class glacier_network:
    - creates glacier graph and keras sequencial model

network_functions.py
  - contains functions used for aproximating air teperature, pressure and precipitation at a coordinate and altitude. Air T and precip are gotten from a netCDF file: air temp from http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html: http://schubert.atmos.colostate.edu/~cslocum/code/air.sig995.2012.nc. 

analysis.py
  - builds keras sequencial model with glacier_network
  - plots results
