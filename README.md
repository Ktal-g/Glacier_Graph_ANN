# Glacier_Graph_ANN
Machine model that trains on directional graph data

Feature_inputs.ipynb
  - Creates glacier shp files for flowlines and catchments. 
  - Open from the oggm binder test space: https://docs.oggm.org/en/latest/cloud.html. 
  
glacier_network.py
  - contains class glacier_network:
    - creates glacier graph and keras sequencial model

network_functions.py
  - contains functions used for proximating air teperature, pressure and precipitation at a coordinate and altitude. Air T and precip are gotten from a netCDF file: air temp from http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html: http://schubert.atmos.colostate.edu/~cslocum/code/air.sig995.2012.nc. 
