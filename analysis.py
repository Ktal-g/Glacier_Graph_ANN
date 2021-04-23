# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:31:52 2021

@author: Jack
"""
#%%Import moduals
import numpy as np
import geopandas as gpd
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
import networkx as nx
import numpy as np
import sys
import tensorflow as tf
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/Jack/glacier_proj_home/Codeing_files/functions')

tf.compat.v1.enable_eager_execution()

from glacial_network import glacier_network
#%%define glaciers
glacier_list = [
'RGI60-13.37523',
'RGI60-13.30888',
'RGI60-13.26415',
'RGI60-13.54431',
'RGI60-13.43528',
'RGI60-14.04668',
'RGI60-14.02150',
'RGI60-14.01670',
'RGI60-14.00449',
'RGI60-14.03334',
'RGI60-15.06881',
'RGI60-15.06977',
'RGI60-15.06720', #13 originat tuning sample size
'RGI60-13.53689',
'RGI60-13.52478',
'RGI60-13.53689',
'RGI60-13.54059',
'RGI60-14.02849',
'RGI60-14.02653',
'RGI60-14.03405', #20
'RGI60-14.01226',
'RGI60-14.04477',
'RGI60-14.04638',
'RGI60-14.04404',
'RGI60-14.04593',
'RGI60-14.04411',
'RGI60-14.05872',
'RGI60-14.04477',
'RGI60-14.03893',
'RGI60-14.06580',
'RGI60-14.06794',
'RGI60-14.07239',
'RGI60-14.06425',
'RGI60-14.07073',
'RGI60-14.07098',
'RGI60-14.06675',
'RGI60-14.05964',
'RGI60-14.07022',
'RGI60-14.07524',
'RGI60-14.08555',
'RGI60-14.07794',
'RGI60-14.07987',
'RGI60-14.07756',
'RGI60-14.08081',
'RGI60-14.08184',
'RGI60-14.08520',
'RGI60-14.08634',
'RGI60-14.08819',
'RGI60-14.08295',
'RGI60-14.09952', #50
'RGI60-15.03473',
'RGI60-15.02846',
'RGI60-14.26942',
'RGI60-15.01485',
'RGI60-15.04541',
'RGI60-15.04121',
'RGI60-15.03619',
'RGI60-15.01152',
'RGI60-15.06574',
'RGI60-15.02709',
'RGI60-15.00621' #61
]

file_id = [
'2220to2250',
'2220to2250',
'2220to2250',
'2340to2370',
'2280to2310',
'2400to2430',
'2370to2400',
'2370to2400',
'2370to2400',
'2370to2400',
'2580to2610',
'2580to2610',
'2580to2610', #originat tuning sample size
'2340to2370',
'2340to2370',
'2340to2370',
'2340to2370',
'2370to2400',
'2370to2400',
'2370to2400',
'2370to2400',
'2400to2430',
'2400to2430',
'2400to2430',
'2400to2430',
'2400to2430',
'2400to2430',
'2400to2430',
'2400to2430',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2430to2460',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2460to2490',
'2490to2520',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580',
'2550to2580'
]

evaluation_glaciers = [
    "RGI60-14.07524", #Siachen Glacier
    'RGI60-01.21014', #Carroll glacier 
    "RGI60-01.20983", #Sea Otter glacier
    "RGI60-14.00005", #Biafo glacier # catchment 17
    'RGI60-15.09991', #Rongbuk glacier
    'RGI60-15.03473' #Ngozumpa
    ]
evaluation_glacier_file_ids = [
    "2460to2490",#Siachen Glacier
    '390to420', #Carroll glacier
    "360to390", #Sea Otter glacier
    "2370to2400", #Biafo glacier
    '2580to2610', #Rongbuk glacier
    '2550to2580'
    ]
#%%
networks = glacier_network()

#%%
#%%
#Training glaciers
for file_i, glacier_id in enumerate(glacier_list):
        
    print("file num and glacier: ", file_i, glacier_id)
        
    if file_i < 200: 
            networks.Add_GlacierFromFileId(glacier_id, file_id = file_id[file_i], training = True)
        
#%%Analysis glaciers
for file_i, glacier_id in enumerate(evaluation_glaciers):
    print("file num and glacier: ", file_i, glacier_id)
        
    networks.Add_GlacierFromFileId(glacier_id, file_id = evaluation_glacier_file_ids[file_i], training = False)
#%%
#features
node_features = ['width', 'mean velocity', 'dS', 'dY']
edge_features = []#['area', '_precip', 'area_temp', 'P_diff']
inflow_features=['mean thickness']

#ANN model hyper params
n_layers = 6
n_nodes = 25
exclude_filter_lowwer = 0
exclude_filter_upper = 0

#Training hyperparams
optimizer_num = [3,3]   #optimizer_names = 'Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD'
epochs = 80      #number of times to train on training data
drop_frac = [0.02, 0.05]
validation_split =[0.3,0.3]

#%%
#"informed-model, training samples: 20 (1)", "informed-model, training samples: 20 (2)", "informed-model, training samples: 20 (3)"
#"informed-model, training samples: 25 (1)", "informed-model, training samples: 25 (2)", "informed-model, training samples: 25 (3)"
#"informed-model, training samples: 30 (1)", "informed-model, training samples: 30 (2)", "informed-model, training samples: 30 (3)"
#"informed-model, d=0 training samples: 30 (1)", "informed-model, d=0 training samples: 30 (2)", "informed-model, d=0 training samples: 30 (3)"
#"blind-model, training samples: 20 (1)", "blind-model, training samples: 20 (2)", "blind-model, training samples: 20 (3)"
#"blind-model, training samples: 25 (1)", "blind-model, training samples: 25 (2)", "blind-model, training samples: 25 (3)"
#"blind-model, training samples: 30 (1)", "blind-model, training samples: 30 (2)", "blind-model, training samples: 30 (3)"
model_names = ["informed-model, d=.05 training samples: 20, filter 100 to 150"]
parameters = [[0.02, 0.5, 2, 'relu', True]]


#model_names = ['informed: Adam small', 'blind: Nadam small']#['informed: Adagrad', 'informed: Adadelta', 'informed: Adam', 'informed: Nadam', 'informed: RMSprop', 'informed: SGD', 'informed: NSGD']#, 'informed-model: sigmoid, d=.01']
#parameters = [[0.02,0.5, 2, True],[0.05, 0.5, 3, False]]

#%%train model
for i, model in enumerate(model_names):
    networks.Train_Model(model_name = model_names[i],
                         training_glaciers =[
'RGI60-13.37523',
'RGI60-13.30888',
'RGI60-13.26415',
'RGI60-13.54431',
'RGI60-13.43528',
'RGI60-14.04668',
'RGI60-14.02150',
'RGI60-14.01670',
'RGI60-14.00449',
'RGI60-14.03334',
'RGI60-15.06881',
'RGI60-15.06977',
'RGI60-15.06720', #13 originat tuning sample size
'RGI60-13.53689',
'RGI60-13.52478',
'RGI60-13.53689',
'RGI60-13.54059',
'RGI60-14.02849',
'RGI60-14.02653',
'RGI60-14.03405', #20
],
                         node_feature_vairables=node_features,
                         inflow_features=inflow_features,
                         edge_features=edge_features, 
                         epochs=epochs, n_layers = n_layers, n_nodes = n_nodes,
                         exclude_filter_lowwer = 100, exclude_filter_upper = 150,
                         drop_frac = parameters[i][0], validation_split = parameters[i][1], optimizer_num = parameters[i][2], activation_function= parameters[i][3],
                         include_edge_features=parameters[i][4])


#%%Pop glaciers that don't work with training:
#remove glacier: 'RGI60-14.05872', 'RGI60-15.00621'
remove_glacier = 'RGI60-14.05872'
if remove_glacier in networks.nxGraphDict.keys(): 
    networks.nxGraphDict.pop(remove_glacier)
    print("removed glacier: ", remove_glacier)
remove_glacier = 'RGI60-15.00621'
if remove_glacier in networks.nxGraphDict.keys(): 
    networks.nxGraphDict.pop(remove_glacier)
    print("removed glacier: ", remove_glacier)
#%%
networks.plot_epoch_RMSE(vairable='train size',models=["informed-model, training samples: 20 (1)", "informed-model, training samples: 20 (2)", "informed-model, training samples: 20 (3)"
                                                       ,"informed-model, training samples: 25 (1)", "informed-model, training samples: 25 (2)", "informed-model, training samples: 25 (3)"
                                                       ,"informed-model, training samples: 30 (1)", "informed-model, training samples: 30 (2)","informed-model, training samples: 30 (3)"], figsize=(12,10))
#%%

glacier_to_test_flowline_thickness = {'Biafo glacier': {'RGI_id':"RGI60-14.00005", "main flowline":'17'},
                                      'Siachen Glacier':{'RGI_id':"RGI60-14.07524", "main flowline":'20'},
                                      'Rongbuk Glacier':{'RGI_id':'RGI60-15.09991', "main flowline":'5'},
                                      'Ngozumpa':{'RGI_id':'RGI60-15.03473', "main flowline":'6'}}

networks.Plot_Network(glacier_to_test_flowline_thickness['Ngozumpa']['RGI_id'], 'mean thickness', plot=True)

#%%
#from the network choose the flowline/catchment number
for glacier in glacier_to_test_flowline_thickness.keys():
    networks.Plot_GlacierThickness(RGI_id = glacier_to_test_flowline_thickness[glacier]['RGI_id'], 
                                   catchment_number = glacier_to_test_flowline_thickness[glacier]["main flowline"],
                                   glacier_name = glacier,
                                   model_names=model_names,
                                   model_custom_names=[])

#%%
glacier = 'Siachen Glacier'
print(glacier_to_test_flowline_thickness[glacier]["main flowline"])
#%%
networks.Plot_GlacierThickness(RGI_id = glacier_to_test_flowline_thickness[glacier]['RGI_id'], 
                                   catchment_number = glacier_to_test_flowline_thickness[glacier]["main flowline"],
                                   glacier_name = glacier,
                                   model_names=["informed-model, d=.02 training samples: 20, filter 100 to 150", "blind-model, d=.05 training samples: 20, filter 100 to 150"],
                                   model_custom_names=[])
#%%

#define custom model names
model_custom_names = []
#define glaciers to test RMSE sensitivity to features
test_glaciers = ["RGI60-14.07524", #Siachen Glacier
    "RGI60-14.00005", #Biafo glacier # catchment 17
    'RGI60-15.09991', #Rongbuk glacier
    'RGI60-15.03473']
#define target feature
target_feature = 'mean thickness'

#%%

networks.Test_ModelFeature(feature = 'mean thickness',
                           models=["informed-model, d=.02 training samples: 20, filter 100 to 150", "blind-model, d=.05 training samples: 20, filter 100 to 150"],
                           model_custom_names=["informed-model, training samples: 20", "blind-model, training samples: 20"],
                           test_glaciers = test_glaciers, training_glaciers=['RGI60-13.37523',
'RGI60-13.30888',
'RGI60-13.26415',
'RGI60-13.54431',
'RGI60-13.43528',
'RGI60-14.04668',
'RGI60-14.02150',
'RGI60-14.01670',
'RGI60-14.00449',
'RGI60-14.03334',
'RGI60-15.06881',
'RGI60-15.06977',
'RGI60-15.06720', #13 originat tuning sample size
'RGI60-13.53689',
'RGI60-13.52478',
'RGI60-13.53689',
'RGI60-13.54059',
'RGI60-14.02849',
'RGI60-14.02653',
'RGI60-14.03405'], 
                           feature_bin_size=20, bar_width = 8, tick_num = 10,
                           figsize=(14,8))









