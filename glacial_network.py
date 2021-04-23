# -*- coding: utf-8 -*-


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

from FnExtract_raster_data import haversine
from FnExtractLineSegment import (Calculate_flowline_crossections)
from network_functions import (get_precipitaion_and_temp, 
                               height_to_pressure_std)
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD

from keras.models import Sequential
from keras.layers import Dense, Dropout

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#%%
class glacier_repository(object):
    def __init__(self, path_to_points, path_to_catchment_polygons):
        print("glacier_repository object created")
        
    def Define_DataFrames(self, path_to_points, path_to_catchment_polygons, RGI_id): #!!!Add a way to automatically get the glacier data from the RGIid. ATM you need to make sure the glacier is in the gpd provided
        """
        From file paths reads shp data and makes geopandas dataframes for the 
        glacier RGI_id.
        """
        catchments = gpd.read_file(path_to_catchment_polygons, encoding="utf-8").set_index('index')
        catchments = catchments.set_crs("EPSG:4326")
        
        points = gpd.read_file(path_to_points).set_index('index')
        points = points.set_crs("EPSG:4326")
        
        glacier_catchments = catchments[catchments['RGI'] == RGI_id]
        glacier_points = points[points['RGIid'] == RGI_id]
        
        return glacier_catchments, glacier_points
    
    def Get_YPredictionAndTrue(self, RGI_id, model_name, y_true = 'mean thickness'):
        pred_and_true = np.zeros((len(self.nxGraphDict[RGI_id]['graph'].nodes),2))
        
        for i, node in enumerate(self.nxGraphDict[RGI_id]['graph'].nodes):
            pred_and_true[i, 0] = self.nxGraphDict[RGI_id]['graph'].nodes[node][model_name]
            pred_and_true[i, 1] = self.nxGraphDict[RGI_id]['graph'].nodes[node][y_true]
            
        return pred_and_true
    
    def Get_NodeFeatures(self, RGI_id, node, features = ['altitude', 'width', 'mean velocity']):
        """
        Returns numpy array of features of node 'node'. Defalt featrues are: 'altitude', 'width', 'mean velocity'
        """
        feature_array = np.zeros(len(features))
        
        for feature_index in range(len(features)):
            
            feature_array[feature_index] = self.nxGraphDict[RGI_id]['graph'].nodes[node][features[int(feature_index)]]
        
        return feature_array
    
    def Get_EdgeFeatures(self, RGI_id, in_node, current_node, features):
        """
        Returns numpy array of features of the edge going from 'in_node' to 'current_node'
        """
        feature_array = np.zeros(len(features))
        
        for feature_index in range(len(features)):
            
            feature_array[feature_index] = self.nxGraphDict[RGI_id]['graph'][in_node][current_node][features[int(feature_index)]]
        
        return feature_array
    
    def Plot_Network(self, RGI_id, feature = 'altitude', plot = False):
        
        edge_trace = self.Create_EdgeTrace(RGI_id)
        
        node_trace = self.Create_NodeTrace(RGI_id)
        
        #To add colour and test to nodes:
        node_adjacencies = []
        node_colour_features = []
        node_text = []
        
        for num, adjacencies in enumerate(self.nxGraphDict[RGI_id]['graph'].adjacency()):
            catchment_number = self.nxGraphDict[RGI_id]['graph'].nodes[adjacencies[0]]['catchment']
            inflow = self.nxGraphDict[RGI_id]['graph'].nodes[adjacencies[0]]['inflow']
            node_colour_feature = self.nxGraphDict[RGI_id]['graph'].nodes[adjacencies[0]][feature]
            precipitation = self.nxGraphDict[RGI_id]['graph'].nodes[adjacencies[0]]['precipitation']
            temp = self.nxGraphDict[RGI_id]['graph'].nodes[adjacencies[0]]['surface temp']
            
            if type(node_colour_feature) == np.ndarray:
                node_colour_feature = node_colour_feature.item(0)
            
            node_colour_features.append(node_colour_feature)
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('node ' + str(adjacencies[0]) +
                             ': # of inflows: ' + str(len(inflow)) +
                             ', catchment: ' + catchment_number + 
                             ', {} '.format(feature) + str((node_colour_feature)) +
                             'precip: ' + str(precipitation) +
                             'temp: ' + str(temp))
            
            if len(adjacencies[1]) == 0:
                self.nxGraphDict[RGI_id]['end_node'] = str(adjacencies[0])
                
        node_trace.marker.color = node_colour_features
        node_trace.text = node_text
        if plot == True:
            fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='<br>glacier directional network, {}'.format(feature),
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                #annotations=[ dict(text="glacier directional network", showarrow=True,xref="paper", yref="paper", x=0.04, y=-0.002 ) ],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            fig.show()
    
    def Get_GlacierNodes(self, RGI_id):
        """
        Function to get x and y positions for plotting networkx graph
        """
        self.nxGraphDict[RGI_id]['node_x'] = []
        self.nxGraphDict[RGI_id]['node_y'] = []
        #for later use in plotting flowlines: to store catchment node starts
        self.nxGraphDict[RGI_id]['catchment node start'] = {}
        
        for node in self.nxGraphDict[RGI_id]['graph'].nodes():
            x, y = self.nxGraphDict[RGI_id]['graph'].nodes[node]['pos']
            self.nxGraphDict[RGI_id]['node_x'].append(x)
            self.nxGraphDict[RGI_id]['node_y'].append(y)
            if len(self.nxGraphDict[RGI_id]['graph'].nodes[node]['inflow']) == 0:
                
                catchment = self.nxGraphDict[RGI_id]['graph'].nodes[node]['catchment']
                self.nxGraphDict[RGI_id]['catchment node start'][catchment] = node
        
    
    def Build_GlacierEdges(self, RGI_id):
        self.nxGraphDict[RGI_id]['edge_x'] = []
        self.nxGraphDict[RGI_id]['edge_y'] = []
        self.nxGraphDict[RGI_id]['glacier_catchments_inflow'] = []
        #for catchnent in glacier
        for catchment_number, catchment_index in enumerate(self.nxGraphDict[RGI_id]['catchments'].index.tolist()):   
            catchment_num = catchment_index[14:]
            segments_per_flowline = int(self.nxGraphDict[RGI_id]['catchments'].loc[catchment_index]["segments"])
            inflows = 0
            #for segment in catchment
            for segment_index in range(0, segments_per_flowline - 1):
                #indexintg for segment
                segments_df_index0 = str(catchment_index + str(segment_index))
                segments_df_index1 = str(catchment_index + str(segment_index + 1))
                
                #indexintg for for the graph
                graph_index0 = catchment_num + str(segment_index)
                graph_index1 = catchment_num + str(segment_index + 1)
                
                inflows += self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]["inflow"] #.loc[segments_df_index0]["inflow"] is a 0 or 1 value => inflows is 0 ir 1 as well
                x0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]["Longitude"]
                y0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]["Latitude"]
                x1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]["Longitude"]
                y1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]["Latitude"]
                #get edge data
                #widths
                width0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['width, Y (']
                width1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['width, Y (']
                #edge length = (horizontal dist**2 + vertical dist**2 )**0.5
                horizontal_len = haversine([x0, y0], [x1, y1])
                horizontal_len_2 = (horizontal_len)**2
                altitude_dif_2 = (self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude ('] - self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['altitude ('])**2
                
                altitude0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude (']
                altitude1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['altitude (']
                mean_altitude = (altitude0 + altitude1)/2
                
                edge_length = (horizontal_len_2 + altitude_dif_2)**(0.5)
                #area = mean segment width x edge length
                mean_width = (width0+width1)/2
                edge_area = mean_width*edge_length/1000000 #divide by 1000000, it was dominating the model
                
                #mean surface mass balance
                precipitation_0, temp_0 = get_precipitaion_and_temp([x0, y0]) #Resolution is very bad, so cant tell the diff between node 0 and 1
                preassure_diff_mean = height_to_pressure_std(mean_altitude)
                edge_total_precip = precipitation_0*edge_area
                edge_total_temp = temp_0*edge_area
                
                self.nxGraphDict[RGI_id]['edge_x'].append(x0)
                self.nxGraphDict[RGI_id]['edge_x'].append(x1)
                self.nxGraphDict[RGI_id]['edge_x'].append(None)
                self.nxGraphDict[RGI_id]['edge_y'].append(y0)
                self.nxGraphDict[RGI_id]['edge_y'].append(y1)
                self.nxGraphDict[RGI_id]['edge_y'].append(None)
                
                self.nxGraphDict[RGI_id]['graph'].add_edge(graph_index0, graph_index1, catchment=catchment_num, width = mean_width, 
                                                           hor_len = horizontal_len, area = edge_area,
                                                           area_precip = edge_total_precip, area_temp = edge_total_temp, 
                                                           P_diff = preassure_diff_mean)
                self.nxGraphDict[RGI_id]['graph inverse'].add_edge(graph_index1, graph_index0, catchment=catchment_num)
                
                self.Add_NodeFeatures(graph_index0, segments_df_index0, RGI_id)
                self.Add_NodeFeatures(graph_index1, segments_df_index1, RGI_id)
                
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0]['pos'] = [x0, y0]
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['pos'] = [x1, y1]
                
                if 'inflow' not in self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0].keys():
                    self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0]['inflow'] = []
                
                if 'inflow' not in self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1].keys():
                    self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['inflow'] = []
                
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['inflow'].append(graph_index0)
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0]['catchment'] = catchment_num
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['catchment'] = catchment_num
                #Add precip and temp
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0]['precipitation'] = precipitation_0
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index0]['surface temp'] = temp_0
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['precipitation'] = precipitation_0
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['surface temp'] = temp_0
                
                
            self.nxGraphDict[RGI_id]['glacier_catchments_inflow'].append(inflows)
                
    def Create_NodeTrace(self, RGI_id):
        node_trace = go.Scatter(x=self.nxGraphDict[RGI_id]['node_x'], y=self.nxGraphDict[RGI_id]['node_y'],
                                     mode='markers',
                                     hoverinfo='text',
                                     marker=dict(showscale=True,
                                                 colorscale='YlGnBu',
                                                 reversescale=True,
                                                 color=[],
                                                 size=10,
                                                 colorbar=dict(thickness=15,
                                                               title='glacier {} segments'.format(RGI_id),
                                                               xanchor='left',
                                                               titleside='right'),
                                                 line_width=2))
        return node_trace
    
    def Create_EdgeTrace(self, RGI_id):
        edge_trace = go.Scatter(x=self.nxGraphDict[RGI_id]['edge_x'], y=self.nxGraphDict[RGI_id]['edge_y'],
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines')
        return edge_trace
    
    def Find_TributaryPoints(self, RGI_id):
        
        altitudes = []
        Glacier_branches = []
        reorganized_inflows = {}
        i = 0
        
        while i < len(self.nxGraphDict[RGI_id]['glacier_catchments_inflow']):
            #if the number of inflows is in the dictionary: Add the catchment index to the list in the dictionary
            if self.nxGraphDict[RGI_id]['glacier_catchments_inflow'][i] in reorganized_inflows.keys():
                reorganized_inflows[self.nxGraphDict[RGI_id]['glacier_catchments_inflow'][i]].append(i)
            #if not: create a dictionary list object with key the num of inflows
            else:
                reorganized_inflows[self.nxGraphDict[RGI_id]['glacier_catchments_inflow'][i]] = [i]
            i+= 1
            
        for inflow_number in set(reorganized_inflows.keys()):
            for glacier_index in reorganized_inflows[inflow_number]:
                catchment_num = str(int(glacier_index))
                #get coord of last node:
                catchment_index = RGI_id + catchment_num
                last_node_in_catchment = str(int(self.nxGraphDict[RGI_id]['catchments'].loc[catchment_index]["segments"] - 1))
                segment_id = catchment_index+last_node_in_catchment
                
                altitudes.append(self.nxGraphDict[RGI_id]['points'].loc[segment_id]['altitude ('])
                
                graph_index0 = catchment_num + last_node_in_catchment
                x0 = self.nxGraphDict[RGI_id]['points'].loc[segment_id]["Longitude"]
                y0 = self.nxGraphDict[RGI_id]['points'].loc[segment_id]["Latitude"]
                
                Glacier_branches.append([graph_index0, [x0, y0]])
        
        #drop the end point with lowest altitude
        branch_to_drop = Glacier_branches[np.argmin(altitudes)]
        Glacier_branches.remove(branch_to_drop)
        
        self.nxGraphDict[RGI_id]['tributary_points'] = Glacier_branches
    
    def Create_TributaryEdges(self, RGI_id):
        """
        Function to create edges between tributary flowlines and inflow nodes.
        This is done by constructing an array of possible inflow nodes for 
        each tributary node (inflow nodes with aaltitude < tributary node), 
        then creating an edge between the tributary noded and the closest node 
        in possible inflow nodes.
        """
        
        tributary_nodes_to_connect = []
        
        for node_id, coord in self.nxGraphDict[RGI_id]['tributary_points']:
            tributary_nodes_to_connect.append(node_id)
        
        for catchment_index0 in self.nxGraphDict[RGI_id]['catchments'].index.tolist():  
            
            catchment_num0 = catchment_index0[14:]
            last_node_in_catchment0 = str(int(self.nxGraphDict[RGI_id]['catchments'].loc[catchment_index0]["segments"] - 1))
            graph_index0 = catchment_num0 + str(last_node_in_catchment0)
            segments_df_index0 = str(catchment_index0 + str(last_node_in_catchment0))
            tributary_altitude0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude (']
            
            
            if graph_index0 in tributary_nodes_to_connect:
                #Create list for possible inflow index and coords
                inflow_index_and_coord = []
                
                #loop through other catchments and add nodes that have an inflow to a list, which have a lowwer altitude.
                for catchment_index1 in self.nxGraphDict[RGI_id]['catchments'].index.tolist():
                    #for different catchments if
                    catchment_num1 = catchment_index1[14:]
                    
                    if catchment_num1 != catchment_num0:
                        last_node_in_catchment1 = str(int(self.nxGraphDict[RGI_id]['catchments'].loc[catchment_index1]["segments"] - 1))
                        for segment_index1 in range(0, int(last_node_in_catchment1)):
                            segments_df_index1 = str(catchment_index1 + str(segment_index1))
                            tributary_altitude1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude (']
                            if int(self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]["inflow"]) == 1 :
                                if tributary_altitude1 <= tributary_altitude0:
                                    #add
                                    coord1 = [self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]["Longitude"], 
                                              self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]["Latitude"]]
                                    
                                    inflow_index_and_coord.append([segments_df_index1, coord1, catchment_num1])
                
                x0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]["Longitude"]
                y0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]["Latitude"]
                
                inflow_coord_array = np.zeros([len(inflow_index_and_coord),2])
                inflow_index_list = []
                
                for i, tributary in enumerate(inflow_index_and_coord):
                    inflow_coord_array[i] = inflow_index_and_coord[i][1]
                    inflow_index_list.append([inflow_index_and_coord[i][0], inflow_index_and_coord[i][2]])
                
                #look up closest inflow node
                distances_from_inflow = np.linalg.norm(inflow_coord_array - [x0, y0], axis = 1)
                
                closest_tributary_index = np.argmin(distances_from_inflow)
                index_of_tributary = inflow_index_list[closest_tributary_index][0] #to get the RGI index + catchment and segment number
                catchment_of_inflow = inflow_index_list[closest_tributary_index][1] #to get catchment number
                print("tributary {} flows to segment {} in catchment {}".format(graph_index0, index_of_tributary,catchment_of_inflow))
                
                graph_index1 = index_of_tributary[14:]
                
                x1 = self.nxGraphDict[RGI_id]['points'].loc[index_of_tributary]["Longitude"]
                y1 = self.nxGraphDict[RGI_id]['points'].loc[index_of_tributary]["Latitude"]
                
                #Get data for the edge connection:ç
                #widths
                width0 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['width, Y (']
                width1 = self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['width, Y (']
                mean_width = (width0+width1)/2
                
                horizontal_len = haversine([x0, y0], [x1, y1])
                horizontal_len_2 = (horizontal_len)**2
                altitude_dif_2 = (self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude ('] - self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['altitude ('])**2
                mean_altitude = (self.nxGraphDict[RGI_id]['points'].loc[segments_df_index0]['altitude ('] + self.nxGraphDict[RGI_id]['points'].loc[segments_df_index1]['altitude ('])/2
                edge_length = (horizontal_len_2 + altitude_dif_2)**(0.5)
                
                #area = mean segment width x edge length
                edge_area = mean_width*edge_length
                
                precipitation_0, temp_0 = get_precipitaion_and_temp([x0, y0]) #Resolution is very bad, so cant tell the diff between node 0 and 1
                
                preassure_diff_mean = height_to_pressure_std(mean_altitude)
                edge_total_precip = precipitation_0*edge_area
                edge_total_temp = temp_0*edge_area
                edge_area = mean_width*edge_length
                
                
                #add to graph
                self.nxGraphDict[RGI_id]['graph'].add_edge(graph_index0, graph_index1, catchment=catchment_of_inflow, width = mean_width, 
                                                           hor_len = horizontal_len, area = edge_area,
                                                           area_precip = edge_total_precip, area_temp = edge_total_temp, 
                                                           P_diff = preassure_diff_mean)
                
                self.nxGraphDict[RGI_id]['graph inverse'].add_edge(graph_index1, graph_index0, catchment=catchment_num0)
                #add graph_index0 to list of inflow sources
                self.nxGraphDict[RGI_id]['graph'].nodes[graph_index1]['inflow'].append(graph_index0)
                
                self.nxGraphDict[RGI_id]['edge_x'].append(x0)
                self.nxGraphDict[RGI_id]['edge_x'].append(x1)
                self.nxGraphDict[RGI_id]['edge_x'].append(None)
                self.nxGraphDict[RGI_id]['edge_y'].append(y0)
                self.nxGraphDict[RGI_id]['edge_y'].append(y1)
                self.nxGraphDict[RGI_id]['edge_y'].append(None)
                
    def Add_NodeFeatures(self, node_index, segment_index, RGI_id):
        self.nxGraphDict[RGI_id]['graph'].nodes[node_index]['altitude'] = self.nxGraphDict[RGI_id]['points'].loc[segment_index]['altitude (']
        self.nxGraphDict[RGI_id]['graph'].nodes[node_index]['width'] = self.nxGraphDict[RGI_id]['points'].loc[segment_index]['width, Y (']
        self.nxGraphDict[RGI_id]['graph'].nodes[node_index]['dS'] = self.nxGraphDict[RGI_id]['points'].loc[segment_index]['dS (m)']
        self.nxGraphDict[RGI_id]['graph'].nodes[node_index]['dY'] = self.nxGraphDict[RGI_id]['points'].loc[segment_index]['dY (m)']
        self.nxGraphDict[RGI_id]['graph'].nodes[node_index]['inflow_df'] = self.nxGraphDict[RGI_id]['points'].loc[segment_index]['inflow']
    
    def Extract_CrossSectionData(self, RGI_id):
        """
        Uses the function Calculate_flowline_crossections on geopandas data and 
        gets mean values of cross sectional thicknesses and velocities.
        """
        Calculate_flowline_crossections(path_to_catchment_polygons = self.nxGraphDict[RGI_id]['catchments'], #!!!go in and change refrence to glacier network
                                        segment_input_GDF_path = self.nxGraphDict[RGI_id]['points'],
                                        tif_file_end = '_thickness.tif',
                                        step = 1,
                                        plot_thicknesses = False,
                                        plot_velocities = False,
                                        run_on_single_glacier = True,
                                        RGI_id_single = RGI_id,
                                        testing = False,
                                        glacier_network = self.nxGraphDict[RGI_id]['graph'],
                                        save_gpd = False,
                                        glacier_inputs = "gpd")
    
    def Build_Model(self, input_size, model_name, 
                    n_layers = 4, n_nodes = 10, 
                    drop_frac = 0.2, optimizer_num = 2, 
                    loss_fn = root_mean_squared_error,
                    activation_fn = 'relu'):
        """
        Build keras sequencial model.
        """
        optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
        optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]
        # selecting the optimizer
        
        optimizer_val = optimizer_vals[optimizer_num]
        
        self.glacierModelDict[model_name] = {}
        
        
        self.glacierModelDict[model_name]['model'] = Sequential() #https://www.tensorflow.org/guide/keras/sequential_model#transfer_learning_with_a_sequential_model
        for layer in np.arange(n_layers):
            if layer == 0:
                self.glacierModelDict[model_name]['model'].add(Dense(n_nodes, activation= activation_fn, input_shape=(input_size,)))
            else:
                self.glacierModelDict[model_name]['model'].add(Dense(n_nodes, activation= activation_fn))
            self.glacierModelDict[model_name]['model'].add(Dropout(drop_frac))
        self.glacierModelDict[model_name]['model'].add(Dense(1, activation='linear'))
        self.glacierModelDict[model_name]['model'].summary()
        
        #Add hyper parameters
        self.glacierModelDict[model_name]['optimizer'] = optimizer_names[optimizer_num]
        self.glacierModelDict[model_name]['layers'] = n_layers
        self.glacierModelDict[model_name]['nodes per layer'] = n_nodes
        self.glacierModelDict[model_name]['drop fraction'] = drop_frac
        self.glacierModelDict[model_name]['error'] = loss_fn
        self.glacierModelDict[model_name]['activation'] = activation_fn
        
        self.glacierModelDict[model_name]['model'].compile(loss=loss_fn, 
                                                           optimizer=optimizer_val,
                                                           metrics=None,
                                                           run_eagerly=True)
        
    def Build_NxGraph(self, path_to_points, path_to_catchment_polygons, RGI_id, training, plot=True):
        """
        Function to call operations required to build a Networkx directional glacier
        graph.
        1. Make directional graph objects for the glacier and reverse glacier 
        (the reverse glacier will contain information of inflow nodes and edges,
         where the directional graph contains information on the direction of
         flow)
        2. Get cathcment and point geometry data as geopandas dataframe.
        3. Add geopandas and networkx graphs to nxGraphDict dictionary object.
        4. Build graph edges for graph and inverse graph.
        5. Find tributary flowline end nodes
        6. Join tributary nodes to inflow nodes (nodes that have an inflow).
        7. Get cross sectional data of velocity and ensemble thickness
        8. Get x and y positions for plotting networkx graph
        9. Plot the glacier graph network (Note: self.nxGraphDict[RGI_id]['end_node']
        is defined in this function and is required for later operations, so
        this function must be called. Should move this definition somewhere
        else).
        """
        
        Glacier_graph = nx.DiGraph()
        Glacier_graph_inverse = nx.DiGraph()
        
        #Get chatchments and points gpds:
        glacier_catchments, glacier_points = self.Define_DataFrames(path_to_points, path_to_catchment_polygons, RGI_id)
        
        self.nxGraphDict[RGI_id] = {'graph' : Glacier_graph, 'graph inverse' : Glacier_graph_inverse, 'points': glacier_points, 'catchments': glacier_catchments, 'training':training}
        
        #add edges
        self.Build_GlacierEdges(RGI_id)
        #find tributary points
        self.Find_TributaryPoints(RGI_id)
        #join tributary edges if there are multiple catchment areas
        if len(list(glacier_catchments.index)) != 1: self.Create_TributaryEdges(RGI_id)
        #get cross sectional data of velocity and ensemble thickness
        self.Extract_CrossSectionData(RGI_id) 
        print("created edges")
        self.Get_GlacierNodes(RGI_id)
        print("got node positions")
        self.Plot_Network(RGI_id, plot = plot)
        
        
        
class glacier_network(glacier_repository):
    def __init__(self):
        """
        Creates the object dictionaries where all glaceir network data, keras
        model data and results will be stored.
        Glaciers in RGI_id_list must have their point and catchment data in
        path_to_points, path_to_catchment_polygons.
        """
        self.glacierModelDict = {}
        self.nxGraphDict = {}
        self.model_results ={}
        
    def Add_Glacier(self, RGI_id_list, path_to_points, path_to_catchment_polygons, training):
        """
        Create a network graph with known corresponding path_to_points and
        path_to_catchment_polygons
        """
        #https://networkx.org/documentation/networkx-1.0/tutorial/tutorial.html
        #https://plotly.com/python/network-graphs/
        #can try implement https://stackoverflow.com/questions/61496833/is-there-a-way-to-read-images-stored-locally-into-plotly/61497919#61497919
        print("Adding glaciers: ", RGI_id_list)
        for glacier_id in RGI_id_list:
            self.Build_NxGraph(path_to_points, path_to_catchment_polygons, glacier_id, training=training)
        
    def Add_GlacierFromFileId(self, RGI_id, file_id, training, path_oggm_polygons = "files_glacier_shape/input_data", path_oggm_points = "files_glacier_shape/point_data"):
        
        polygon_file = "glaciers_polygons_TH20_" + file_id + ".shp"
        point_file = "segments_points_TH20_" + file_id + ".shp"
        print("point_file", point_file)
        path_to_polygons = path_oggm_polygons + "/" + polygon_file
        path_to_points = path_oggm_points + "/" + point_file
        
        self.Build_NxGraph(path_to_points, path_to_polygons, RGI_id, training=training, plot=False)
        
    def loss(self, model, x, y, training):
        #https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
        #http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/losses/MeanSquaredError.html
        #https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#speeding-up_your_training_step_with_tffunction
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        
        tf.config.run_functions_eagerly(True)
        y_pred = model(x, training=True)
        print("predicted value of {} using inputs x {}".format(y_pred, x))
        print(y_pred)
        print("true value of {}".format(y))
        return self.loss_object(y_true=y, y_pred=y_pred)
      
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            #https://www.tensorflow.org/api_docs/python/tf/GradientTape
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    def ClearNodeAttribute(self, attribute, RGI_list_to_clear = 'all'):
        if RGI_list_to_clear == 'all':
            RGI_list_to_clear = list(self.nxGraphDict.keys())
        for RGI_id in RGI_list_to_clear:
            for node_i, inflow_node in enumerate(self.nxGraphDict[RGI_id]['graph'].nodes):
                _ = self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node].pop(attribute, None)
    
    def Pred_and_True(self, RGI_id, model_name):
        """
        Returns array prtedicted and array true."""
        pred_and_true = self.Get_YPredictionAndTrue(RGI_id, model_name)
        return pred_and_true[:,0],pred_and_true[:,1]
        
    def Calculate_NodeThickness(self, RGI_id, model_name, current_node, thickness_inflow_array, includes_node_thk, features, inflow_nodes = None):
        """
        Function to calculate thickness of a node using a model
        """
        
        ############################print("calculating thickness of node: \n")
        
        if self.glacierModelDict[model_name]['previous node']:
            #############################################print("Including inflow thicknesses:\t {} from nodes {}\n".format(thickness_inflow_array, inflow_nodes))
            #compile current node features:
            node_features = self.Get_NodeFeatures(RGI_id, current_node, features)
            
            inflow_node_features = np.zeros((2, 1))
            edge_features = np.zeros((1,len(self.glacierModelDict[model_name]['edge features'])))
            
            if inflow_nodes != None: 
                edge_features = np.zeros((len(inflow_nodes),len(self.glacierModelDict[model_name]['edge features'])))
                for inflow_node_i, inflow_node in enumerate(inflow_nodes):
                    edge_features[inflow_node_i] = self.Get_EdgeFeatures(RGI_id, in_node=inflow_node, current_node=current_node, features=self.glacierModelDict[model_name]['edge features'])
                
                inflow_node_features[0] = thickness_inflow_array[0]
                inflow_node_features[1] = thickness_inflow_array[1]
                    
            model_input = np.concatenate((edge_features.sum(axis=0), inflow_node_features.sum(axis=0), node_features), axis=0)
            #Join edge and node features
            print("\n\nmodel_input:",model_input, "normalized: ", model_input / self.glacierModelDict[model_name]['features_norm'])
            print("node features {}: ".format(features), node_features)
            print("edge features {}: ".format(self.glacierModelDict[model_name]['edge features']), edge_features)
            print("previous node features {}: ".format(self.glacierModelDict[model_name]['previous node features']), inflow_node_features)       
        
        elif self.glacierModelDict[model_name]['previous node'] == False:
            model_input = self.Get_NodeFeatures(RGI_id, current_node, features = self.glacierModelDict[model_name]['node features'])
            
        target_data = self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['mean thickness']
        
        if (model_input.ndim == 1):
            model_input = np.array([model_input]) 
        
        target_data = np.array([target_data])
        if (target_data.ndim == 1):
            target_data = np.array([target_data])
        #print("model input = ", model_input)
        
        if self.glacierModelDict[model_name]['normalized']:
            print("model_input not normilized: ", model_input)
            model_input = model_input / self.glacierModelDict[model_name]['features_norm']
            print("model_input normilized: ", model_input)
            model_output0 = self.glacierModelDict[model_name]['model'].predict(model_input)
            print("model output[:] : ", model_output0[:],  model_output0[:] * self.glacierModelDict[model_name]['y_norm'])
            self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name] = model_output0 * self.glacierModelDict[model_name]['y_norm']
        
        else:
            model_output0 = self.glacierModelDict[model_name]['model'].predict(model_input)
            print("model output: ", model_output0)
            #print("\tmodel_output = ", model_output0)
            self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name] = model_output0#.numpy().item(0)
    
    def Calculate_GlacierThickness(self, RGI_id, start_node=None, 
                                   model_name=None, includes_node_thk = True,
                                   yield_catchment = None):
        """
        
        Parameters
        ----------
        model : keras.model
            Model to be used in modeing thickness of the glacier.
        includes_node_thk : Bool, default: True
            Does the keras model take thickness of the previous node into 
            account.
        features : List object, default: ['altitude', 'width', 'mean velocity']
            Needs to be the features the model takes. If includes_node_thk,
            the thickness features and edge features will be added: don't need
            to specify here.
        Returns
        -------
        
        """
        features = self.glacierModelDict[model_name]['node features']
        
        if yield_catchment != None:
            self.model_results[model_name] = {}
            self.model_results[model_name]['altitude'] = []
            self.model_results[model_name]['thickness'] = []
            self.model_results[model_name]['true thickness'] = []
            self.model_results[model_name]['horizontal distance from source'] = []
                
            self.model_results[model_name]['inflow altitude'] = []
            self.model_results[model_name]['horizontal inflow distance'] = []
            
        
        def get_inflow_node_thickness(current_node, dist_from_source):
            
            #function to model node thickness
            #get inflow nodes:
            thickness_inflow_nodes = np.zeros(2) #array to store previous node features
            edge_features = np.zeros((2, len(self.glacierModelDict[model_name]['edge features'])))
            
            inflow_nodes = []
            
            #if there is in inflow from another node
            if len(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['inflow']) != 0:
                #for inflow nodes (inflow node index, inflow node id)
                
                for node_i, inflow_node in enumerate(self.nxGraphDict[RGI_id]['graph inverse'][current_node]):
                    #get thicknesses for model
                    print("inverse test: ", inflow_node)
                    if model_name not in self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node].keys():
                        #If there is no node thickness: model thickness
                        print("Node {} has no thickness, changing current node to {}".format(inflow_node, inflow_node))
                        get_inflow_node_thickness(inflow_node, dist_from_source)
                    
                    #if there is a thickness in inflow_node, get the thickness val and add it to thickness_inflow_nodes
                    if model_name in self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node].keys():
                        #compile node features
                        #print("\n_______________________________________________________\n")
                        print("\nworking node: ", current_node)
                        print("found thickness {} in node {}, adding to input data".format(self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node][model_name],inflow_node))
                        thickness_inflow_nodes[node_i] = self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node][model_name]
                        inflow_nodes.append(inflow_node)
                    
                    #compile edge features
                    #if add edge features to model:
                        #self.get_edge features(edge coordinates, altitude) #returns temperature, precipitation, area, slope
                
                #For method 2, the model trains step by step, thus needs to extract features and mean thickness true to run outside where tf.GradientTape can watch the loss function
                #get thickness of this current_node
                print("thickness_inflow_nodes[node_i]: ", thickness_inflow_nodes)
                self.Calculate_NodeThickness(RGI_id, model_name, current_node, thickness_inflow_array = thickness_inflow_nodes, 
                                         includes_node_thk = includes_node_thk, features=features, 
                                         inflow_nodes = inflow_nodes)
                
                if self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['catchment'] == yield_catchment: #if current node is in the flowline
                    
                    #add thickness of node to the ['thickness'] list
                    self.model_results[model_name]['thickness'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name].item(0))
                    self.model_results[model_name]['altitude'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['altitude'])
                    self.model_results[model_name]['true thickness'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['mean thickness'])
                    
                    for inflow_node_id in inflow_nodes: #need this extra for loop to set the dist from source
                        if self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node_id]['catchment'] == yield_catchment:
                            dist_from_source += self.nxGraphDict[RGI_id]['graph'][inflow_node_id][current_node]['hor_len']
                            
                    #go through inflow nodes
                    for inflow_node_id in inflow_nodes: 
                        #if in flowline, update ['horizontal distance from source']
                        if self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node_id]['catchment'] == yield_catchment:
                            self.model_results[model_name]['horizontal distance from source'].append(self.model_results[model_name]['horizontal distance from source'][-1] + dist_from_source)
                            print("found node {} in catchment {}, adding thickness {} at distance {}".format(current_node, yield_catchment, self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name], dist_from_source))
                        
                        elif self.nxGraphDict[RGI_id]['graph'].nodes[inflow_node_id]['catchment'] != yield_catchment:
                            
                            self.model_results[model_name]['inflow altitude'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['altitude'])
                            
                            self.model_results[model_name]['horizontal inflow distance'].append(self.model_results[model_name]['horizontal distance from source'][-1])
                            
                            print("found inflow at node {}, adding inflow distance from source {}".format(current_node, dist_from_source))
                            
            #if inflow == 0, the node is a glacier source: calculate thickness using just node features and with thickness
            elif len(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['inflow']) == 0:
                #calculate thickness using the model
                print("Found glacier source: ", current_node)
                self.Calculate_NodeThickness(RGI_id ,model_name, current_node, thickness_inflow_array = thickness_inflow_nodes, includes_node_thk = includes_node_thk, features=features)
                
                if self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['catchment'] == yield_catchment:
                    print("yeilding catchment: ", yield_catchment)
                    print("found source node {}, adding thickness {} and distance {} from source node.".format(current_node, self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name], dist_from_source))
                    
                    self.model_results[model_name]['thickness'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node][model_name].item(0))
                    self.model_results[model_name]['altitude'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['altitude'])
                    self.model_results[model_name]['true thickness'].append(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['mean thickness'])
                    self.model_results[model_name]['horizontal distance from source'].append(0)
                    
                
        if start_node == None:
            start_node = self.nxGraphDict[RGI_id]['end_node']
        
        print("startying with node: ", start_node)
        
        get_inflow_node_thickness(start_node, dist_from_source = 0)
        
        
    def Get_FeatureData(self, RGI_id_list, node_feature_vairables = ['altitude', 'width', 'mean velocity'], 
                        node_target_vairable = ['mean thickness'], include_edge_features = True,
                        inflow_features = ['mean thickness', 'width'], edge_feature_vairables = ['width'],
                        exclude_filter_lowwer_band = 0, exclude_filter_upper_band = 10000, normalize = True):
        #https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        #https://www.tensorflow.org/tutorials/load_data/numpy
        print("Getting feature data for glaciers: ", RGI_id_list)
        
        print("Using node features: ", node_feature_vairables)
        print("Including previous node features: ", inflow_features)
        
        
        #Find the number of total
        length_of_data = 0
        for glacier_id in RGI_id_list:
            length_of_data += len(self.nxGraphDict[glacier_id]['graph'].nodes)
        
        if include_edge_features: 
            featuren_num = int(len(inflow_features)+len(edge_feature_vairables)+len(node_feature_vairables))
        else:
            featuren_num = int(len(node_feature_vairables))
        
        node_feature_data = np.zeros((length_of_data, featuren_num)) #!!! add if statement for if not including inflow data
        node_test_data = np.zeros((length_of_data, 1))
        offset_id = 0
        for RGI_id in RGI_id_list:
            #loop through ALL nodes in glacier_graph
            for node_index, current_node in enumerate(self.nxGraphDict[RGI_id]['graph'].nodes):
                
                #if the target vairable is outside the excluding filter
                if self.nxGraphDict[RGI_id]['graph'].nodes[current_node][node_target_vairable[0]] <= exclude_filter_lowwer_band or (self.nxGraphDict[RGI_id]['graph'].nodes[current_node][node_target_vairable[0]] >= exclude_filter_upper_band):
                    node_features = self.Get_NodeFeatures(RGI_id ,current_node, node_feature_vairables) #get node features
                    
                    if include_edge_features:
                        
                        inflow_node_features = np.zeros((2,len(inflow_features))) #this will contain the feature data for the features of the inflow nodes #!!max of 2
                        edge_features = np.zeros((2,len(edge_feature_vairables)))
                        
                        #if there are inflows to this node:
                        if len(self.nxGraphDict[RGI_id]['graph'].nodes[current_node]['inflow']) != 0: #add this check to avoid error in next loop
                            
                            #for every node that inflows to the current node:
                            print("self.nxGraphDict[RGI_id]['graph inverse'][current_node] = ", self.nxGraphDict[RGI_id]['graph inverse'][current_node])
                            print("RGI_id and current_node: ", RGI_id , current_node)
                            for inflow_node_i, inflow_node in enumerate(self.nxGraphDict[RGI_id]['graph inverse'][current_node]):
                                
                                inflow_node_features[inflow_node_i] = self.Get_NodeFeatures(RGI_id, node=inflow_node, features=inflow_features)
                                print("node_features at {}, {} = ".format(inflow_node_i, inflow_node), self.Get_NodeFeatures(RGI_id, node=inflow_node, features=inflow_features))
                                edge_features[inflow_node_i] = self.Get_EdgeFeatures(RGI_id, in_node=inflow_node, current_node=current_node, features=edge_feature_vairables)
                                print("edge_features at {}, {} = ".format(inflow_node_i, inflow_node), self.Get_EdgeFeatures(RGI_id, in_node=inflow_node, current_node=current_node, features=edge_feature_vairables))
                            
                            print("inflow_node_features: ",inflow_node_features)
                            print("edge_features: ",edge_features)
                        
                        node_features = np.concatenate((edge_features.sum(axis=0), inflow_node_features.sum(axis=0), node_features), axis=0)
                    
                    node_feature_data[int(node_index + offset_id)] = node_features
                    node_test_data[int(node_index + offset_id)] = self.Get_NodeFeatures(RGI_id, current_node, node_target_vairable)
                
                #if the target vairable is inside the excluding filter
                elif self.nxGraphDict[RGI_id]['graph'].nodes[current_node][node_target_vairable[0]] > exclude_filter_lowwer_band and self.nxGraphDict[RGI_id]['graph'].nodes[current_node][node_target_vairable[0]] < exclude_filter_upper_band:
                    node_test_data[int(node_index + offset_id)] = np.nan
                    
            offset_id += len(self.nxGraphDict[RGI_id]['graph'].nodes)
        
        #remove np.nan from node_test_data and the corresponding rows in node_feature_data
        rows_to_keep = np.logical_not(np.isnan(node_test_data))
        node_test_data= node_test_data[rows_to_keep] #removing null values
        
        #to index the True rows and put into new array...
        new_node_feature_data = np.zeros((len(node_test_data), featuren_num))
        
        new_node_feature_data_index = 0
        for row_num, bool_val in enumerate(rows_to_keep):
            if bool_val: 
                new_node_feature_data[new_node_feature_data_index] = node_feature_data[row_num]
                new_node_feature_data_index += 1 #update index to the next value
            
        
        node_feature_data = new_node_feature_data #removing rows from node_feature_data values
        
        if normalize==True:
            feature_normalized_array = np.linalg.norm(node_feature_data)
            y_normalized_array = np.linalg.norm(node_test_data)
            
            node_feature_data = node_feature_data / feature_normalized_array
            node_test_data = node_test_data / y_normalized_array
            
            
        return node_feature_data, node_test_data, feature_normalized_array, y_normalized_array
    
    def Train_Model(self, model_name, training_glaciers = [], epochs = 50,
                   node_feature_vairables=['altitude', 'width', 'mean velocity'],
                   target_vairable = ['mean thickness'],
                   inflow_features=['mean thickness'], edge_features=['width'], include_edge_features=True,
                   n_layers = 4, n_nodes = 10, drop_frac = 0.2, validation_split = 0.5, optimizer_num = 2,
                   exclude_filter_lowwer = 0, exclude_filter_upper = 0, activation_function = 'relu',
                   normalize_data = True, fit_epochs = 1,
                   clear_model=True):
        
        #To store training results
        if len(training_glaciers) == 0:
            for glacier_id_key in self.nxGraphDict.keys():
                if self.nxGraphDict[glacier_id_key]['training'] == True:
                    training_glaciers.append(glacier_id_key)
        
        
        trainX, trainY, trainX_norm, trainY_norm = self.Get_FeatureData(RGI_id_list = training_glaciers,
                                              node_feature_vairables=node_feature_vairables,
                                              node_target_vairable=target_vairable,
                                              include_edge_features=include_edge_features,
                                              inflow_features=inflow_features, edge_feature_vairables=edge_features,
                                              exclude_filter_lowwer_band = exclude_filter_lowwer, exclude_filter_upper_band = exclude_filter_upper,
                                              normalize = normalize_data)
        print("trainX, trainY", trainX, trainY)
        print("trainX_norm, trainY_norm", trainX_norm, trainY_norm)
        print("Training model on glaciers:\n", training_glaciers)
        
        #allow for custome fuctions to be built
        if clear_model:
            self.Build_Model(input_size = np.shape(trainX)[1], model_name = model_name, 
                             n_layers = n_layers, n_nodes = n_nodes, 
                             drop_frac = drop_frac, optimizer_num = optimizer_num,
                             activation_fn = activation_function)
        
        elif model_name not in self.glacierModelDict.keys():
            self.Build_Model(input_size = np.shape(trainX)[1], model_name = model_name, 
                             n_layers = n_layers, n_nodes = n_nodes, 
                             drop_frac = drop_frac, optimizer_num = optimizer_num,
                             activation_fn = activation_function)
        
        self.glacierModelDict[model_name]['previous node'] = include_edge_features
        self.glacierModelDict[model_name]['normalized'] = normalize_data
        self.glacierModelDict[model_name]['features_norm'] = trainX_norm
        self.glacierModelDict[model_name]['y_norm'] = trainY_norm
        self.glacierModelDict[model_name]['previous node features'] = inflow_features
        
        self.glacierModelDict[model_name]['training glaciers'] = training_glaciers
        self.glacierModelDict[model_name]['epoch losses'] = []
        self.glacierModelDict[model_name]['node features'] = node_feature_vairables
        self.glacierModelDict[model_name]['edge features'] = edge_features
        self.glacierModelDict[model_name]['target vairable'] = target_vairable[0]
        self.glacierModelDict[model_name]['validation split'] = validation_split
        self.glacierModelDict[model_name]['train size'] = len(trainY)
        self.glacierModelDict[model_name]['validation size'] = len(trainY)*validation_split
        self.glacierModelDict[model_name]['train size with v split'] = len(trainY)*(1-validation_split)
        
        epoch_acheved = 0
        for epoch in range(epochs):
            print("start of epoch {}".format(epoch))
            self.epoch_loss_avg = tf.keras.metrics.Mean()
            self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            
            self.glacierModelDict[model_name]['history'] = self.glacierModelDict[model_name]['model'].fit(trainX, trainY,
                                                                                                          epochs = fit_epochs,
                                                                                                          verbose=1,
                                                                                                          validation_split=validation_split)
            
            
                    
            #print("glacier_loss = ", glacier_loss)
            print('train_loss_1', self.glacierModelDict[model_name]['history'].history['loss'])
            print('val_train_loss', self.glacierModelDict[model_name]['history'].history['val_loss'])
            epoch_acheved += 1
            self.glacierModelDict[model_name]['epoch losses'].append(self.glacierModelDict[model_name]['history'].history['loss'])
            #train_accuracy_results.append(self.glacierModelDict[model_name]['history'].history['val_loss'])
            
        self.glacierModelDict[model_name]['epochs'] = epoch_acheved
        plt.plot(range(self.glacierModelDict[model_name]['epochs']), self.glacierModelDict[model_name]['epoch losses'])
        plt.show()
        
    def plot_epoch_RMSE(self, vairable, models, figsize = (12, 8), include_vairable = False):
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name in models:
            
            if include_vairable:
                label = model_name+ ": "+str(vairable)+" = "+ str(self.glacierModelDict[model_name][vairable])
            else:
                label = model_name
            
            plt.plot(range(self.glacierModelDict[model_name]['epochs']), self.glacierModelDict[model_name]['epoch losses'], label = label)
        
        plt.title("RMSE vs epoch")
        ax.legend(loc ="upper right")
        ax.set_ylabel("RMSE value")
        ax.set_xlabel("epoch")
        plt.show()
        
    def Plot_GlacierThickness(self, RGI_id, catchment_number, glacier_name = None ,model_names=None, model_custom_names = []):
        #have the same issue as: https://stackoverflow.com/questions/42706761/closing-session-in-tensorflow-doesnt-reset-graph
        if model_names == None:
            model_names = self.glacierModelDict.keys()
        
        if len(model_custom_names) == 0:
            model_custom_names = model_names
        #Get altitude and 'true' thickness
        
        fig = plt.figure(figsize=(12,8))
        
        for model_i, model_name in enumerate(model_names):
            self.ClearNodeAttribute(model_name, RGI_list_to_clear=[RGI_id])
            """
            model_name_results['thickness']
            model_name_results['true thickness']
            model_name_results['horizontal distance from source']
            model_results['altitude']
                
            model_name_results['inflow altitude']
            model_name_results['horizontal inflow distance']
            
            """
            
            self.Calculate_GlacierThickness(RGI_id, start_node=None, 
                                   model_name=model_name, includes_node_thk = self.glacierModelDict[model_name]['previous node'],
                                   yield_catchment = catchment_number)
            
            
            if model_i == 0: 
                plt.scatter(self.model_results[model_name]['horizontal inflow distance'], self.model_results[model_name]['inflow altitude'], c='r', label='inflow point')
                plt.plot(self.model_results[model_name]['horizontal distance from source'], self.model_results[model_name]['altitude'], label="glacier surface", marker='.')
                plt.plot(self.model_results[model_name]['horizontal distance from source'], np.array(self.model_results[model_name]['altitude']) - np.array(self.model_results[model_name]['true thickness']), label="composite model", marker='x')
            
            thickness = np.array(self.model_results[model_name]['altitude']) - np.array(self.model_results[model_name]['thickness'])
            plt.plot(self.model_results[model_name]['horizontal distance from source'], thickness, label=model_custom_names[model_i])
        
        if glacier_name == None:
            glacier_name = RGI_id
        
        plt.title("mean thickness - {}".format(glacier_name))
        
        plt.ylabel("Altitude (m)")
        plt.xlabel("horizontal distance allong flowline (m)")
        plt.legend()
        plt.show()
        
    def Test_ModelFeature(self, feature, models=[], model_custom_names = [], test_glaciers = [], training_glaciers=[], 
                          feature_bin_size = 10, bar_width = 3, tick_num = 15, figsize=(17,15), 
                          y_label="frequency",y_label2="RMS error", x_label="feature", title_name = "Error and sample plot",
                          alpha_test = 0.6, alpha_train = 0.4):
        #if not specifying: add glaciers to train and test
        if len(training_glaciers) == 0:
            for glacier_id_key in self.nxGraphDict.keys():
                if self.nxGraphDict[glacier_id_key]['training'] == True:
                    training_glaciers.append(glacier_id_key)
        
        if len(test_glaciers) == 0:
            for glacier_id_key in self.nxGraphDict.keys():
                if self.nxGraphDict[glacier_id_key]['training'] == False:
                    test_glaciers.append(glacier_id_key)
        
        #for model in models get the RMS error of the mean thickness vs feature
        #for the test glaciers
        fig, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        
        #if not adding custom names, set names of models as dict keys
        if len(model_custom_names) == 0:
            model_custom_names = models
            
        #get feature data sampled:
        test_feature_data_list = []
        
        for glacier in test_glaciers:
            
            for i, node in enumerate(self.nxGraphDict[glacier]['graph'].nodes):
                #feature data 
                test_feature_data_list.append(self.nxGraphDict[glacier]['graph'].nodes[node][feature])
                
        #store test glacier featrue data as np.ndarray...
        test_feature_data = np.array(test_feature_data_list)
        test_feature_max = np.amax(test_feature_data)
        
        #this is assuming models used the same training data:
        train_feature_data_list = []
        for glacier in training_glaciers:
            
            for i, node in enumerate(self.nxGraphDict[glacier]['graph'].nodes):
                #feature data 
                train_feature_data_list.append(self.nxGraphDict[glacier]['graph'].nodes[node][feature])
                
        #store training featrue data in np.ndarray:
        train_feature_data = np.array(train_feature_data_list)
        train_feature_max = np.amax(train_feature_data) 
        
        #plot histograms of training and testing sample data:
        max_val = np.amax([train_feature_max, test_feature_max])
        bins = np.arange(0, max_val, feature_bin_size)
        
        bin_x_pos = np.round(np.linspace(0 + feature_bin_size/2 , max_val - feature_bin_size/2, (len(bins)-1)), 2)
        bin_x_tick_locations = np.round(np.linspace(0 + feature_bin_size/2 , max_val - feature_bin_size/2, tick_num), 2)
        
        #map:
        map_test_feature_data_to_bins_index = np.digitize(test_feature_data, bins) #index in your bins of each element
        map_train_feature_data_to_bins_index = np.digitize(train_feature_data, bins) #index in your bins of each element
        
        frequency_of_test_feature_data, bins = np.histogram(test_feature_data, bins) #returns: hist: array and bin_edges (length(hist)+1)
        frequency_of_train_feature_data, bins = np.histogram(train_feature_data, bins) #returns: hist: array and bin_edges (length(hist)+1)
          
        _ = ax.bar(bin_x_pos, frequency_of_test_feature_data, width = bar_width,
                   alpha=alpha_test,
                   label='test sample data')
            
        _ = ax.bar(bin_x_pos, frequency_of_train_feature_data, width = bar_width,
                   alpha=alpha_train,
                   label='train sample data')
        #!!! Test this, the 
        #get model data
        for model_i, model_name in enumerate(models):
            #place to store data to plot:
            
            
            error_squared = np.zeros(len(bins))
            RMS_error_of_bins =  np.zeros(len(bins))
            RMS_STD_estimate = np.zeros(len(bins))
            
            for glacier in test_glaciers:
                model_glacier_list = [] #predicted glacier values
                true_glacier_list = [] #true glacier values
                error_squared_glacier = np.zeros(len(bins))
                #if there is not already a model result for this glacier.
                if model_name not in self.nxGraphDict[glacier]['graph'].nodes[self.nxGraphDict[glacier]['end_node']]:
                    self.Calculate_GlacierThickness(glacier, 
                                           model_name=model_name, includes_node_thk = self.glacierModelDict[model_name]['previous node'],
                                           yield_catchment = None)
                #for all glaciers in the test glacier
                for i, node in enumerate(self.nxGraphDict[glacier]['graph'].nodes):
                    
                    model_glacier_list.append(self.nxGraphDict[glacier]['graph'].nodes[node][model_name])
                    true_glacier_list.append(self.nxGraphDict[glacier]['graph'].nodes[node]["mean thickness"])
                    
                test_model_results = np.array(model_glacier_list)    
                test_true_y = np.array(true_glacier_list)
                
                #glacier specific np.digitize map:
                map_glacier_feature_data_to_bins_index = np.digitize(test_true_y, bins) #index in your bins of each element
                
                #for each map index, get the true and pred y of that feature value !!! probably error from here
                #put the squared error into error_squared
                for index in map_glacier_feature_data_to_bins_index:
                    true_y = test_true_y[index-1]
                    pred_y = test_model_results[index-1]
                    previous_error = error_squared_glacier[index-1] 
                    error_squared_glacier[index-1] = previous_error + float(np.square(true_y - pred_y))
                
                #!!! uncoment to see squared error ax2.plot(bins, error_squared_glacier, label="error^2 of glacier {}, model {}".format(glacier,model_custom_names[model_i]))
                #Add glacier error squared to total error squred
                error_squared = np.add(error_squared, error_squared_glacier)
                
            #sum error_squared and error_squared_glacier
            for i in range(len(frequency_of_test_feature_data)):
                number_in_bin = frequency_of_test_feature_data[i] 
                if int(number_in_bin) == 0:
                    RMS_error_of_bins[i] = -99
                else:
                    RMS_error_of_bins[i] = np.sqrt((error_squared[i]/number_in_bin))
                    
            
            ax2.plot(bins, RMS_error_of_bins, label="RMS error, model {}".format(model_custom_names[model_i]))
        
        ax2.set_ylabel(y_label2)
        ax2.set_ylim(bottom=0)
            
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_xticks(bin_x_tick_locations)
        ax.set_xlim(left=0, right=test_feature_max)
        ax.set_xticklabels(bin_x_tick_locations)
        ax2.legend(loc ="upper right")
        ax.legend(loc ="upper left")
        #ax.title(title_name)
            
        fig.tight_layout()
        plt.show()
"""
#%%
glacier_list = ['RGI60-01.21014', 
                'RGI60-01.12077', 
                'RGI60-01.10689', 
                'RGI60-01.10575', 
                'RGI60-01.12275', 
                'RGI60-01.17566', 
                'RGI60-01.17183', 'RGI60-01.16201', 'RGI60-01.16198', 'RGI60-01.17348', 'RGI60-01.23649', 'RGI60-01.26738', 'RGI60-01.27108', 'RGI60-02.03586', 'RGI60-02.05157', 'RGI60-03.02849', 'RGI60-03.02836', 'RGI60-04.03171', 'RGI60-04.03263', 'RGI60-04.05514', 'RGI60-04.06998', 'RGI60-13.37536', 'RGI60-13.37523', 'RGI60-13.30888', 'RGI60-13.26415', 'RGI60-13.54431', 'RGI60-13.43528', 'RGI60-14.00005', 'RGI60-14.02150', 'RGI60-14.01670', 'RGI60-14.00449', 'RGI60-14.03334', 'RGI60-15.06881', 'RGI60-15.06977', 'RGI60-15.06720', 'RGI60-17.15897', 'RGI60-17.15804', 'RGI60-17.14015']
file_id = ['390to420', 
           '150to180', 
           '150to180', '150to180', '150to180', '270to300', '270to300', '270to300', '270to300', '270to300', '420to450', '450to480', '450to480', '480to510', '480to510', '990to1020', '990to1020', '1200to1230', '1200to1230', '1320to1350', '1410to1440', '2220to2250', '2220to2250', '2220to2250', '2220to2250', '2340to2370', '2280to2310', '2370to2400', '2370to2400', '2370to2400', '2370to2400', '2370to2400', '2580to2610', '2580to2610', '2580to2610', '2760to2790', '2760to2790', '2760to2790']

#%%

#'RGI60-13.37536',
#'2220to2250',
glacier_list = [
'RGI60-13.37523',
'RGI60-13.30888',
'RGI60-13.26415',
'RGI60-13.54431',
'RGI60-13.43528',
'RGI60-14.00005',
'RGI60-14.02150',
'RGI60-14.01670',
'RGI60-14.00449',
'RGI60-14.03334',
'RGI60-15.06881',
'RGI60-15.06977',
'RGI60-15.06720']

file_id = [
'2220to2250',
'2220to2250',
'2220to2250',
'2340to2370',
'2280to2310',
'2370to2400',
'2370to2400',
'2370to2400',
'2370to2400',
'2370to2400',
'2580to2610',
'2580to2610',
'2580to2610']

evaluation_glaciers = [
    "RGI60-14.07524", #Siachen Glacier
    'RGI60-01.21014', #Carroll glacier
    "RGI60-01.20983", #Sea Otter glacier
    "RGI60-14.00005", #Biafo glacier
    ]
evaluation_glacier_file_ids = [
    "2460to2490",#Siachen Glacier
    '390to420', #Carroll glacier
    "360to390", #Sea Otter glacier
    "2370to2400" #Biafo glacier
    ]"""
#%%

"""
edge featrures:
catchment=catchment_num,
 width = mean_width,
 hor_len = horizontal_len,
 area = edge_area,
 area_precip = edge_total_precip, 
 area_temp = edge_total_temp, 
 P_diff = preassure_diff_mean
Node features:
    ['altitude','width', 'mean velocity', 'dS', 'dY']
 """
"""
if __name__ == "__main__":
    path_oggm_polygons = "files_glacier_shape/input_data"
    path_oggm_points = "files_glacier_shape/point_data"
    
    polygon_file = "glaciers_polygons_TH20_2220to2250.shp"
    point_file = "segments_points_TH20_2220to2250.shp"
    print("running on polygons:", polygon_file)
    path_to_polygons = path_oggm_polygons + "/" + polygon_file
    path_to_points = path_oggm_points + "/" + point_file
    
    #, 'RGI60-01.22193'
    #networks to train
    networks = glacier_network(['RGI60-13.37536'], path_to_points, path_to_polygons)
    
    #%%Add glaciers based on file_id:
    training_bool_options = np.array([True, False])
    working_glaciers = []
    glaciers_file_ids = []
    #%%
    for file_i, glacier_id in enumerate(glacier_list):
        training = np.random.choice(training_bool_options, 1) 
        print("file num and glacier: ", file_i, glacier_id)
        
        if file_i < 17: 
            networks.Add_GlacierFromFileId(glacier_id, file_id = file_id[file_i], training = True)
        
            if glacier_id not in working_glaciers:
                working_glaciers.append(glacier_id)
                glaciers_file_ids.append(file_id[file_i])
#%%    
    for file_i, glacier_id in enumerate(evaluation_glaciers):
        print("file num and glacier: ", file_i, glacier_id)
        
        networks.Add_GlacierFromFileId(glacier_id, file_id = evaluation_glacier_file_ids[file_i], training = False)
        
    #%%
    print(working_glaciers)
    print(glaciers_file_ids)
    #%%
    networks.Train_Model(model_name = 'width, mean velocity, dS, dY, area,informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                         training_glaciers =[],# 'RGI60-01.12077','RGI60-01.17183','RGI60-01.10689','RGI60-01.17566'],
                         node_feature_vairables=['width', 'mean velocity', 'dS', 'dY'],
                         inflow_features=['mean thickness'],
                         edge_features=['area'],#,'_precip', 'area_temp', 'P_diff'], 
                         epochs=400, n_layers = 6, n_nodes = 25,
                         exclude_filter_lowwer = 0, exclude_filter_upper = 0, drop_frac = 0.1, validation_split = 0.5, optimizer_num = 3,
                         include_edge_features=True)
    
    #%%
    networks.Test_ModelFeature('mean thickness',
                               models=['width, mean velocity, dS, dY, area,informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                                       'width, mean velocity, dS, dY, informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                                       'width, mean velocity, dS, dY, blind NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'],
                               model_custom_names=["area","informed", "blind"],
                               test_glaciers = ["RGI60-01.20983"], training_glaciers=[], feature_bin_size=20, bar_width = 8, tick_num = 10,
                               figsize=(14,8))
#%%
networks.Plot_Network("RGI60-14.00005", 'mean thickness', plot=True)
#%%
#Good models: 
    #'width, mean velocity, dS, dY, previous thickness NADAM, epochs 500 6lx20n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'
    #This one works a lot better just training two one glacier. Feeding it Two produces a bad model#'width, mean velocity, dS, dY, previous thickness NADAM, epochs 500 6lx20n, 6 features, v-split 0.7, d-frac 0.1, threshold 0m' and 'width, mean velocity, dS, dY, previous thickness NADAM, epochs 2000 6lx20n, 6 features, v-split 0.7, d-frac 0.1, threshold 0m'
    #^try adding more, or trainig another glacier after fitting it to one, or adding a thickness threshold
    #Don't train for too long! may become overtrained e.g. networks.Plot_GlacierThickness(RGI_id = 'RGI60-01.17566', catchment_number = '17', model_names=['width, mean velocity, dS, dY, previous thickness NADAM, epochs 500 6lx20n, 6 features, v-split 0.7, d-frac 0.1, threshold 0m',
    #                                                                                            'width, mean velocity, dS, dY, previous thickness NADAM, epochs 2000 6lx25n, 6 features, 4 glaciers, v-split 0.7, d-frac 0.1, threshold 0m'])
    
#Bad models:
    #'width, mean velocity, dS, dY, previous thickness, edge area NADAM, epochs 500 6lx20n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'
    #'mean velocity, dS, dY, previous thickness, edge area NADAM, epochs 500 6lx20n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'#
#%%
networks.nxGraphDict.pop('RGI60-01.17348')
networks.nxGraphDict.pop('RGI60-01.26738')

#networks.Plot_Network('RGI60-01.22193', 'altitude', plot=True)
#networks.Plot_GlacierThickness(RGI_id = 'RGI60-01.17566', catchment_number = '17', model_names=['width, mean velocity, dS, dY, previous thickness NADAM, epochs 500 6lx20n, 6 features, v-split 0.7, d-frac 0.1, threshold 0m',
#                                                                                                'width, mean velocity, dS, dY, previous thickness NADAM, epochs 2000 6lx25n, 6 features, 4 glaciers, v-split 0.7, d-frac 0.1, threshold 0m'])
#%%
networks.Plot_GlacierThickness(RGI_id = 'RGI60-14.02150', catchment_number = '6', 
                               model_names=['width, mean velocity, dS, dY, informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                                            'width, mean velocity, dS, dY, blind NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'],
                               model_custom_names=['informed', 'blind'])
#%%
networks.Plot_GlacierThickness(RGI_id = "RGI60-14.00005", catchment_number = '17', 
                               model_names=['width, mean velocity, dS, dY, informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                                            'width, mean velocity, dS, dY, blind NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'],
                               model_custom_names=['informed', 'blind'])
#%%
networks.Plot_GlacierThickness(RGI_id = 'RGI60-01.21014', catchment_number = '10',
                               model_names=['width, mean velocity, dS, dY, informed NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m',
                                            'width, mean velocity, dS, dY, blind NADAM, epochs 400 6lx25n, 6 features, v-split 0.5, d-frac 0.1, threshold 0m'])



"""

















