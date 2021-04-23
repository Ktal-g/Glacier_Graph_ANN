# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:34:38 2021

@author: Jack Blunt
"""
#%% Modules:

#for making mask:    
import fiona 
import shapely.vectorized

import gdal
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from math import degrees, copysign, ceil
from shapely.geometry import Point, LineString, shape
from shapely import wkt

import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

import networkx as nx

#%%
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/Jack/glacier_proj_home/Codeing_files/functions')

from FnExtract_raster_data import (GetRasterEdgeData, 
                                   point_and_vector_to_AB,
                                   plot_cross_vel,
                                   extract_cross_section_features, 
                                   looker,
                                   map_rgi_to_tiff_data,
                                   length_to_degrees)

from extract_glacier_dirs import (find_file_ends_with)

#%%
def Calculate_flowline_crossections(segment_input_GDF_path,
                                    path_to_catchment_polygons,
                                    tif_file_end = '_thickness.tif',
                                    output_point_data_dir = 'files_glacier_shape/point_feature_data',
                                    output_line_data_dir = 'files_glacier_shape/line_data',
                                    step = 1,
                                    clip_to_catchments= True,
                                    velocity = True,
                                    plot_velocities = False,
                                    plot_thicknesses = False,
                                    heatmap = False,
                                    testing = False,
                                    run_on_single_glacier = False,
                                    RGI_id_single = None,
                                    output_adition = "", 
                                    glacier_network = None,
                                    save_gpd = True,
                                    glacier_inputs = "path"):
    """
    
    Parameters
    ----------
    segment_input_GDF_path : TYPE
        shapefile containg point data of the flowlines
    path_to_catchment_polygons : TYPE
        Shapefile of glacier, devided into segment features
    tif_directory : TYPE
        str of the relative path to tiff data for thicknesses
    tif_file_end : TYPE, optional
        DESCRIPTION. The default is '_thickness.tif'.
    point_data_dir : TYPE, optional
        DESCRIPTION. The default is 'files_glacier_shape/point_data'.
    line_data_dir : TYPE, optional
        DESCRIPTION. The default is 'files_glacier_shape/line_data'.
    segment_points_out_file : TYPE, optional
        DESCRIPTION. The default is 'segment_points_out_file.shp'.
    lines_out_file_name : TYPE, optional
        DESCRIPTION. The default is 'segment_cross_sections.shp'.
    step : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    
    #geopandas method
    if glacier_inputs == "path":
        glacier_catchments = gpd.read_file(path_to_catchment_polygons, encoding="utf-8").set_index('index')
        segments_df = gpd.read_file(segment_input_GDF_path).set_index('index')
    elif glacier_inputs == "gpd":
        glacier_catchments = path_to_catchment_polygons
        segments_df = segment_input_GDF_path
    
    glacier_catchments = glacier_catchments.set_crs("EPSG:4326")
    
        
    
    
    segment_indices = list(glacier_catchments.index.values)
    RGI_set = set(glacier_catchments.RGI.values)
    print("RGI_set",RGI_set)
    print("RGI_id_single",RGI_id_single)
    if run_on_single_glacier == True:
        if RGI_id_single in RGI_set:
            print("RGI id in this file")
            RGI_set = [str(RGI_id_single)]
        else: print("RGI_id_single not in this gdf")
    
    
    Line_coordinates = []
    
    
    #dictionary of cros section max heights to be added to segments geoDF:
    max_cross_section_thickness = {}
    max_thickness_offset = {}
    mean_thickness = {}
    
    catchment_area_dict = {}
    catchment_lon_dif_dict = {}
    catchment_lat_dif_dict = {}
    
    
    output_df = pd.DataFrame(columns=["segment_index","geometry","RGIid","catchment","segment"])
    #transfer to GeoDataFrame, defineing the geometry as the "geometry" column.
    output_lines_gdf = gpd.GeoDataFrame(output_df, geometry= "geometry").set_index(['segment_index'])
    #define coords
    output_lines_gdf = output_lines_gdf.set_crs("EPSG:4326")
    
    #for when using velocities:
    rgi_region_previous = 0 
    velocity_region = True
    SEG_NUM = 0
    for RGIid in RGI_set:
        """
        Loops through glacier RGI indexes from the catchments to get the tiff file
        """
        
        print("collecting data for glacier: ", RGIid)
        rgi_region = str(RGIid[6:8])
        
        #if the rgi region has changed:
        if rgi_region != rgi_region_previous:    
            rgi_region_previous = rgi_region
            abs_vel_path, thickness_tiff_directory = map_rgi_to_tiff_data(rgi_region)
            
            if velocity:
                #updata velocity path
                if abs_vel_path == 0:
                    print("velocity data not avalable for region {}".format(rgi_region))
                    velocity_region = False                                    
                else:    
                    velocity_l = looker(abs_vel_path)
                                        #update rgi previous region vairable
                    
                    max_abs_vel_thickness = {}
                    max_vel_offset = {}
                    
                    
                    abs_vel_rasterDataSet = gdal.Open(abs_vel_path, 0)
                    
                    #(abs_vel_coord_top_left,
                    # abs_vel_left_norm, 
                    # abs_vel_top_norm, 
                    # abs_vel_length_left, 
                    # abs_vel_length_top, 
                    # abs_vel_cols, 
                    # abs_vel_rows,
                    # abs_vel_dleft_rows,
                    # abs_vel_dtop_column) = 
                    abs_vel_rbDims= GetRasterEdgeData(rasterDataSet=abs_vel_rasterDataSet)
        
        # Define where the tif file is. Be cairful what directory you run this from
        RGIid_fari_tif = thickness_tiff_directory + '/' + RGIid + tif_file_end
        
        
        # Read raster data
        # Open glacier raster file - do once for each glacier being passes
        rasterDataSet = gdal.Open(RGIid_fari_tif, 0)
        rb = rasterDataSet.ReadAsArray() #rb is the array form of rasterDataSet, containing all the evevations
        if heatmap == True:
            plt.imshow(rb, cmap='hot', interpolation='nearest')
            plt.show()
        
        #(coord_top_left,
        # left_norm, 
        # top_norm, 
        # length_left, 
        # length_top, 
        # cols, 
        # rows,
        # dleft_rows,
        # dtop_column) = 
        rbDims= GetRasterEdgeData(rasterDataSet=rasterDataSet)
        
        
        #then loop through glacier catchments
        for catchment_number, catchment_index in enumerate(glacier_catchments.index[glacier_catchments['RGI'] == RGIid].tolist()): #or do enumerate()
            #catchment contains the 14 characters of the RGIid and remaining 
            #characters of the catchment number it is:
            RGIindex = catchment_index[0:14]
            catchment_num_in_RGIid = catchment_index[14:]
            segments_per_flowline = int(glacier_catchments.loc[catchment_index]["segments"])
            
            
            catchment_geometry_gpd = glacier_catchments[glacier_catchments.index == catchment_index]
            #catchment_geometry_json = gpd.GeoSeries([catchment_geometry_gpd.loc[catchment_index]['geometry']]).__geo_interface__
            #catchment_geometry_json['type'] = 'polygon'
            
            #Calculate polygon area:
            #using method outlined at: https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas
            tost = catchment_geometry_gpd.copy()
            tost= tost.to_crs({'init': 'epsg:3857'})
            catchment_area = tost.loc[catchment_index]["geometry"].area/ 10**6
            if testing==True: print("catchment_area",catchment_area)
            
            (minLon, minLat, maxLon, maxLat) = catchment_geometry_gpd.loc[catchment_index]["geometry"].bounds
            catchment_lon_dif = maxLon - minLon
            catchment_lat_dif = maxLat - minLat
            
            segment_line_coordinates = []   
            segments_df_index_array = []
            RGIid_seg_array = []
            catchment_array = []
            
            segment_array = []
            
            #for calibrating sample number in raster data extraction
            segment_width = 0
            
            for segment_index in range(0, segments_per_flowline):
                SEG_NUM += 1
                segments_df_index = str(catchment_index + str(segment_index))
                
                
                try:
                    C = [segments_df.loc[segments_df_index]["Longitude"], segments_df.loc[segments_df_index]["Latitude"]]
                except KeyError:
                    print("segments_per_flowline:", segments_per_flowline)
                    
                    segments_df_index = segments_df_index[0:-1] #try taking off last digit
                    print("new index:",segments_df_index)
                    C = [segments_df.loc[segments_df_index]["Longitude"], segments_df.loc[segments_df_index]["Latitude"]]
                
                
                if testing == True: print("running segment number {} in catchment {}".format(segment_index, catchment_number))
                
                A, B = point_and_vector_to_AB(point_lon = C[0],
                                              point_lat = C[1],
                                              normal_lon = segments_df.loc[segments_df_index]["normal x"],
                                              normal_lat = segments_df.loc[segments_df_index]["normal y"],
                                              width = segments_df.loc[segments_df_index]["width, Y ("]) #################################### Play around with this number
                
                segment_width = segments_df.loc[segments_df_index]["width, Y ("]
                
                segments_df_index_array.append(segments_df_index)
                RGIid_seg_array.append(RGIindex)
                catchment_array.append(catchment_index)
                segment_array.append(segment_index)
                
                Line_coordinates.append(LineString([Point(A), Point(B)])) #might not be needed
                segment_line_coordinates.append(LineString([Point(A), Point(B)]))
            
            #for lines:
            seg_line_df = pd.DataFrame({"segment_index":segments_df_index_array,
                                        "geometry":segment_line_coordinates,
                                        "RGIid": RGIid_seg_array,
                                        "catchment":catchment_array,
                                        "segment":segment_array})
                
            seg_line_df["geometry"] = seg_line_df["geometry"].astype('str').apply(wkt.loads)
            #transfer to GeoDataFrame, defineing the geometry as the "geometry" column.
            seg_lines_output_gdf = gpd.GeoDataFrame(seg_line_df, geometry= "geometry").set_index(['segment_index'])
            #define coords
            seg_lines_output_gdf = seg_lines_output_gdf.set_crs("EPSG:4326")
            
            #clip to segment:
            # If there are duplicate values, try this: print(seg_lines_output_gdf[seg_lines_output_gdf.index.duplicated()])
            seg_lines_output_gdf_cliped = gpd.clip(seg_lines_output_gdf, catchment_geometry_gpd)
            
            #add cliped shape data to output geopandas geodataframe.
            output_lines_gdf = output_lines_gdf.append(seg_lines_output_gdf_cliped)
    
            
            for segment_index in range(0, segments_per_flowline):
                segments_df_index = str(catchment_index + str(segment_index))
                C = [segments_df.loc[segments_df_index]["Longitude"], segments_df.loc[segments_df_index]["Latitude"]]
                
                
                try:
                    AB_cliped_coords = output_lines_gdf.loc[segments_df_index].geometry
                except KeyError:
                    print("segments_df_index", segments_df_index)
                    
                
                try:
                    [A_cliped, B_cliped] = AB_cliped_coords.coords[:]
                except NotImplementedError:
                    [A_cliped, B_cliped] = AB_cliped_coords[0].coords[:]
                except AttributeError:
                    [A_cliped, B_cliped] = AB_cliped_coords[0].coords[:]
                
                
                
                (max_cross_section_thickness[segments_df_index], 
                max_thickness_offset[segments_df_index],
                mean_thickness[segments_df_index]) = extract_cross_section_features(rb,
                                                                                         rbDims,
                                                                                         A_cliped,
                                                                                         B_cliped,
                                                                                         C,
                                                                                         RGIindex, 
                                                                                         catchment_index, 
                                                                                         segment_index,
                                                                                         point_num=20,
                                                                                         smooth=False,
                                                                                         plot = plot_thicknesses)
                if glacier_network != None:
                    if type(glacier_network) == glacier_network:
                        Glacier_graph = glacier_network.Glacier_graph
                    elif type(glacier_network) == nx.classes.digraph.DiGraph:
                        Glacier_graph = glacier_network
                    else:
                        Glacier_graph = glacier_network.Glacier_graph
                    node_index = catchment_num_in_RGIid + str(segment_index)
                    Glacier_graph.nodes[node_index]["mean thickness"] = mean_thickness[segments_df_index]
                    Glacier_graph.nodes[node_index]["max thickness"] = max_thickness_offset[segments_df_index]
                    Glacier_graph.nodes[node_index]["catchment_area"] = catchment_area
                    Glacier_graph.nodes[node_index]["catch_dLat"] = catchment_lat_dif
                    Glacier_graph.nodes[node_index]["catch_dLon"] = catchment_lon_dif
                        
                if velocity and velocity_region:
                    #print("ceil(segment_width/120) = ", ceil(segment_width/120))
                    vel_output_type = "mean"
                    (max_abs_vel_thickness[segments_df_index], 
                    max_vel_offset[segments_df_index])  = plot_cross_vel(velocity_l, abs_vel_rbDims, A_cliped,
                                                                              B_cliped, C, RGIindex, catchment_index,
                                                                              segment_index, point_num=ceil(segment_width/120),
                                                                              smooth=True, plot = plot_velocities,
                                                                              Testing=testing, output = vel_output_type)
                    if glacier_network != None:
                        if type(glacier_network) == glacier_network:
                            Glacier_graph = glacier_network.Glacier_graph
                        elif type(glacier_network) == nx.classes.digraph.DiGraph:
                            Glacier_graph = glacier_network
                        else:
                            Glacier_graph = glacier_network.Glacier_graph
                        node_index = catchment_num_in_RGIid + str(segment_index)
                        Glacier_graph.nodes[node_index][vel_output_type + ' velocity'] = max_abs_vel_thickness[segments_df_index]
                    
                catchment_area_dict[segments_df_index] = catchment_area
                #print("len(catchment_area_dict) = ", len(catchment_area_dict))
                catchment_lon_dif_dict[segments_df_index] = catchment_lon_dif
                catchment_lat_dif_dict[segments_df_index] = catchment_lat_dif
    
    if save_gpd:
        RGI_list = list(RGI_set)
        #output shape
        if testing:
            lines_out_file_name = "segment_cross_sections" + RGI_list[0][6:] + "to" + RGI_list[-1][6:] + output_adition  + "test.shp"
            segment_points_out_file = "segment_points_" + RGI_list[0][6:] + "to" + RGI_list[-1][6:] + output_adition  + "test.shp"
        
        else:
            lines_out_file_name = "segment_cross_sections" + RGI_list[0][6:] + "to" + RGI_list[-1][6:] + output_adition + ".shp"
            segment_points_out_file = "segment_points_" + RGI_list[0][6:] + "to" + RGI_list[-1][6:] + output_adition + ".shp"
        
        
        #Uncoment if outputting line data:
        output_lines_gdf.to_file(output_line_data_dir + '/' + lines_out_file_name)
        
        print("\n\n\t\tOputputted line shape file to: ", output_line_data_dir + '/' + lines_out_file_name)
        
        print(catchment_area_dict)
        print("number of segments operated on: ", SEG_NUM)
        print("number of segments in df: ",len(segment_indices))
        
        #map feature dictionaries to point geodataframe:
        segments_df['Catch_area'] = segments_df.index.to_series().map(catchment_area_dict)
        segments_df['Catch_dlon'] = segments_df.index.to_series().map(catchment_lon_dif_dict)
        segments_df['Catch_dlat'] = segments_df.index.to_series().map(catchment_lat_dif_dict)
        #??segments_df['tributary'] = segments_df.index.to_series().map(tributaty_indexs_dict)"""
        
        #map target data
        #Thickness:
        segments_df['max_thk'] = segments_df.index.to_series().map(max_cross_section_thickness)
        segments_df['mean_thk'] = segments_df.index.to_series().map(mean_thickness)
        segments_df['thk_off'] = segments_df.index.to_series().map(max_thickness_offset)
        
        if velocity:
            #add max absolute velocity
            segments_df['mean_v'] = segments_df.index.to_series().map(max_abs_vel_thickness)
            segments_df['v_off'] = segments_df.index.to_series().map(max_vel_offset)
        
        #define coords
        segments_df = segments_df.set_crs("EPSG:4326")
        
        segments_df.to_file(output_point_data_dir + '/' + segment_points_out_file)
        print("\n\n\t\tOputputted point shape file to: ", output_point_data_dir + '/' + segment_points_out_file)


#%%

if __name__ == "__main__":
    
    path_oggm_polygons = "files_glacier_shape/input_data"
    path_oggm_points = "files_glacier_shape/point_data"
    
    
    glacier_point_shps_list = find_file_ends_with(path_oggm_points,'.shp')
    glacier_polygon_shps_list = find_file_ends_with(path_oggm_polygons,'.shp')
    
    glacier_file_num = 0
    print("------------------------------------------------------\n\n")
    print("running glacier file number {}: {}".format(glacier_file_num ,glacier_polygon_shps_list[glacier_file_num]))
    path_to_polygons = path_oggm_polygons + "/" + glacier_polygon_shps_list[glacier_file_num]
    path_to_points = path_oggm_points + "/" + glacier_point_shps_list[glacier_file_num]
    
    Calculate_flowline_crossections(path_to_catchment_polygons = path_to_polygons,
                                            segment_input_GDF_path = path_to_points,
                                            tif_file_end = '_thickness.tif',
                                            step = 1, 
                                            testing= False,
                                            plot_thicknesses = False,
                                            plot_velocities = False,
                                            output_adition = "",
                                            run_on_single_glacier=True,
                                            RGI_id_single=('RGI60-01.01731','RGI60-01.00031'))



















