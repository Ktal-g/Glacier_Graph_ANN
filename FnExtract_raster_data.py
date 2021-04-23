# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:41:45 2021

@author: Wolf
"""

from osgeo import osr
import numpy as np
from math import degrees, copysign, radians
import matplotlib.pyplot as plt
import gdal, osr


class looker(object):
    """let you look up pixel value,
    source: https://stackoverflow.com/questions/13439357/extract-point-from-raster-in-gdal,
    """

    def __init__(self, tifname='test.tif'):
        """Give name of tif file (or other raster data?)"""

        # open the raster and its spatial reference
        self.ds = gdal.Open(tifname)
        srRaster = osr.SpatialReference(self.ds.GetProjection())
        

        # get the WGS84 spatial reference
        srPoint = osr.SpatialReference()
        srPoint.ImportFromEPSG(4326) # WGS84

        # coordinate transformation
        self.ct = osr.CoordinateTransformation(srPoint, srRaster)

        # geotranformation and its inverse
        gt = self.ds.GetGeoTransform()
        dev = (gt[1]*gt[5] - gt[2]*gt[4])
        gtinv = ( gt[0] , gt[5]/dev, -gt[2]/dev, 
                gt[3], -gt[4]/dev, gt[1]/dev)
        self.gt = gt
        self.gtinv = gtinv

        # band as array
        b = self.ds.GetRasterBand(1)
        self.arr = b.ReadAsArray()

    def lookup(self, lon, lat):
        """look up value at lon, lat"""

        
        # get coordinate of the raster
        xgeo,ygeo,zgeo = self.ct.TransformPoint(lon, lat, 0)
        
        # convert it to pixel/line on band
        u = xgeo - self.gtinv[0]
        v = ygeo - self.gtinv[3]
        # FIXME this int() is probably bad idea, there should be 
        # half cell size thing needed
        xpix =  int(self.gtinv[1] * u + self.gtinv[2] * v)
        ylin = int(self.gtinv[4] * u + self.gtinv[5] * v)

        # look the value up
        return self.arr[ylin,xpix]


def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def GetRasterEdgeData(rasterDataSet):
    """Calculates lengths and normal vectors of the sides of the tiff.
    input: 
        raster aata set (osgeo.gdal.Dataset)
    output:
        normal vectors of the top and left sides of the tiff, and length of
        the fiff sides left and top, and number of columns and rows.
        (left normal, top normal, left length, top length, columns, rows)
    """
    cols = rasterDataSet.RasterXSize
    rows = rasterDataSet.RasterYSize
    
    # Get start/end point of the map in lon/lat
    coords = rasterDataSet.GetGeoTransform()
    ext=GetExtent(coords,cols,rows)
    
    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(rasterDataSet.GetProjection())
    #tgt_srs=osr.SpatialReference()
    #tgt_srs.ImportFromEPSG(4326)
    tgt_srs = src_srs.CloneGeogCS()
    geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)
    
    coord_top_left = np.array([geo_ext[0][1], geo_ext[0][0]])
    coord_bottom_left = np.array([geo_ext[1][1], geo_ext[1][0]])
    #coord_bottom_right = np.array([geo_ext[2][1], geo_ext[2][0]]) #not used
    coord_top_right = np.array([geo_ext[3][1], geo_ext[3][0]])
    
    #Left side:
        #calculate total length of vector going from top to bottom, this is how
        #the raster band array is indexd:
    left_length = np.linalg.norm(coord_bottom_left - coord_top_left)
        #left normal vector= botom_left - top_left divided by length:
    left_normal = np.array(coord_bottom_left - coord_top_left)/left_length
    
    #Top side:
        #Ranter band array is indexed from left to right, so top right - top left:
    top_length = np.linalg.norm(coord_top_right - coord_top_left)
        #Top normal vector:
    top_normal = np.array(coord_top_right - coord_top_left)/top_length
    
    return(coord_top_left, left_normal, top_normal, left_length, top_length, cols, rows, left_length/rows, top_length/cols)

def Reproject_to_Index(Refrence_coordinate, old_coordinate_A, x_normal_top, y_normal_left, dx_col, dy_row):
    """Calculates col, row index of coordinates 'old_coordinate_A' in a new refrence
    frame with given normal axes and column and row widths.
    inputs:
        Refrence_coordinate = the origin of array, where col, row = 0.
        old_coordinate_A = coordinate we want to find col, row index of.
        x_normal_top = the axes parallel to the top side of the array/vector with constant row index.
        y_normal_left = the axes parallel to the left side of the array/vector with constant column index.
        dx_col = width of each column/x-width of element.
        dy_row = width of each row/y-width of element.
    output:
        (A_x_index, A_y_index) = col, row index of old_coordinate_A
    """
    #Get the vector to the old_coordinate_A position:
    vector_A = np.array(old_coordinate_A - Refrence_coordinate)
    
    #Then find the length projection on the x_axes of the raster by
    #dotting the x_normal to vector_A. And do the same for y_axes.
    new_A_x = np.dot(vector_A, x_normal_top)
    new_A_y = np.dot(vector_A, y_normal_left)
    
    #Then the column number of A is the (projected length) / (length per index)
    A_x_index =new_A_x // dx_col
    A_y_index =new_A_y // dy_row
    
    return [A_x_index, A_y_index]

def length_to_degrees(length):
    # approximate radius of earth in km
    R = 6373.0
    #print("the width of the section is:", width, "m, type:", type(width))
    length_in_rad = length/(R*1000)
    length_deg = degrees(length_in_rad)
    return length_deg

def degrees_to_length(degrees_of_arc):
    # approximate radius of earth in km
    R = 6373.0
    
    #print("the width of the section is:", width, "m, type:", type(width))
    arc_in_rad = radians(degrees_of_arc)
    length = arc_in_rad * (R*1000)
    return length

'''
Calculate distance using the Haversine Formula
'''

def haversine(coord1: object, coord2: object):
    import math

    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters, 3)
    km = round(km, 3)


    #print(f"Distance: {meters} m")
    #print(f"Distance: {km} km")
    return meters


def point_and_vector_to_AB(point_lon, point_lat, normal_lon, normal_lat, width):
    """
    Takes point, normal to the flowline going through point and with of
    flowline. Outputs ends of cross section: A and B.
    """
    # approximate radius of earth in km
    R = 6373.0
    #print("the width of the section is:", width, "m, type:", type(width))
    width_in_rad = width/(R*1000)
    width_deg = degrees(width_in_rad)
    
    A_lon = point_lon - normal_lon*width_deg/2 #The normal lon is reversed, so inversed it to account for it
    A_lat = point_lat + normal_lat*width_deg/2
    #print("the location of point A long is: ",point_lon, " plus the normal_lon",normal_lon ,"times the width in deg", width_deg,"divided by", 2)
    #print("which equals:", A_lon)
    
    B_lon = point_lon + normal_lon*width_deg/2 #The normal lon is reversed, so inversed it to account for it
    B_lat = point_lat - normal_lat*width_deg/2
    
    return([A_lon, A_lat], [B_lon, B_lat])

def plot_cross_elv(raster_band_array=None, catchment_geom=None, A=None, B=None, C=None, RGIid = None, catchment_index = None, segment_index = None, col_max = None, row_max = None, point_num=1000, smooth = False, plot = False):
    """
    Parameters
    ----------
    raster : TYPE, optional
        DESCRIPTION. The default is None.
    A : TYPE, optional
        DESCRIPTION. The default is None.
    B : TYPE, optional
        DESCRIPTION. The default is None.
    point_num : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    Array of cross section

    """
    
    # Define Profile using lon/lat. Goes from A to B.
    
    if A[0] >= col_max:
        A[0] = int(col_max - 1)
        #print("adjusted A lon to", A[0])
    if A[0] < 0:
        A[0] = 0
        #print("adjusted A lon to", A[0])
    if B[0] >= col_max:
        B[0] = int(col_max - 1)
        #print("adjusted B lon to", B[0])
    if B[0] < 0:
        B[0] = 0
        #print("adjusted B lon to", B[0])
    
    if A[1] >= row_max:
        A[1] = int(row_max - 1)
        #print("adjusted A lon to", A[0])
    if A[1] < 0:
        A[1] = 0
        #print("adjusted A lon to", A[0])
    if B[1] >= row_max:
        B[1] = int(row_max - 1)
        #print("adjusted B row to", B[1])
    if B[1] < 0:
        B[1] = 0
        #print("adjusted B row to", B[1])
    
    ABx   = np.linspace(A[0], B[0], point_num)
    ABy   = np.linspace(A[1], B[1], point_num)
    
    profile = np.array([np.array([x,y]) for x, y in zip(ABx, ABy)])
    profile_half_width = len(profile)/2
    
    # Find Nearest points of profile in Map and Prepare for plot
    cross_dis = [] #store distance between cross section points
    cross_elv = []

    for profile_i, p in enumerate(profile):
        #Loops through all the cros section coordinates
        
        #print("int(p[0]), int(p[1]):", int(p[0]), int(p[1]))
        #raster_band_array has long and lat switched, so should be indexed [row, col]
        h = raster_band_array[int(p[1])][int(p[0])]
        #column/row distance = point - centre:
        d_column = int(p[0]) - C[0]
        d_row = int(p[1]) - C[1]
        
        #We want the distance to relative to the centre (half way through the
        #profile, at profile_half_width):
        d_sign = profile_i - profile_half_width
        d = copysign(np.linalg.norm([d_column, d_row]), d_sign)
        
        cross_dis.append(d)
        cross_elv.append(h)
    
    
    
    if smooth: cross_elv = smooth_array(cross_elv, 3)
    
    max_height= np.amax(cross_elv)
    max_height_index = np.argmax(cross_elv)
    
    max_height_offset = cross_elv[max_height_index]
    
    mean_thk = np.mean(cross_elv)
    
    if plot == True:
        ax = plt.subplot(111)
        ax.plot(cross_dis,cross_elv,linewidth=2,color='k')
        ax.plot(cross_dis,cross_elv,linewidth=2,color='k')
        ax.axvline(x=cross_dis[max_height_index])
        plt.title("Glacier: {}, catchment: {}, segment: {} cross section".format(str(RGIid), str(catchment_index), str(segment_index)))
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Thickness [m]')
        #ax.fill_between(cross_dis, 0, cross_elv, color='bisque')
        #ax.set_xlim(min(cross_dis), max(cross_dis))
        ax.set_xlim(-40, 40)
        ax.locator_params(axis='x',nbins=6)
        ax.locator_params(axis='y',nbins=6)
        ax.set_ylim(0,550)
        plt.show()
    
    return (max_height, max_height_offset, mean_thk)

def extract_cross_section_features(rb,
                                   rbDims,
                                   A_cliped,
                                   B_cliped,
                                   C,
                                   RGIindex, 
                                   catchment_index, 
                                   segment_index,
                                   point_num=20,
                                   smooth=False,
                                   plot=False):
    
    #Extract raster band dimensions:
    (coord_top_left,
     left_norm, 
     top_norm, 
     length_left, 
     length_top, 
     cols, 
     rows,
     dleft_rows,
     dtop_column) = rbDims
    
    A_index = Reproject_to_Index(Refrence_coordinate = coord_top_left,
                                 old_coordinate_A = A_cliped,
                                 x_normal_top = top_norm,
                                 y_normal_left = left_norm,
                                 dx_col = dtop_column,
                                 dy_row = dleft_rows)
                
    B_index = Reproject_to_Index(Refrence_coordinate = coord_top_left,
                                 old_coordinate_A = B_cliped,
                                 x_normal_top = top_norm,
                                 y_normal_left = left_norm,
                                 dx_col = dtop_column,
                                 dy_row = dleft_rows)
                #Centre coordinate
    C_index = Reproject_to_Index(Refrence_coordinate = coord_top_left,
                                 old_coordinate_A = C,
                                 x_normal_top = top_norm,
                                 y_normal_left = left_norm,
                                 dx_col = dtop_column,
                                 dy_row = dleft_rows)
                
    #dictionaries to add to point data:
    return plot_cross_elv(raster_band_array=rb, 
                          A=A_index, 
                          B=B_index,
                          C=C_index,
                          RGIid = RGIindex, 
                          catchment_index = catchment_index, 
                          segment_index = segment_index,
                          col_max = cols,
                          row_max = rows,
                          point_num=point_num,
                          smooth = smooth,
                          plot = plot)

def plot_cross_vel(looker_object, catchment_geom=None, A=None,
                   B=None, C=None, RGIid = None, catchment_index = None,
                   segment_index = None, col_max = None, row_max = None,
                   point_num=10, smooth = False, plot = False, output = "mean",
                   Testing=False):
    
    
    # Define Profile using lon/lat. Goes from A to B.
    #A = [lon, lat]
    #B = [lon, lat]
    ABx   = np.linspace(A[0], B[0], point_num)
    ABy   = np.linspace(A[1], B[1], point_num)
    
    profile = np.array([np.array([x,y]) for x, y in zip(ABx, ABy)])
    profile_half_width = len(profile)/2
    
    # Find Nearest points of profile in Map and Prepare for plot
    cross_dis = [] #store distance between cross section points
    cross_elv = []

    for profile_i, p in enumerate(profile):
        #We want the distance to relative to the centre (half way through the profile, at profile_half_width):
        #column/row distance = point - centre:
        d_column = int(p[0]) - C[0]
        d_row = int(p[1]) - C[1]
        
        
        #lon, lat = p
        p_lon = p[1]
        p_lat = p[0]
        
        if profile_i == 0 or profile_i == len(profile):
            h = 0
        else:    
            try: #QUICK FIX: I am adding this due to error: IndexError: index -10250 is out of bounds for axis 0 with size 8928. At line: >> h = looker_object.lookup(p_lon,p_lat)
                h = looker_object.lookup(p_lon,p_lat)
            except IndexError:
                h = 0
        
        d_sign = profile_i - profile_half_width
        d = copysign(np.linalg.norm([d_column, d_row]), d_sign)
        
        cross_dis.append(d)
        cross_elv.append(h)
    
    if smooth: cross_elv = smooth_array(cross_elv, 3)
    
    
    max_height_index = np.argmax(cross_elv)
    
    max_velocity_offset = cross_dis[max_height_index]
    
    if output == "mean":
        vel_output = np.mean(cross_elv)
    elif output == "max":
        vel_output = np.amax(cross_elv)
        
    if Testing == True: print("vel_output", vel_output)
    
    if plot == True and len(cross_dis) == len(cross_elv):
        ax = plt.subplot(111)
        ax.plot(cross_dis,cross_elv,linewidth=2,color='k')
        #ax.plot(cross_dis,cross_elv,linewidth=2,color='k')
        ax.axvline(x=cross_dis[max_height_index])
        plt.title("Glacier: {}, catchment: {}, segment: {} cross section velocity\nNumber of samples: {}".format(str(RGIid), str(catchment_index), str(segment_index), point_num ))
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('velocity [m/s]')
        #ax.fill_between(cross_dis, 0, cross_elv, color='bisque')
        #ax.set_xlim(min(cross_dis), max(cross_dis))
        ax.set_xlim(-5, 6)
        ax.locator_params(axis='x',nbins=6)
        ax.locator_params(axis='y',nbins=6)
        ax.set_ylim(0,250)
        plt.show()
    
    return (vel_output, max_velocity_offset)

def smooth_array(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
def pixel2coord(x, y, coords):
    """
    x pixel in x dir, y = pixel in y dir
    """
    xoff, a, b, yoff, d, e = coords
    print("xoff, a, b, yoff, d, e:", xoff, a, b, yoff, d, e)

    xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
    yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff
    return(xp, yp)

def map_rgi_to_tiff_data(rgi_region):
    """
    https://its-live.jpl.nasa.gov/#data
    Velocity data generated using auto-RIFT (Gardner et al., 2018) 
    and provided by the NASA MEaSUREs ITS_LIVE project (Gardner et al., 2019).
    """
    
    print("rgi_region: ",rgi_region)
    region_to_vel_path = {'01':['data_velocity/abs_v/ALA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-01'],
                          '02':['data_velocity/abs_v/ALA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-02'],
                          '03':['data_velocity/abs_v/CAN_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-03'],
                          '04':['data_velocity/abs_v/CAN_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-04'],
                          '06':['data_velocity/abs_v/ICE_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-06'],
                          '07':['data_velocity/abs_v/SRA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-07'],
                          '08':[0,'files_glacier_shape/thickness_tiffs/RGI60-08'],
                          '09':['data_velocity/abs_v/SRA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-09'],
                          '10':[0,'files_glacier_shape/thickness_tiffs/RGI60-10'],
                          '11':[0,'files_glacier_shape/thickness_tiffs/RGI60-11'],
                          '12':[0,'files_glacier_shape/thickness_tiffs/RGI60-12'],
                          '13':['data_velocity/abs_v/HMA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-13'],
                          '14':['data_velocity/abs_v/HMA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-14'],
                          '15':['data_velocity/abs_v/HMA_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-15'],
                          '16':[0,'files_glacier_shape/thickness_tiffs/RGI60-16'],
                          '17':['data_velocity/abs_v/PAT_G0120_0000_v.tif','files_glacier_shape/thickness_tiffs/RGI60-17'],
                          '18':[0,'files_glacier_shape/thickness_tiffs/RGI60-18']}
    
    print("testing region:", str(rgi_region))
    print("changed velocity path to: {}".format(region_to_vel_path[str(rgi_region)][0]))
    return region_to_vel_path[str(rgi_region)][0], region_to_vel_path[str(rgi_region)][1]
    