#!/usr/bin/env python3

import csv
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from more_itertools import locate
import glob
import os
import random
import math

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane 

    def segment(self, distance_threshold=0.04, ransac_n=3, num_iterations=200):  # Only parameters to change

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model
        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)
        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")

    dataset_path = 'assignment_2/SAVI---Where-s-my-coffee-mug/datasets/scene_pc/' # relative path

    point_cloud_filenames = glob.glob(dataset_path+'/*.ply')
    pcl_random = random.choice(point_cloud_filenames) # TEST

    pcl_random = dataset_path+'/01.ply' # 09-12 problem for the sofa and 5-8 of the z axis not pointing towards the table

    os.system('pcl_ply2pcd ' +pcl_random+ ' pcd_point_cloud.pcd')
    point_cloud_original = o3d.io.read_point_cloud('pcd_point_cloud.pcd')

    # downsampling : here create problem to the plane detection
    # point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.01)
    # print('after downsampling point cloud has '+str(len(point_cloud_downsampled.points))+'points')

    number_of_planes = 2
    minimum_number_points = 25
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    point_cloud = deepcopy(point_cloud_original) 
    planes = []
    while True: # run consecutive plane detections

        plane = PlaneDetection(point_cloud) #ex2/factory_without_ground.ply create a new plane instance
        point_cloud = plane.segment() # new point cloud are the outliers of this plane detection
        print(plane)

        # colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        plane.colorizeInliers(r=color[0], g=color[1], b=color[2])

        planes.append(plane)

        if len(planes) >= number_of_planes: # stop detection planes
            print('Detected planes >=' + str(number_of_planes))
            break
        elif len(point_cloud.points) < minimum_number_points:
            print('number of remaining points <' + str(minimum_number_points))
            break

    # select table plane
    table_plane = None
    table_plane_mean_xy = 1000
    for plane_idx, plane in enumerate(planes):
        center = plane.inlier_cloud.get_center()
        print('Cloud ' + str(plane_idx) + ' has center '+str(center))
        mean_x = center[0]
        mean_y = center[1]
        mean_z = center[2]

        mean_xy = abs(mean_x) + abs(mean_y)
        if mean_xy < table_plane_mean_xy:
            table_plane = plane
            table_plane_mean_xy = mean_xy

    # paint in red the table plane
    table_plane.colorizeInliers(r=1,g=0,b=0)

    # downsampling
    table_plane_downsampled = table_plane.inlier_cloud.voxel_down_sample(voxel_size=0.01)
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.01)
    print('after downsampling point cloud has '+str(len(table_plane_downsampled.points))+'points')

    # Clustering

    cluster_idxs = list(table_plane_downsampled.cluster_dbscan(eps=0.10, min_points=50, print_progress=True))

    # print(cluster_idxs)
    # print(type(cluster_idxs))

    possible_values = list(set(cluster_idxs))
    if -1 in cluster_idxs:
        possible_values.remove(-1)
    print(possible_values)

    # search for the biggest cluster
    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    # search for the nearest cluster to the z axis
    nearest_z_cluster_idx = None
    table_cloud_mean_xy = 1000
    for value in possible_values:
        idxs = list(locate(cluster_idxs, lambda x: x == value))
        center = table_plane_downsampled.select_by_index(idxs).get_center()
        # select the cluster closest to the z axis --> min x and min y
        mean_x = center[0]
        mean_y = center[1]
        mean_z = center[2]

        mean_xy = abs(mean_x) + abs(mean_y)
        if mean_xy < table_cloud_mean_xy:
            nearest_z_cluster_idx = value
            #table_cloud = table_plane_downsampled.select_by_index(idxs)
            table_cloud_mean_xy = mean_xy


    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))
    nearest_z_idxs= list(locate(cluster_idxs, lambda x: x == nearest_z_cluster_idx))

    # now select the table: the table is the nearest to the z axis only if it has a certain amount of points, otherwise is the biggest clust
    num_points = len(table_plane_downsampled.select_by_index(nearest_z_idxs).points)
    if num_points >= 5000:
        cloud_table = table_plane_downsampled.select_by_index(nearest_z_idxs)
    else:
        cloud_table = table_plane_downsampled.select_by_index(largest_idxs)

    # print num of points
    print('number of points of the nearest to z cluster: '+str(num_points))
    num_points = len(table_plane_downsampled.select_by_index(largest_idxs).points)
    print('number of points of the biggest cluster: '+str(num_points))

    cloud_table.paint_uniform_color([0,1,0]) #paint in green the biggest clust.
    #table_cloud.paint_uniform_color([0,0,1]) #paint in blue the nearest to z clust.

    # ------------------------------------------
    # crop the table
    # ------------------------------------------

    center = cloud_table.get_center()

    # traslate point cloud
    point_cloud.translate(-center) # the one with the table plane segmented

    # Calculate rotation angle between plane normal & z-axis
    plane_normal = tuple([table_plane.a,table_plane.b,table_plane.c])
    z_axis = (0,0,1)
    rotation_angle = np.arccos(np.dot(plane_normal, z_axis) / (np.linalg.norm(plane_normal)* np.linalg.norm(z_axis)))

    # Calculate rotation axis
    plane_normal_length = math.sqrt(table_plane.a**2 + table_plane.b**2 + table_plane.c**2)
    u1 = table_plane.b / plane_normal_length
    u2 = -table_plane.a / plane_normal_length
    rotation_axis = (u1, u2, 0)

    # Generate axis-angle representation
    optimization_factor = 1
    axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

    # Rotate point cloud
    R = point_cloud.get_rotation_matrix_from_axis_angle(axis_angle)
    point_cloud.rotate(R, center=(0,0,0)) # the one with the table plane segmented

    # Create a list of entities to draw

    entities = [point_cloud_downsampled]
    entities.append(table_plane_downsampled)
    entities.append(cloud_table)

    for x in entities:
        x = x.translate(-center)
        x = x.rotate(R,center=(0,0,0))

    # take a bbox
    obb = cloud_table.get_oriented_bounding_box()
    box_points=obb.get_box_points()
    box_points = np.asarray(box_points)
    box_points[3:7,2]=box_points[3:7,2] - 0.3 # extend the bbox to take also the object on the table
    box_points[:,2]=box_points[:,2] - 0.05 # to avoid to take some piece of table
    # From numpy to Open3D
    box_points = o3d.utility.Vector3dVector(box_points) 
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)
    bbox.color = (0, 1, 0)

    table = point_cloud_downsampled.crop(bbox)

    entities.append(bbox)
    entities_2 = [table]

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # draw the xyz axis
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin = np.array([0.,0.,0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities)
    o3d.visualization.draw_geometries(entities_2)

if __name__ == "__main__":
    main()