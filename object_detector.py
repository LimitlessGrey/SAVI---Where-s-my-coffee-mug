#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import cv2
import numpy as np
from matplotlib import cm
from more_itertools import locate
import glob
import os
import random
import math
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from Classifier import Classifier
from PIL import Image
from TTS import TTS

# define the view for visualize different results
view={
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 3.0000000000000004, 3.0000000000000004, 3.0000000000000004 ],
			"boundingbox_min" : [ -2.5373308564675199, -2.1335212159612991, -1.327641381237858 ],
			"field_of_view" : 60.0,
			"front" : [ -0.067906654377651754, -0.70142886552419004, -0.70949716905755311 ],
			"lookat" : [ 0.23133457176624028, 0.43323939201935069, 0.83617930938107121 ],
			"up" : [ -0.084219703952140443, 0.71263056388397028, -0.69646587919626657 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
view_2 = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.57766820479332304, 0.53542418579498685, 0.011344715613939613 ],
			"boundingbox_min" : [ -0.64853301295736199, -0.58922634959095666, -0.33045609592173575 ],
			"field_of_view" : 60.0,
			"front" : [ 0.68255579678994682, -0.41451757792185218, -0.60190760242934471 ],
			"lookat" : [ 0.0933088775718625, -0.12451645369686579, -0.23988061191282783 ],
			"up" : [ -0.50625636487562431, 0.3258133621214645, -0.79846737321322425 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane 

    def segment(self, distance_threshold=0.04, ransac_n=3, num_iterations=100):

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

    # personal path to the Washington_RGB-D_Dataset containing:  rgbd-scenes-v2/pc/..    and:   rgbd-scenes-v2/imgs/..
    personal_path = '/home/rafael/Desktop/Washington_RGB-D_Dataset'
    # personal_path = '/home/igino/Desktop/SAVI_dataset/Washington_RGB-D_Dataset'

    dataset_path = personal_path + '/rgbd-scenes-v2/pc' 

    point_cloud_filenames = glob.glob(dataset_path+'/*.ply')
    point_cloud_filename = random.choice(point_cloud_filenames)

    # uncomment this to manually select a scenario
    point_cloud_filename = dataset_path+'/03.ply' 

    # find the correspondent rgb images
    parts = point_cloud_filename.split('/')
    part = parts[-1]
    part = part.split('.')
    part = part[0]

    # take only the first image for the moment
    image_path = personal_path + '/rgbd-scenes-v2/imgs/scene_'+str(part)+'/00000-color.png'
    image = cv2.imread(image_path)

    os.system('pcl_ply2pcd ' +point_cloud_filename+ ' pcd_point_cloud.pcd')
    point_cloud_original = o3d.io.read_point_cloud('pcd_point_cloud.pcd')
    point_cloud = deepcopy(point_cloud_original)
    point_cloud_2 = deepcopy(point_cloud_original)# second copy for visualization  

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    number_of_planes = 2
    minimum_number_points = 25
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    planes = []
    while True: # run consecutive plane detections

        plane = PlaneDetection(point_cloud) 
        point_cloud = plane.segment(distance_threshold=0.035, ransac_n=3, num_iterations=200) # new point cloud are the outliers of this plane detection
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
    table_plane_mean_xy = 1000 # large number
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
    print('after downsampling table point cloud has '+str(len(table_plane_downsampled.points))+'points')

    # Clustering

    cluster_idxs = list(table_plane_downsampled.cluster_dbscan(eps=0.08, min_points=50, print_progress=True))

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

    # search for the nearest cluster to the z axis (mean xy closest to zero)
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
            table_cloud_mean_xy = mean_xy


    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))
    nearest_z_idxs= list(locate(cluster_idxs, lambda x: x == nearest_z_cluster_idx))

    # Choose the table: the table is the nearest to the z axis only if it has a certain amount of points, otherwise is the biggest clust
    num_points = len(table_plane_downsampled.select_by_index(nearest_z_idxs).points)
    if num_points >= 5000:
        cloud_table = table_plane_downsampled.select_by_index(nearest_z_idxs)
    else:
        cloud_table = table_plane_downsampled.select_by_index(largest_idxs)

    cloud_table.paint_uniform_color([0,1,0]) # paint in green the table

    # ------------------------------------------
    # crop the table
    # ------------------------------------------

    center = cloud_table.get_center()

    # traslate point cloud
    point_cloud.translate(-center) 

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
    optimization_factor = 1.1
    axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

    # Rotate point cloud
    R = point_cloud.get_rotation_matrix_from_axis_angle(axis_angle)
    point_cloud.rotate(R, center=(0,0,0)) 

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
  
    entities.append(obb)

    # Extend in the z direction the box , since the axis are not perfectly alligned we need to make some calculation
    # search for the couple of vertices (the ones with closest x and y)
    couples = []
    for i in range(np.shape(box_points)[0]):
        xy = 0
        min_difference = 100 #big number
        for j in range(np.shape(box_points)[0]):
            if i != j:
                x_difference = abs(box_points[i,0]-box_points[j,0])
                y_difference = abs(box_points[i,1]-box_points[j,1])
                difference = x_difference+y_difference
                if difference < min_difference:
                    min_difference = difference
                    couple = [i,j]
        # save the couple
        couple = sorted(couple)
        if not couple in couples:
            couples.append(couple)
        couple = []

    z = sorted(box_points[:,2])

    # z axis is pointing from the table to the floor
    high_z = z[4] # is the highest of the four bottom vertices
    low_z = z[0] # is the smallest so the highest of the bbox

    # Uniform the z values of the bbox
    for i in range(len(couples)):
        if box_points[couples[i][0],2] < box_points[couples[i][1],2]:
            box_points[couples[i][0],2] = low_z
            box_points[couples[i][1],2] = high_z
        else:
            box_points[couples[i][1],2] = low_z
            box_points[couples[i][0],2] = high_z

        # for each couple extend one vertices in z direction
        if box_points[couples[i][0],2] < box_points[couples[i][1],2]:
            box_points[couples[i][0],2] = box_points[couples[i][0],2] - 0.4
        else:
            box_points[couples[i][1],2] = box_points[couples[i][1],2] - 0.4

    # From numpy to Open3D
    box_points = o3d.utility.Vector3dVector(box_points) 
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)
    bbox.color = (0, 1, 0)
    entities.append(bbox)

    # ------------------------------------------
    # from the cropped table isolate the object
    # ------------------------------------------

    # take another copy of the original point cloud so it isn't downsampled
    point_cloud_2.translate(-center) 
    point_cloud_2.rotate(R, center=(0,0,0))

    table = point_cloud_2.crop(bbox)

    # another plane detection to identify better the table plane
    plane = PlaneDetection(table)
    table= plane.segment(distance_threshold=0.02) #return outlier cloud
    print(plane)
    color = [0,0,1]
    plane.colorizeInliers(r=color[0], g=color[1], b=color[2])

    # define entities to draw (second representation)
    entities_2 = [table]

    # downsapling for clustering faster
    table_plane_down= plane.inlier_cloud.voxel_down_sample(voxel_size=0.01)
    entities_2.append(table_plane_down)

    # clustering of the object
    table_downsampled = table.voxel_down_sample(voxel_size=0.005)
    cluster_idxs = list(table_downsampled.cluster_dbscan(eps=0.025, min_points=100, print_progress=True))

    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = table_downsampled.select_by_index(object_point_idxs)
        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points

        # compute bbox of the object
        d['center'] = d['points'].get_center()
        d['bbox'] = d['points'].get_axis_aligned_bounding_box()
        d['bbox'].color=(0,1,0)

        # compute mean color of the object
        mean_color = [0,0,0]
        for color in np.asarray(object_points.colors):
            mean_color=mean_color+color
        mean_color = mean_color/np.asarray(object_points.colors).shape[0]
        d['mean_color']=mean_color

        # compute other properties of objects:
        max_bound = d['points'].get_max_bound()
        min_bound = d['points'].get_min_bound()
        d['length'] = abs(abs(max_bound[0])-abs(min_bound[0])) 
        d['width'] = abs(abs(max_bound[1])-abs(min_bound[1])) 
        d['height'] = abs(abs(max_bound[2])-abs(min_bound[2])) 
        d['volume'] = d['length']*d['width']*d['height']
        d['distance'] = math.sqrt(d['center'][0]**2 + d['center'][1]**2)

        # select the object that are over the table, near the center of the table, and have a reasonable size
        if d['center'][2] <= -0.03 and d['volume'] <= 0.03 and d['distance'] <= 0.6 : 
            objects.append(d) # add the dict of this object to the list

    # Draw objects
    for object_idx, object in enumerate(objects):
        entities_2.append(object['bbox'])

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # draw the xyz axis
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin = np.array([0.,0.,0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities,   
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'],
                                    point_show_normal=False)
    o3d.visualization.draw_geometries(entities_2,
                                    zoom=view_2['trajectory'][0]['zoom'],
                                    front=view_2['trajectory'][0]['front'],
                                    lookat=view_2['trajectory'][0]['lookat'],
                                    up=view_2['trajectory'][0]['up'],
                                    point_show_normal=False)

    # Make a more complex open3D window to show object labels on top of 3d
    app = gui.Application.instance
    app.initialize() # create an open3d app

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  # set black background
    material = rendering.Material()#Record()
    material.shader = "defaultUnlit"
    material.point_size = 2 * w.scaling

    # define entities to draw in this representation
    entities_3 = []
    for object_idx, object in enumerate(objects):
        entities_3.append(object['points'])
        entities_3.append(object['bbox'])
    
    for entity_idx, entity in enumerate(entities_3):
        widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)

    # Draw labels over the objects
    for object_idx, object in enumerate(objects):

        label_pos = [object['center'][0], object['center'][1], object['center'][2] - object['height'] -0.1]
        label_text = 'object idx: '+object['idx']+'\nheight: '+str(object['height']*100)+'\nwidth: '+str(object['width']*100)+'\nlength: '+str(object['length']*100)+'\nvolume: '+str(object['volume']*100*100*100)+'\ndistance from center table:'+str(object['distance']*100)+'\nmean color:'+str(object['mean_color'])
        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(1,1,1)
        
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()

    #--------------------------------------
    # image processing
    #--------------------------------------

    # define camera parameters
    intrinsic = np.array([[570.3, 0, 320],[ 0 , 570.3, 240],[0,0,1]])
    rvec = np.eye(3)
    tvec = np.float32([0,0,0]).reshape(1,3)
    distCoeff = np.empty((1,4),dtype=float)

    # apply the inverse transformation to return in the camera point of view
    for object_idx, object in enumerate(objects):
        object['center_wrt_cam'] = object['points'].rotate(np.transpose(R),center=(0,0,0))
        object['center_wrt_cam']= object['center_wrt_cam'].translate(center)
        object['center_wrt_cam']= object['center_wrt_cam'].get_center()

        [img_points,_] = cv2.projectPoints(object['center_wrt_cam'], rvec, tvec, intrinsic,distCoeffs=distCoeff)
        object['img_point']= img_points[0][0]
        object['x_pix'] = int(object['img_point'][0])
        object['y_pix'] = int(object['img_point'][1])

    # validate the object view: in a image you may find overlapped objects
    for object_idx, object in enumerate(objects):
        for object_idx_2, object_2 in enumerate(objects):
            if object['idx'] != object_2['idx'] and object['idx'] < object_2['idx']:

                # if abs( abs(object['x_pix']) - abs(object_2['x_pix']) ) > 80 :
                if math.sqrt( (abs(object['x_pix']) - abs(object_2['x_pix']))**2 + (abs(object['y_pix']) - abs(object_2['y_pix']))**2 ) > 80 :
                    object['valid'] = True
                    object_2['valid'] = True
                else: # take the object closest to the camera meaning the one with highest y
                    if object['y_pix'] > object_2['y_pix']:
                        object['valid'] = True
                        object_2['valid'] = False
                    else:
                        object['valid'] = False
                        object_2['valid'] = True

    # take object that are into the camera view
    # x_bound = image.shape[0]
    # y_bound = image.shape[1]
    # for object_idx, object in enumerate(objects):
    #     if object['x_pix'] > x_bound or object['y_pix'] > y_bound:
    #         object['valid'] = False

    # image cropped size
    width = 80
    height = 70

    # crop the valid object from the image
    for object_idx, object in enumerate(objects):
        if object['valid'] == True:
            
            top_l_x = object['x_pix'] - int(width/2)
            top_l_y =  object['y_pix'] - int(height/2)
            bot_r_x = object['x_pix'] + int(width/2)
            bott_r_y =  object['y_pix'] + int(height/2)
            object['crop'] = image[ top_l_y:bott_r_y , top_l_x:bot_r_x ]

            # print and wait 
            cv2.imshow('win',object['crop'])
            cv2.waitKey(0)

            # pass the image to the classifier
            im = Image.fromarray(object['crop'])
            object['class_name'] = Classifier(im)
            print(object['class_name'])
            text_to_speach = 'the object detected is a'+ object['class_name']
            TTS(text_to_speach)

            # draw on the image
            img_2 = cv2.circle(image, (object['x_pix'],object['y_pix']), 3 ,[0,255,0], -1)
            img_2 = cv2.rectangle(image, (top_l_x,top_l_y) ,(bot_r_x,bott_r_y), [0,255,0], 2)

    # show the final image with the valid objects
    cv2.imshow('win',image)
    cv2.waitKey(0)

    #-------------------------------------------------- 
    # elaborate all the other images
    #---------------------------------------------


    # 0.813492 -0.0473971 -0.515618 -0.264773 0.959567 -0.513907 0.9545 # for image 00887

    # entities_4 = []
    # entities_4.append(point_cloud_original)
    # rotation = point_cloud_original.get_rotation_matrix_from_quaternion(np.array([0.813492,-0.0473971, -0.515618, -0.264773]))
    # traslation = np.array([0.959567, -0.513907, 0.9545])

    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=2, origin = traslation)
    # entities_4.append(frame)

    # center_points=np.array([0,0,0])
    # for object_idx, object in enumerate(objects):
    #     rotation = object['points'].get_rotation_matrix_from_quaternion(np.array([0.813492,-0.0473971, -0.515618, -0.264773]))
    #     traslation = np.array([0.959567, -0.513907, 0.9545])

    #     object['center_wrt_cam']= object['points'].rotate(rotation,center=[0,0,0])
    #     object['center_wrt_cam']= object['center_wrt_cam'].translate(traslation)
    #     object['center_wrt_cam']= object['center_wrt_cam'].get_center()
    #     center_points=np.vstack([center_points,object['center_wrt_cam']])
    #     frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin = np.array(object['center_wrt_cam']))
    #     entities_4.append(frame)

    # o3d.visualization.draw_geometries(entities_4,   
    #                         zoom=view['trajectory'][0]['zoom'],
    #                         front=view['trajectory'][0]['front'],
    #                         lookat=view['trajectory'][0]['lookat'],
    #                         up=view['trajectory'][0]['up'],
    #                         point_show_normal=False)

    # center_points = np.delete(center_points,0,0)


    # [img_points,_] = cv2.projectPoints(center_points, rvec, tvec, intrinsic,distCoeffs=distCoeff)

    # for i in range(np.size(img_points,0)) :
    #     x_pix = int(img_points[i][0][0])
    #     y_pix = int(img_points[i][0][1])
    #     img_2 = cv2.circle(image, (x_pix,y_pix), 3 ,[0,255,0], -1)
    #     top_l_x = x_pix - int(width/2)
    #     top_l_y =  y_pix - int(height/2)
    #     bot_r_x = x_pix + int(width/2)
    #     bott_r_y =  y_pix + int(height/2)
    #     img_2 = cv2.rectangle(image, (top_l_x,top_l_y) ,(bot_r_x,bott_r_y), [0,255,0], 2)


if __name__ == "__main__":
    main()
