#!/usr/bin/env python
# TODO: speed up spatial
import sys
import os
import rospkg
rospack = rospkg.RosPack()
pack_path = rospack.get_path('jsk_2016_01_baxter_apc')
sys.path.insert(0, pack_path)
from node_scripts.helper import *
import node_scripts.helper as helper
# segmenter
from node_scripts.segmentation.apc_data import APCSample, APCSamplePredicted
from node_scripts.segmentation.probabilistic_segmentation import ProbabilisticSegmentationBP, ProbabilisticSegmentation
# ros
import rospy
import tf2_ros
import message_filters
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from jsk_recognition_msgs.msg import BoundingBox
from std_msgs.msg import Header 
from geometry_msgs.msg import Pose
from image_geometry import cameramodels 
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2
from tf2_geometry_msgs import do_transform_point
# other
import numpy as np
from matplotlib.path import Path
import pickle
import cv2
from cv_bridge import CvBridge, CvBridgeError 
from time import strftime


class Bin(object):
    def __init__(self, initial_pos, initial_rot, dimensions, frame_id, 
            camera_direction, bin_name, depth):
        assert initial_rot == [0, 0, 0, 0]
        self.bin_name = bin_name
        self.direction = camera_direction

        _header = Header(
                stamp=rospy.Time.now(), 
                frame_id=frame_id)
        self.bbox = BoundingBox(
                header=_header,
                pose=Pose(
                        position=helper.point(initial_pos), 
                        orientation=helper.quaternion(initial_rot)),
                dimensions=helper.vector3(dimensions))

        assert type(dimensions[1]) == float
        for j in xrange(4):
            if camera_direction == 'x':
                # an order in which the points are appended is important
                # mask image depends on the order
                # x axis is directing away from Baxter
                self.corners = [
                        helper.corner_point(
                                initial_pos, dimensions, depth, _header,
                                signs=[-1, 1, 1]),
                        helper.corner_point(
                                initial_pos, dimensions, depth, _header, 
                                signs=[-1, -1, 1]),
                        helper.corner_point(
                                initial_pos, dimensions, depth, _header, 
                                signs=[-1, -1, -1]),
                        helper.corner_point(
                                initial_pos, dimensions, depth, _header, 
                                signs=[-1, 1, -1])
                    ]
            elif camera_direction == 'y':
                raise NotImplementedError
            elif camera_direction == 'z':
                raise NotImplementedError
            else:
                raise NotImplementedError


class Pipeline(object):
    """
    param: target: name of the current targeting bin 
    type:  target: string
    """
    def __init__(self, *args, **kwargs):
        self.mask_image = None
        self.dist_img = None
        self.bin_dict = {}
        self.camera_model = cameramodels.PinholeCameraModel()
        if args or kwargs:
            self.target_object = kwargs['target_object']
            initial_pos_list = kwargs['initial_pos_list']
            initial_rot_list = kwargs['initial_rot_list']
            dimensions_list = kwargs['dimensions_list']
            frame_id_list = kwargs['frame_id_list']
            camera_directions = kwargs['camera_directions']
            trained_pkl_path = kwargs['trained_pkl_path']

            self.objects = kwargs['objects']

            for i, bin_name in enumerate(kwargs['prefixes']):
                ## currently the code does not support pose rotation
                # calculate corners of bounding boxes
                self.bin_dict[bin_name] = Bin(initial_pos_list[i],
                        initial_rot_list[i], dimensions_list[i], 
                        frame_id_list[i], camera_directions[i],
                        bin_name=bin_name, depth=1.0)

            if 'target' in kwargs:
                self.target = kwargs['target']
            #camera
            if 'camera_info' in kwargs:
                self.camera_info = kwargs['camera_info']
            #img_color
            if 'img_color' in kwargs:
                self.img_color = kwargs['img_color']

            # trained
            with open(trained_pkl_path, 'rb') as f:
                self.trained_segmenter = pickle.load(f)

    @property
    def camera_info(self):
        return self._camera_info

    @camera_info.setter
    def camera_info(self, camera_info):
        self._camera_info = camera_info 
        if camera_info is not None:
            self.camera_model.fromCameraInfo(camera_info)

    @property
    def img_color(self):
        return self._img_color

    @img_color.setter
    def img_color(self, img_color):
        # following RBO's convention that img is loaded as HSV
        if img_color is not None:
            self._img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        else:
            self._img_color = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target 
        self.target_bin = self.bin_dict[self._target]
        
    def get_mask_img(self, transform):
        """
        :param point: point that is going to be transformed 
        :type point: PointStamped
        :param transform: camera_frame -> bbox_frame
        :type transform: Transform
        """
        ## check frame_id of a point and transform just in case     
        assert self.camera_info.header.frame_id == transform.header.frame_id
        assert self.target_bin.bbox.header.frame_id == transform.child_frame_id
        
        transformed_list = [do_transform_point(corner, transform) 
                for corner in self.target_bin.corners]
        projected_points =  self._project_points(transformed_list)
        
        ## generate an polygon that covers the region
        path = Path(projected_points)
        x, y = np.meshgrid(
                np.arange(self.camera_info.width), 
                np.arange(self.camera_info.height))
        x, y = x.flatten(), y.flatten()    
        points = np.vstack((x,y)).T  
        self.mask_image = path.contains_points(
                points).reshape(
                        self.camera_info.height, self.camera_info.width
                    ).astype('bool')
        return self.mask_image
        
    def _project_points(self, points):
        """
        :param points: list of geometry_msgs.msg.PointStamped
        :type list of stamped points : 
        :param projected_points: list of camera_coordinates
        :type  projected_points: (u, v)
        
        The frames of the points and the camera_model are same. 
        """
        # generate mask iamge 
        for point in points:
            if point.header.frame_id != self.camera_model.tf_frame:
                raise valueError
        if len(points) != 4:
            raise valueError
        
        projected_points = []
        for point in points:
            projected_points.append(
                    self.camera_model.project3dToPixel(
                            helper.list_from_point(point.point)
                        )
                )
        return projected_points 

    def get_spatial_img(self, bb_base2camera, cloud):
        """
        :param bb_base2camera: transform from the boundingbox's frame
                               to camera's frame
        :type  bb_base2camera: transformStamped 
        :param dist_img      : distance of a point from shelf (mm)
        :type  dist_img      : np.array, dtype=uint8
        """
        bb_base2camera_mat = helper.tfmat_from_tf(bb_base2camera)
        bbox2bb_base_mat = helper.inv_tfmat(
            helper.tfmat_from_bbox(self.target_bin.bbox))
        bbox2camera_mat = np.dot(bbox2bb_base_mat, bb_base2camera_mat)
        bbox2camera = helper.tf_from_tfmat(bbox2camera_mat)

        dist_list, height_list = self._get_spatial(
                cloud, self.target_bin.bbox, bbox2camera, self.target_bin.direction)
        
        cloud_shape = (cloud.height, cloud.width)
        self.dist_img = np.array(dist_list).reshape(cloud_shape)
        self.height_img = np.array(height_list).reshape(cloud_shape)
        #scale to mm from m
        self.dist_img = (self.dist_img * 1000).astype(np.uint8) 
        self.height_img = (self.height_img * 2)  #this is how height is processed..
        with open(pack_path + '/somefile.pkl', 'wb') as f:
            pickle.dump(self.height_img, f)

    @timing
    def _get_spatial(self, cloud, bbox, trans, direction):
        """
        :param trans: transformation from the cloud' parent frame 
                      to the bbox's center 
        :param direction: on the axis of "direction", the distance from shelf is
                          calculated only from a wall in positive coordinate of the axis
                          (Currently only x axis is supported)
        """
        assert direction == 'x' 

        ## represent a point in bounding box's frame 
        # http://answers.ros.org/question/9103/how-to-transform-pointcloud2-with-tf/
        cloud_transformed = do_transform_cloud(cloud, trans)
        gen = point_cloud2.read_points(
                cloud_transformed, 
                skip_nans=False, 
                field_names=("x", "y", "z"))
        points = [point for point in gen]

        def spatial(point, bbox):
            def d2wall(coord, width):
                if coord >= 0 and coord < width/2:
                    return abs(width/2 - coord)
                elif coord < 0 and abs(coord) < width/2:
                    return abs(coord + width/2)
                else:
                    return 0
            def d2front(coord, width):
                if abs(coord) <= width/2:
                    return width/2 - coord
                else:
                    return 0
            d2wall_x_back = d2front(point[0], float(bbox.dimensions.x))
            d2wall_y = d2wall(point[1], float(bbox.dimensions.y))
            d2wall_z = d2wall(point[2], float(bbox.dimensions.z))
            d2wall_z_bottom = d2front(-point[2], float(bbox.dimensions.z))
            return (min(d2wall_x_back, d2wall_y, d2wall_z), d2wall_z_bottom)

        spatial_list = [spatial(point,bbox) for point in points]
        dist_list = [p_info[0] for p_info in spatial_list]
        height_list = [p_info[1] for p_info in spatial_list]
        return dist_list, height_list

    def set_apc_sample(self):
        # TODO: work on define_later later
        define_later = np.zeros((self.camera_info.height, self.camera_info.width))
        data = {}
        data['objects'] = self.objects
        data['dist2shelf_image'] = self.dist_img
        data['depth_image'] = define_later
        data['has3D_image'] = define_later
        data['height3D_image'] = self.height_img
        data['height2D_image'] = define_later
        self.apc_sample = APCSample(
                image_input=self.img_color,
                bin_mask_input=self.mask_image,
                data_input=data,
                labeled=False,
                infer_shelf_mask=False,
                pickle_mask=False)

    def segmentation(self):
        self.predicted_segment = self.trained_segmenter.predict(
                apc_sample=self.apc_sample, 
                desired_object=self.target_object)

        samp_pred = APCSamplePredicted(
                self.apc_sample, 
                self.trained_segmenter.segments,
                self.trained_segmenter.posterior_images_smooth)
        time = strftime("%Y_%m_%d_%H_%M_")
        bin_object = self.target_bin.bin_name + '_' + self.target_object
        fpath = pack_path + '/data/segmented/' 
        with open(fpath + time + bin_object + '.pkl', 'wb') as f:
            pickle.dump(samp_pred, f)
        # debug
#        APCSamplePredicted.init_global_display()
#        samp_pred.save_segment(fpath + time + bin_object + '.png')


class PipelineROS(Pipeline):
    """
    TODO: make it compatible with changing targets
    """
    def __init__(self):
        super(self.__class__, self).__init__(
            initial_pos_list=rospy.get_param('~initial_pos_list'),
            initial_rot_list=rospy.get_param('~initial_rot_list'),
            dimensions_list=rospy.get_param('~dimensions_list'),
            frame_id_list=rospy.get_param('~frame_id_list'),
            prefixes=rospy.get_param('~prefixes'),
            camera_directions=rospy.get_param('~camera_directions'),
            trained_pkl_path=rospy.get_param('~trained_pkl_path'),
            target='bin_g',
            target_object=rospy.get_param('~target_object'),
            objects=rospy.get_param('~objects'),
            camera_info=None,
            img_color=None)

    def subscribe(self): 
        self.buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.buffer) 
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.bridge = CvBridge()

        self.pc_sub = message_filters.Subscriber('~input', PointCloud2) 
        self.cam_info_sub = message_filters.Subscriber('~input/info', CameraInfo) 
        self.img_sub = message_filters.Subscriber('~input/image', Image) 
        self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.pc_sub,  self.img_sub, self.cam_info_sub], 
                queue_size=100,
                slop=0.5)
        self.sync.registerCallback(self._callback)

    def unsubscribe(self):
        pass

    def _callback(self, cloud, img_msg, camera_info):
        print "started"
        ## mask image
        self.camera_info = camera_info  #this can be optimized
        try:
            img_color = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.img_color = img_color

        # get transform
        camera2bb_base = self.buffer.lookup_transform(
                target_frame=camera_info.header.frame_id,
                source_frame=self.target_bin.bbox.header.frame_id,
                time=rospy.Time.now(),
                timeout=rospy.Duration(1.0))
        # get mask_image  
        self.get_mask_img(camera2bb_base)

        ## dist image
        bb_base2camera = self.buffer.lookup_transform(
                target_frame=self.target_bin.bbox.header.frame_id,
                source_frame=cloud.header.frame_id,
                time=rospy.Time.now(),
                timeout=rospy.Duration(1.0)) 
        self.get_spatial_img(bb_base2camera, cloud)

        self.set_apc_sample()
        self.segmentation()
        print "ended"


if __name__ == '__main__':
    rospy.init_node('segmentation')
    seg = PipelineROS()
    seg.subscribe()
    rospy.spin()
