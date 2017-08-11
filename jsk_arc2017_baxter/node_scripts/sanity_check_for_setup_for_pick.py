#!/usr/bin/env python

import sys

import rospy

from jsk_tools.sanity_lib import checkNodeState
from jsk_tools.sanity_lib import TopicPublishedChecker


def main():
    # check node is exists
    nodes = [
        '/left_hand_camera/left/left_nodelet_manager',
        '/left_hand_camera/right/right_nodelet_manager',
        '/left_hand_camera/stereo/stereo_image_proc',

        '/right_hand_camera/left/left_nodelet_manager',
        '/right_hand_camera/right/right_nodelet_manager',
        '/right_hand_camera/stereo/stereo_image_proc',
    ]
    for node in nodes:
        checkNodeState(node, needed=True)

    topic_checkers = []

    # common for left/right hand camera
    topics = [
        '/robot/joint_states',
        '/transformable_bin_markers/output/boxes',

        '/gripper_front/limb/left/dxl/motor_states/port',
        '/gripper_front/limb/left/dxl/finger_tendon_controller/state',
        '/gripper_front/limb/left/dxl/finger_yaw_joint_controller/state',
        '/gripper_front/limb/left/dxl/prismatic_joint_controller/state',
        '/gripper_front/limb/left/dxl/vacuum_pad_tendon_controller/state',

        '/gripper_front/limb/right/dxl/motor_states/port',
        '/gripper_front/limb/right/dxl/finger_tendon_controller/state',
        '/gripper_front/limb/right/dxl/finger_yaw_joint_controller/state',
        '/gripper_front/limb/right/dxl/prismatic_joint_controller/state',
        '/gripper_front/limb/right/dxl/vacuum_pad_tendon_controller/state',

        '/vacuum_gripper/limb/left/state',
        '/vacuum_gripper/limb/right/state',

        '/lgripper_sensors',
        '/rgripper_sensors',
    ]
    for topic in topics:
        topic_checkers.append(TopicPublishedChecker(topic, timeout=5))

    # for left/right hand camera
    topics = [
        # setup_for_pick.launch
        'left/rgb/camera_info',
        'left/rgb/image_rect_color',
        'left/depth_registered/camera_info',
        'left/depth_registered/sw_registered/image_rect',
        'left/depth_registered/points',

        'right/rgb/camera_info',
        'right/rgb/image_rect_color',

        'stereo/depth_registered/image_rect',

        'fused/rgb/camera_info',
        'fused/rgb/image_rect_color',
        'fused/depth_registered/image_rect',
        'fused/depth_registered/points',

        'fcn_object_segmentation/output/proba_image',
        'apply_context_to_label_proba/output/label',

        'attention_clipper_target_bin/output/point_indices',
        'extract_indices_target_bin/output',

        'label_to_cluster_indices/output',
        'cluster_indices_decomposer_label/boxes',
        'cluster_indices_decomposer_label/centroid_pose_array',

        'label_to_mask/output',
        'mask_to_point_indices/output',
        'extract_indices_target_label/output',

        'cluster_indices_decomposer_target/boxes',
        'cluster_indices_decomposer_target/centroid_pose_array',
        'bbox_array_to_bbox/output',
    ]
    for side in ['left', 'right']:
        for topic in topics:
            topic = '/%s_hand_camera/%s' % (side, topic)
            topic_checkers.append(TopicPublishedChecker(topic, timeout=5))

    for checker in topic_checkers:
        checker.check()


if __name__ == '__main__':
    rospy.init_node('sanity_check_for_setup_for_pick')
    sys.stdout = sys.__stderr__
    main()
    sys.stdout = sys.__stdout__
