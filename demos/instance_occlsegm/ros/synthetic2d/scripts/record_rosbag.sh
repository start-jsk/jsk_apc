#!/bin/bash -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rosbag record -b 0 \
  /tf \
  /robot/joint_states \
  /left_hand_camera/left/rgb/camera_info \
  /left_hand_camera/left/rgb/image_raw/compressed \
  /left_hand_camera/left/depth/camera_info \
  /left_hand_camera/left/depth/image_raw/compressedDepth \
  /left_hand_camera/right/rgb/camera_info \
  /left_hand_camera/right/rgb/image_raw/compressed \
  /left_hand_camera/right/depth/camera_info \
  /left_hand_camera/right/depth/image_raw/compressedDepth \
  /left_hand_camera/mask_rcnn_relook/output/viz \
  /left_hand_camera/mask_rcnn_relook/output/target_mask \
  /left_hand_camera/bboxes_to_bbox_target0/output \
  /left_hand_camera/poses_to_pose_target0/output \
  /left_hand_camera/cluster_indices_decomposer_target/debug_output \
  -O $HERE/../rosbags/pick_from_apile_$(date +%Y%m%d_%H%M%S).bag

  # /right_hand_camera/left/rgb/camera_info \
  # /right_hand_camera/left/rgb/image_raw/compressed \
  # /right_hand_camera/left/depth/camera_info \
  # /right_hand_camera/left/depth/image_raw/compressedDepth \
  # /right_hand_camera/right/rgb/camera_info \
  # /right_hand_camera/right/rgb/image_raw/compressed \
  # /right_hand_camera/right/depth/camera_info \
  # /right_hand_camera/right/depth/image_raw/compressedDepth \
