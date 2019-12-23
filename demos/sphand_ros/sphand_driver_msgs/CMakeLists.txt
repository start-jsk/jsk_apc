cmake_minimum_required(VERSION 2.8.3)
project(sphand_driver_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  force_proximity_ros
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  IntensityProxCalibInfo.msg
  IntensityProxCalibInfoArray.msg
  ProximityStampedArray.msg
)

add_service_files(
  FILES
  TurnOffSensors.srv
)

add_action_files(
  FILES
  CalibJoint.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  force_proximity_ros
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

# XXX

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS
  actionlib_msgs
  force_proximity_ros
  message_runtime
)

###########
## Build ##
###########

# XXX

#############
## Install ##
#############

# XXX

#############
## Testing ##
#############

# XXX

