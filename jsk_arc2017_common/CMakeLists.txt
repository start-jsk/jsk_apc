cmake_minimum_required(VERSION 2.8.3)
project(jsk_arc2017_common)

find_package(catkin REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

# XXX

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

# XXX

###################################
## catkin specific configuration ##
###################################

catkin_package()

###########
## Build ##
###########

add_custom_target(install_data ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_data.py)

#############
## Install ##
#############

# XXX

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python()
  roslint_add_test()
  find_package(rostest REQUIRED)
  add_rostest(test/publish_shelf_bin_bbox.test)
endif()
