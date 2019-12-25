cmake_minimum_required(VERSION 2.8.3)
project(grasp_prediction_arc2017)

find_package(catkin REQUIRED catkin_virtualenv)

catkin_python_setup()

add_custom_target(${PROJECT_NAME}_install_data ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_data.py)

################################################
## Declare ROS messages, services and actions ##
################################################

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES grasp_prediction_arc2017
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

catkin_generate_virtualenv(
  EXTRA_PIP_ARGS
    -qqq  # Suppress log not to exceed maximum length in travis test
    --upgrade
    --upgrade-strategy=eager
    # '--upgrade-strategy=only-if-needed' does not work in following case:
    # python-dateutil is required by matplotlib.
    # It is satisfied and skipped as python-dateutil 2.4.2 is already installed.
    # Later, panda require python-dateutil>=2.5.0, but pip does not execute upgrading.
)

###########
## Build ##
###########

catkin_install_python(
  PROGRAMS
  nodes/fcn_object_segmentation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY nodes scripts
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY samples launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS)
install(FILES requirements.txt
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(nodes)
  roslint_python(python)
  roslint_add_test()

  find_package(rostest REQUIRED)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch/play_rosbag.launch)
  roslaunch_add_file_check(samples/sample_fcn_object_segmentation.launch)
  add_rostest(samples/sample_fcn_object_segmentation.launch ARGS gui:=false GPU:=-1)
endif()
