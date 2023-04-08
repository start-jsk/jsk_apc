cmake_minimum_required(VERSION 2.8.3)
project(grasp_fusion)

find_package(catkin REQUIRED catkin_virtualenv)

catkin_python_setup()

# add_custom_target(${PROJECT_NAME}_install_data ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_data.py)
# Don't install data on build time to prevent build failure on travis due to googledrive access limit

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
#  LIBRARIES grasp_fusion
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

if($ENV{ROS_DISTRO} STREQUAL "noetic")
  catkin_generate_virtualenv(
    PYTHON_INTERPRETER python3
    CHECK_VENV FALSE
    EXTRA_PIP_ARGS
      -qqq  # Suppress log not to exceed maximum length in travis test
      --upgrade
      --upgrade-strategy=eager
      # '--upgrade-strategy=only-if-needed' does not work in following case:
      # python-dateutil is required by matplotlib.
      # It is satisfied and skipped as python-dateutil 2.4.2 is already installed.
      # Later, panda require python-dateutil>=2.5.0, but pip does not execute upgrading.
  )
else()
  catkin_generate_virtualenv(
    PYTHON_INTERPRETER python2
    CHECK_VENV FALSE
    EXTRA_PIP_ARGS
      -qqq  # Suppress log not to exceed maximum length in travis test
      --upgrade
      --upgrade-strategy=eager
      # '--upgrade-strategy=only-if-needed' does not work in following case:
      # python-dateutil is required by matplotlib.
      # It is satisfied and skipped as python-dateutil 2.4.2 is already installed.
      # Later, panda require python-dateutil>=2.5.0, but pip does not execute upgrading.
  )
endif()

###########
## Build ##
###########

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(
  PROGRAMS
  node_scripts/affordance_segmentation.py
  node_scripts/bounding_box_to_tf.py
  node_scripts/get_heightmap.py
  node_scripts/mask_rcnn_instance_segmentation.py
  node_scripts/primitive_matching.py
  node_scripts/sole_affordance_segmentation.py
  node_scripts/weight_change_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY euslisp node_scripts scripts
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY config launch samples
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS)
install(FILES requirements.txt
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(node_scripts)
  roslint_python(python)
  roslint_python(tests)
  roslint_add_test()

  catkin_add_nosetests(tests DEPENDENCIES ${PROJECT_NAME}_generate_virtualenv)
endif()
