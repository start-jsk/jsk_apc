cmake_minimum_required(VERSION 2.8.3)
project(grasp_fusion)

find_package(catkin REQUIRED)

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
#  LIBRARIES grasp_fusion
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

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

install(DIRECTORY euslisp node_scripts scripts
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY config launch samples
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(node_scripts)
  roslint_python(python)
  roslint_python(tests)
  roslint_add_test()

  catkin_add_nosetests(tests)
endif()
