cmake_minimum_required(VERSION 2.8.3)
project(baxtergv6_apc2016)

find_package(catkin REQUIRED)

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
#  LIBRARIES baxtergv6_apc2016
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

# Mark Python executables for installation
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/installing_python.html
catkin_install_python(PROGRAMS
  node_scripts/json_saver.py
  node_scripts/state_server.py
  node_scripts/work_order_publisher.py
  scripts/download_data.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark other files for installation (e.g. launch and bag files, etc.)
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/installing_other.html
install(DIRECTORY
  config data euslisp launch rvizconfig
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
install(PROGRAMS
  scripts/install_data
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(node_scripts)
  roslint_python(scripts)
  roslint_add_test()
  find_package(jsk_tools REQUIRED)
  find_package(jsk_apc2016_common REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "rosrun jsk_apc2016_common euslint ${PROJECT_SOURCE_DIR}/euslisp")
  endif()
endif()
