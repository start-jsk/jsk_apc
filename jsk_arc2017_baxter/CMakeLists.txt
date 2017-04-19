cmake_minimum_required(VERSION 2.8.3)
project(jsk_arc2017_baxter)

find_package(catkin REQUIRED COMPONENTS
    rospy
    roseus
)

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

# XXX

#############
## Install ##
#############

# XXX

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(jsk_tools REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    file(GLOB_RECURSE _eus_files euslisp/*.l)
    foreach(_eus_file IN LISTS _eus_files)
      jsk_tools_add_shell_test(COMMAND "rosrun jsk_2016_01_baxter_apc euslint ${_eus_file}")
    endforeach()
  endif()
endif()
