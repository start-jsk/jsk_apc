cmake_minimum_required(VERSION 2.8.3)
project(jsk_arc2017_baxter)

find_package(catkin REQUIRED COMPONENTS
    force_proximity_ros
    message_generation
    roseus
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(FILES
    GripperSensorStates.msg
)

add_service_files(
    FILES
    UpdateState.srv
    GetState.srv
    CheckCanStart.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
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
    message_runtime
    force_proximity_ros
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

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(node_scripts)
  roslint_add_test()
  find_package(rostest REQUIRED)
  add_rostest(test/test_pick_state_server.test)
  add_rostest(test/test_stow_state_server.test)
  add_rostest(test/test_pick_main.test)
  add_rostest(test/test_stow_main.test)
  add_rostest(test/test_pick_arc_interface.test)
  find_package(jsk_tools REQUIRED)
  find_package(jsk_apc2016_common REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "rosrun jsk_apc2016_common euslint ${PROJECT_SOURCE_DIR}")
  endif()
endif()
