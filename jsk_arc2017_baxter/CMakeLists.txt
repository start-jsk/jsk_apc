cmake_minimum_required(VERSION 2.8.3)
project(jsk_arc2017_baxter)

find_package(catkin REQUIRED COMPONENTS
    force_proximity_ros
    message_generation
    roseus
    std_msgs
    baxter_core_msgs
    hardware_interface
    transmission_interface
    controller_manager
    dynamixel_msgs
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
    std_msgs
    baxter_core_msgs
    hardware_interface
    transmission_interface
    controller_manager
    dynamixel_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${Boost_INCLUDE_DIRS}
)

## -------------------
## baxter.l generation
## -------------------
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/robots/baxter.l
  COMMAND rosrun euscollada collada2eus baxter.dae baxter.yaml baxter.l
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/robots
  DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.dae ${PROJECT_SOURCE_DIR}/robots/baxter.yaml)
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/robots/baxter.dae
  COMMAND rosrun collada_urdf urdf_to_collada baxter.urdf baxter.dae
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/robots
  DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.urdf)
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/robots/baxter.urdf
  COMMAND ROS_PACKAGE_PATH=${PROJECT_SOURCE_DIR}:$ENV{ROS_PACKAGE_PATH} rosrun xacro xacro baxter.xacro > baxter.urdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/robots
  DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.xacro
          ${PROJECT_SOURCE_DIR}/robots/right_vacuum_gripper.xacro)
# get collada-dom version
find_package(PkgConfig)
pkg_check_modules(COLLADA collada-dom>=2.4.4)
if(${COLLADA_FOUND})
  add_custom_target(generate_baxter_lisp ALL DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.l)
else()
  pkg_check_modules(COLLADA collada-dom)
  message(WARNING "urdf_to_collada requries collada-dom >= 2.4.4, installed version is ${COLLADA_VERSION}")
endif()

## Declare a C++ executable
add_executable(gripper_v6_ros_control_node src/gripper_v6_ros_control.cpp)

## Add cmake target dependencies of the executable
add_dependencies(gripper_v6_ros_control_node ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(gripper_v6_ros_control_node ${catkin_LIBRARIES})

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
  find_package(jsk_tools REQUIRED)
  find_package(jsk_apc2016_common REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "rosrun jsk_apc2016_common euslint ${PROJECT_SOURCE_DIR}")
  endif()
endif()
