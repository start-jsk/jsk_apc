cmake_minimum_required(VERSION 2.8.3)
project(sphand_driver)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  baxter_core_msgs
  controller_manager
  dynamixel_controllers
  dynamixel_msgs
  force_proximity_ros
  hardware_interface
  message_generation
  roscpp
  sphand_driver_msgs
  std_msgs
  transmission_interface
  vl53l0x_mraa_ros
)

## https://answers.ros.org/question/247803/mraa-library-inclusion-in-project/
find_package(PkgConfig REQUIRED)
pkg_check_modules(mraa QUIET mraa)
if(NOT "${mraa_FOUND}")
  string(ASCII 27 Esc)
  message(WARNING "${Esc}[1;33m mraa is not found. You cannot use loop node${Esc}[m")
endif()

catkin_python_setup()

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

# XXX

###################################
## catkin specific configuration ##
###################################

if(${mraa_FOUND})
  catkin_package(
    CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    baxter_core_msgs
    controller_manager
    dynamixel_controllers
    dynamixel_msgs
    force_proximity_ros
    hardware_interface
    message_runtime
    roscpp
    sphand_driver_msgs
    std_msgs
    transmission_interface
    vl53l0x_mraa_ros
    DEPENDS mraa
  )
else()
  catkin_package(
    CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    baxter_core_msgs
    controller_manager
    dynamixel_controllers
    dynamixel_msgs
    force_proximity_ros
    hardware_interface
    message_runtime
    roscpp
    sphand_driver_msgs
    std_msgs
    transmission_interface
    vl53l0x_mraa_ros
  )
endif()

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${mraa_INCLUDE_DIRS}
)

if(${mraa_FOUND})
  ## Declare a C++ executable
  add_executable(gripper_v7_loop_node src/gripper_v7_loop.cpp)
  add_executable(gripper_v8_loop_node src/gripper_v8_loop.cpp)

  ## Add cmake target dependencies of the executable
  add_dependencies(gripper_v7_loop_node ${catkin_EXPORTED_TARGETS})
  add_dependencies(gripper_v8_loop_node ${catkin_EXPORTED_TARGETS})

  ## Specify libraries to link a library or executable target against
  target_link_libraries(gripper_v7_loop_node ${mraa_LIBRARIES} ${catkin_LIBRARIES})
  target_link_libraries(gripper_v8_loop_node ${mraa_LIBRARIES} ${catkin_LIBRARIES})
endif()

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
  add_rostest(test/test_baxterlgv8_interface.test)
  find_package(jsk_tools REQUIRED)
  find_package(jsk_apc2016_common REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "rosrun jsk_apc2016_common euslint ${PROJECT_SOURCE_DIR}/euslisp")
  endif()
endif()
