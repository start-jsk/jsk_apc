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

# Mark executables for installation
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
if(${mraa_FOUND})
  install(TARGETS gripper_v7_loop_node gripper_v8_loop_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

# Mark Python executables for installation
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/installing_python.html
catkin_install_python(PROGRAMS
  node_scripts/check_tof_min_max.py
  node_scripts/force_estimator.py
  node_scripts/intensity_prox_calibrator.py
  node_scripts/synchronized_proximity_printer.py
  node_scripts/tof_average.py
  node_scripts/tof_low_pass_filter.py
  scripts/format_printed_prox_to_csv
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark other files for installation (e.g. launch and bag files, etc.)
# See http://docs.ros.org/melodic/api/catkin/html/howto/format1/installing_other.html
install(DIRECTORY
  camera_info config euslisp launch supervisor udev
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
install(PROGRAMS
  scripts/create_supervisor_conf
  scripts/create_udev_rules
  scripts/launch_left_gripper.sh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python()
  roslint_add_test()
  find_package(rostest REQUIRED)
  add_rostest(test/test_baxterlgv8_interface.test)
  find_package(jsk_tools REQUIRED)
  find_package(jsk_apc2016_common REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "rosrun jsk_apc2016_common euslint ${PROJECT_SOURCE_DIR}/euslisp")
  endif()
endif()
