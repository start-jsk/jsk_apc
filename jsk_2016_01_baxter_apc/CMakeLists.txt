cmake_minimum_required(VERSION 2.8.3)
project(jsk_2016_01_baxter_apc)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roseus
    control_msgs
    roscpp
    std_msgs
    baxter_core_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
#catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jsk_apc2016_common
  CATKIN_DEPENDS control_msgs roscpp std_msgs baxter_core_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(jsk_apc2016_common
#   src/${PROJECT_NAME}/jsk_apc2016_common.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(jsk_apc2016_common ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

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
  COMMAND rosrun xacro xacro baxter.xacro > baxter.urdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/robots
  DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.xacro)
# get collada-dom version
find_package(PkgConfig)
pkg_check_modules(COLLADA collada-dom>=2.4.4 )
if ( ${COLLADA_FOUND} )
  add_custom_target(generate_baxter_lisp ALL DEPENDS ${PROJECT_SOURCE_DIR}/robots/baxter.l)
else()
  pkg_check_modules(COLLADA collada-dom)
  message(WARNING "urdf_to_collada requries collada-dom >= 2.4.4, installed version is ${COLLADA_VERSION}")
endif()

## Declare a C++ executable
add_executable(gripper_joint_states_publisher src/gripper_joint_states_publisher.cpp)
add_executable(gripper_trajectory_server src/gripper_trajectory_server.cpp)
add_executable(enable_gripper src/enable_gripper.cpp)

## Add cmake target dependencies of the executable
add_dependencies(gripper_joint_states_publisher ${catkin_EXPORTED_TARGETS})
add_dependencies(gripper_trajectory_server ${catkin_EXPORTED_TARGETS})
add_dependencies(enable_gripper ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(gripper_joint_states_publisher ${catkin_LIBRARIES})
target_link_libraries(gripper_trajectory_server ${catkin_LIBRARIES})
target_link_libraries(enable_gripper ${catkin_LIBRARIES})

add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/worlds/pick_kiva.world
  DEPENDS ${PROJECT_SOURCE_DIR}/worlds/pick_kiva.world.erb
  COMMAND erb ${PROJECT_SOURCE_DIR}/worlds/pick_kiva.world.erb > ${PROJECT_SOURCE_DIR}/worlds/pick_kiva.world
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/worlds/)
add_custom_target(compile_pick_kiva_world ALL
  DEPENDS ${PROJECT_SOURCE_DIR}/worlds/pick_kiva.world)
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/worlds/stow_kiva.world
    DEPENDS ${PROJECT_SOURCE_DIR}/worlds/stow_kiva.world.erb
    COMMAND erb ${PROJECT_SOURCE_DIR}/worlds/stow_kiva.world.erb > ${PROJECT_SOURCE_DIR}/worlds/stow_kiva.world
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/worlds/)
add_custom_target(compile_stow_kiva_world ALL
    DEPENDS ${PROJECT_SOURCE_DIR}/worlds/stow_kiva.world)

#
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS jsk_apc2016_common jsk_apc2016_common_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_jsk_apc2016_common.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
if(CATKIN_ENABLE_TESTING)
  find_package(roslint REQUIRED)
  roslint_python(node_scripts)
  roslint_add_test()
  find_package(rostest REQUIRED)
  add_rostest(test/work_order.test)
  add_custom_target(download_test_data COMMAND ${PROJECT_SOURCE_DIR}/scripts/download_test_data.sh)
  if(TARGET tests)
    add_dependencies(tests download_test_data)
  endif()
  add_rostest(test/test_move_arm_to_bin.test)
  add_rostest(test/test_ik_in_bin.test)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch/include/vacuum_gripper.launch)
  find_package(jsk_tools REQUIRED)
  file(GLOB_RECURSE _eus_files *.l)
  foreach(_eus_file IN LISTS _eus_files)
    jsk_tools_add_shell_test(COMMAND "${PROJECT_SOURCE_DIR}/scripts/euslint ${_eus_file}")
  endforeach()
endif()
