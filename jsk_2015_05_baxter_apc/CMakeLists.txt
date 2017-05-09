cmake_minimum_required(VERSION 2.8.3)
project(jsk_2015_05_baxter_apc)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    geometry_msgs
    message_generation
    roseus
    rosserial_arduino
    rosserial_client
    std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(gazebo REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
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

add_message_files(
  FILES
  BinContents.msg
  BinContentsArray.msg
  BoolStamped.msg
  ObjectRecognition.msg
  ProbabilisticLabels.msg
  QrStamp.msg
  QrStampes.msg
  SetObjectPosition.msg
  SetObjectPositionArray.msg
  WorkOrder.msg
  WorkOrderArray.msg
  )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Cue.srv
  MoveArm.srv
  ObjectMatch.srv
  ObjectVerification.srv
  QrStampsrv.srv
  ReleaseItem.srv
  StringEmpty.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

## Add dynamic reconfigure api
generate_dynamic_reconfigure_options(
  cfg/SIFTMatcher.cfg
  cfg/ColorCutback.cfg
  cfg/DilateImage.cfg
)

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
#  LIBRARIES jsk_2015_05_baxter_apc
  CATKIN_DEPENDS geometry_msgs message_runtime rosserial_arduino rosserial_client std_msgs
#  DEPENDS system_lib
)

if(rosserial_generate_ros_lib) # hydro does not have this macro
  rosserial_generate_ros_lib(
    PACKAGE rosserial_arduino
    SCRIPT make_libraries.py
  )

  rosserial_configure_client(
    DIRECTORY firmware
    TOOLCHAIN_FILE ${ROSSERIAL_ARDUINO_TOOLCHAIN}
  )

  rosserial_add_client_target(firmware switching_gripper_left ALL)
  rosserial_add_client_target(firmware switching_gripper_left-upload)
  rosserial_add_client_target(firmware switching_gripper_right ALL)
  rosserial_add_client_target(firmware switching_gripper_right-upload)
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a cpp library
# add_library(jsk_2015_05_baxter_apc
#   src/${PROJECT_NAME}/jsk_2015_05_baxter_apc.cpp
# )

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

## Declare a cpp executable
# add_executable(jsk_2015_05_baxter_apc_node src/jsk_2015_05_baxter_apc_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(jsk_2015_05_baxter_apc_node jsk_2015_05_baxter_apc_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(jsk_2015_05_baxter_apc_node
#   ${catkin_LIBRARIES}
# )

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
# install(TARGETS jsk_2015_05_baxter_apc jsk_2015_05_baxter_apc_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY
  config euslisp launch meshes node_scripts robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_jsk_2015_05_baxter_apc.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

if(CATKIN_ENABLE_TESTING)
  find_package(jsk_tools REQUIRED)
  if(${jsk_tools_VERSION} VERSION_GREATER 2.0.13)
    jsk_tools_add_shell_test(COMMAND "roseus ${PROJECT_SOURCE_DIR}/euslisp/test/test-move-arm-to-bin.l")
  endif()
endif()
