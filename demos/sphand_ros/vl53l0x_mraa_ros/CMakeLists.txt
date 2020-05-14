cmake_minimum_required(VERSION 2.8.3)
project(vl53l0x_mraa_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  std_msgs
)

## https://answers.ros.org/question/247803/mraa-library-inclusion-in-project/
find_package(PkgConfig REQUIRED)
pkg_check_modules(mraa QUIET mraa)
if(NOT "${mraa_FOUND}")
  string(ASCII 27 Esc)
  message(WARNING "${Esc}[1;33m mraa is not found. You can only use ROS messages${Esc}[m")
endif()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  RangingMeasurementData.msg
  RangingMeasurementDataStamped.msg
  RangingMeasurementDataStampedArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

# XXX

###################################
## catkin specific configuration ##
###################################

if(${mraa_FOUND})
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vl53l0x_mraa
    CATKIN_DEPENDS message_runtime roscpp std_msgs
    DEPENDS mraa
  )
else()
  catkin_package(
    CATKIN_DEPENDS message_runtime roscpp std_msgs
  )
endif()

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${mraa_INCLUDE_DIRS}
)

if(${mraa_FOUND})
  ## Declare a C++ library
  add_library(vl53l0x_mraa
    src/vl53l0x_mraa/vl53l0x_mraa.cpp
    src/vl53l0x_mraa/core/vl53l0x_api.cpp
    src/vl53l0x_mraa/core/vl53l0x_api_calibration.cpp
    src/vl53l0x_mraa/core/vl53l0x_api_core.cpp
    src/vl53l0x_mraa/core/vl53l0x_api_ranging.cpp
    src/vl53l0x_mraa/core/vl53l0x_api_strings.cpp
    src/vl53l0x_mraa/platform/vl53l0x_platform.cpp
    src/vl53l0x_mraa/platform/vl53l0x_i2c_comms.cpp
  )

  ## Specify libraries to link a library or executable target against
  target_link_libraries(vl53l0x_mraa ${mraa_LIBRARIES})

  ## Declare a C++ executable
  add_executable(single_vl53l0x_ros src/single_vl53l0x_ros.cpp)

  ## Add cmake target dependencies of the executable
  add_dependencies(single_vl53l0x_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  ## Specify libraries to link a library or executable target against
  target_link_libraries(single_vl53l0x_ros vl53l0x_mraa ${catkin_LIBRARIES})
endif()

#############
## Install ##
#############

if(${mraa_FOUND})
  # Mark libraries for installation
  # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
  install(TARGETS vl53l0x_mraa
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )

  # Mark executables for installation
  # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
  install(TARGETS single_vl53l0x_ros
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  # Mark cpp header files for installation
  # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )
endif()

#############
## Testing ##
#############

# XXX

