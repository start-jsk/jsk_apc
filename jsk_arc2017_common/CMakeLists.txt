cmake_minimum_required(VERSION 2.8.3)
project(jsk_arc2017_common)

find_package(catkin REQUIRED
    dynamic_reconfigure
    message_generation
    std_msgs
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
    FILES
    Content.msg
    ContentArray.msg
    Weight.msg
    WeightStamped.msg
    WorkOrder.msg
    WorkOrderArray.msg
    ObjectCandidates.msg
)

add_service_files(
    FILES
    UpdateJSON.srv
    GetGraspStyle.srv
)

generate_dynamic_reconfigure_options(
    cfg/CandidatesPublisher.cfg
    cfg/PublishDatasetV3.cfg
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

catkin_package(
    CATKIN_DEPENDS
    message_runtime
)

###########
## Build ##
###########

# add_custom_target(install_data ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_data.py)

#############
## Install ##
#############

if($ENV{ROS_DISTRO} STREQUAL "indigo") # on noetic it needs catkin_install_python to support Python3 and it does not work on indigo for some reason...
  install(DIRECTORY node_scripts
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
          USE_SOURCE_PERMISSIONS
  )
  install(DIRECTORY scripts
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
          USE_SOURCE_PERMISSIONS
  )
else()
  install(DIRECTORY scripts
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
          USE_SOURCE_PERMISSIONS
          PATTERN "*"
          PATTERN "*/*.py" EXCLUDE
  )
  install(DIRECTORY node_scripts
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
          USE_SOURCE_PERMISSIONS
          PATTERN "*"
          PATTERN "*/*.py" EXCLUDE
  )

  file(GLOB SCRIPT_PROGRAMS scripts/*.py)
  catkin_install_python(
    PROGRAMS ${SCRIPT_PROGRAMS}
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  )
  file(GLOB SCRIPT_PROGRAMS node_scripts/*.py)
  catkin_install_python(
    PROGRAMS ${SCRIPT_PROGRAMS}
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/node_scripts
  )
endif()

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(python/jsk_arc2017_common/tests)
  find_package(roslint REQUIRED)
  roslint_python()
  roslint_add_test()
  find_package(rostest REQUIRED)
  add_rostest(test/work_order_publisher.test)
endif()
