cmake_minimum_required(VERSION 2.8.3)
project(jskbaxter2_moveit_config)

find_package(catkin REQUIRED)

catkin_package()

add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/config/baxter.srdf
  COMMAND ROS_PACKAGE_PATH=${PROJECT_SOURCE_DIR}:$ENV{ROS_PACKAGE_PATH} rosrun xacro xacro --inorder baxter.srdf.xacro > baxter.srdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/config
  DEPENDS ${PROJECT_SOURCE_DIR}/config/baxter.srdf.xacro
          ${PROJECT_SOURCE_DIR}/config/baxter_base.srdf.xacro
          ${PROJECT_SOURCE_DIR}/config/vacuum_gripper.srdf.xacro)

add_custom_target(generate_baxter_srdf ALL DEPENDS ${PROJECT_SOURCE_DIR}/config/baxter.srdf)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
