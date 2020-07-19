cmake_minimum_required(VERSION 2.8.3)
project(dualarm_grasping)

find_package(catkin REQUIRED
    dynamic_reconfigure
    jsk_recognition_msgs
    message_generation
    roseus
    std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  GraspClassificationResult.msg
)

add_service_files(
  FILES
  GetAnother.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  jsk_recognition_msgs
)

add_custom_target(${PROJECT_NAME}_install_models ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_models.py)

generate_dynamic_reconfigure_options(
  cfg/DualarmGraspSegmentation.cfg
  cfg/DualarmOccludedGraspInstanceSegmentation.cfg
)

catkin_package(
   CATKIN_DEPENDS
   std_msgs
   jsk_recognition_msgs
   message_runtime
)
