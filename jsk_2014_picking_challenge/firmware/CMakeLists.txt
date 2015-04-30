cmake_minimum_required(VERSION 2.8.3)

include_directories(${ROS_LIB_DIR})

# Remove this if using an Arduino without native USB (eg, other than Leonardo)
add_definitions(-DUSB_CON)

generate_arduino_firmware(switching_gripper_left
  SRCS switching_gripper_left.cpp ${ROS_LIB_DIR}/time.cpp
  BOARD atmega328
  PORT /dev/ttyUSB0
)

generate_arduino_firmware(switching_gripper_right
  SRCS switching_gripper_right.cpp ${ROS_LIB_DIR}/time.cpp
  BOARD atmega328
  PORT /dev/ttyUSB0
)