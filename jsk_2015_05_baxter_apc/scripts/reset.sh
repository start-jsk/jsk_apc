#!/usr/bin/env bash

if [ $# -eq 2 ]; then
    rosparam set /left_process/target "$1"
    rosparam set /right_process/target "$2"
    rosparam set /left_process/state "$3"
    rosparam set /right_process/state "$3"
    rosrun jsk_2015_05_baxter_apc show.sh
elif [ $# -eq 0 ]; then
    rosparam set /left_process/target "a"
    rosparam set /right_process/target "c"
    rosparam set /left_process/state "pick_object"
    rosparam set /right_process/state "pick_object"
    rosrun jsk_2015_05_baxter_apc show.sh
else
    echo "./reset.sh target_name state"
    echo "If no args: "
    echo " [usage] ./reset.sh <right_process target> <left_process target> <state>"
    echo " ./reset.sh a c pick_object"
fi