#!/usr/bin/env bash

if [ $# -eq 2 ]; then
    rosparam set /target "$1"
    rosparam set /left_limb/state "$2"
    rosparam set /right_limb/state "$2"
    rosrun jsk_2014_picking_challenge show.sh
elif [ $# -eq 0 ]; then
    rosparam set /target "a"
    rosparam set /left_limb/state "pick_object"
    rosparam set /right_limb/state "pick_object"
    rosrun jsk_2014_picking_challenge show.sh
else
    echo "./reset.sh target_name state"
    echo "If no args: "
    echo " ./reset.sh a pick_object"
fi