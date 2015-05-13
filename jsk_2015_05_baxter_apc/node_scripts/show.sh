#!/usr/bin/env bash

L_TARGET=`rosparam get /left_process/target`
R_TARGET=`rosparam get /right_process/target`
L_STATE=`rosparam get /left_process/state`
R_STATE=`rosparam get /right_process/state`

echo "/left_process/target  is "$L_TARGET
echo "/right_process/target is "$R_TARGET
echo "/left_process/state   is "$L_STATE
echo "/right_process/state  is "$R_STATE