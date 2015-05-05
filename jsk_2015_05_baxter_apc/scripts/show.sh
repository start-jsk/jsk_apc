#!/usr/bin/env bash

TARGET=`rosparam get /target`
L_STATE=`rosparam get /left_limb/state`
R_STATE=`rosparam get /right_limb/state`

echo "/target           is "$TARGET
echo "/left_limb/state  is "$L_STATE
echo "/right_limb/state is "$R_STATE