#!/bin/bash

# Wait until network is ready
while ! ping -c 1 baxter > /dev/null
do
  sleep 1
done

# Prepare FIFO for logging. Perhaps this FIFO is already prepared by LaunchLogger
mkdir /tmp/supervisor
mkfifo /tmp/supervisor/launch_logger_fifo

# Kill previous roslaunch
tmux kill-session -t gripper

# roslaunch in tmux
set -e
tmux new-session -d -s gripper -n roslaunch "script -f /tmp/supervisor/launch_logger_fifo"
## Using pipe-pane like following does not work when -d is specified in new-session:
## tmux pipe-pane -o -t gripper:roslaunch.0 "cat > /tmp/supervisor/launch_logger_fifo"
## capture-pane works, but it only captures current state and does not know which line is new
tmux send-keys -t gripper:roslaunch.0 "source ~/ros/kinetic/devel/setup.bash && rossetip && rossetmaster baxter && roslaunch sphand_driver setup_gripper_v8.launch left_gripper:=true" Enter
