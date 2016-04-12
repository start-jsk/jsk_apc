#!/usr/bin/env zsh

# source 99.dotfile.sh from same directory as this file
_THIS_DIR=$(builtin cd -q "`dirname "$0"`" > /dev/null && pwd)
if [ -f "$_THIS_DIR/99.dotfile.bash" ]; then
    source "$_THIS_DIR/99.dotfile.bash"
else
    # this is temporary code to avoid error caused by bug in ros/catkin
    # see https://github.com/jsk-ros-pkg/jsk_common/issues/885
    source "$_THIS_DIR/etc/catkin/profile.d/99.dotfile.bash"
fi
unset _THIS_DIR
