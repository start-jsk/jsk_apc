#!/bin/bash

dotfile () {
  [ -z $ROS_HOME ] && ros_home=$HOME/.ros || ros_home=$ROS_HOME
  config_dir=$ros_home/jsk_2016_01_baxter_apc
  link_files=(.bashrc .zshrc .vimrc .vim .gitconfig)
  if [ -e $config_dir/current-dotfile ]; then
    current_user="$(sed -n 1p $config_dir/current-dotfile)"
    current_shell="$(sed -n 2p $config_dir/current-dotfile)"
  else
    mkdir -p $config_dir
  fi
  if [ ! $# -eq 2 ]; then
    echo "Current user: $current_user"
    echo "Current shell: $current_shell"
    if [ -e "$current_shell" ]; then
      exec $current_shell --login
    fi
    return 1
  fi
  gh_user=$1
  shell=$2
  echo "Switching user: $current_user -> $gh_user"
  echo $gh_user > $config_dir/current-dotfile
  echo $shell >> $config_dir/current-dotfile
  for file in $link_files; do
    if [ -e ~/$file.$gh_user ]; then
      if [ -L ~/$file -a -d ~/$file ]; then
        rm ~/$file
      fi
      ln -sf ~/$file.$gh_user ~/$file
      echo "Linked $file -> $file.$gh_user"
    fi
  done
  echo "Logging in as '$user' with '$shell'"
  exec $shell --login
}
