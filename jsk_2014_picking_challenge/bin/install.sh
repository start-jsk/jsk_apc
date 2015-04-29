#!/bin/bash

# for arduino
script_dir="$(cd "$(dirname "${BASH_SOURCE:-${(%):-%N}}")"; pwd)"
sudo ln -sf ${script_dir}/udev-settings/10-arduino.rules /etc/udev/rules.d/10-arduino.rules
sudo ln -sf ${script_dir}/.dotfiles/udev-settings/11-arduino.rules /etc/udev/rules.d/11-arduino.rules
