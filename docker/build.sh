#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo docker build -t jsk-apc-indigo .
(echo -e "FROM jsk-apc-indigo\nRUN apt-get update\nRUN apt-get -y upgrade\nEXPOSE 22" | sudo docker build -t jsk-apc-indigo -)
