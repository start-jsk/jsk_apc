#!/bin/bash

rospack list | grep baxter | cut -d\  -f1 | xargs -t -n1 rosversion

