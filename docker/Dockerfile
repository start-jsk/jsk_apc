ARG ROS_DISTRO
ARG UBUNTU_VERSION
FROM osrf/ros:${ROS_DISTRO}-desktop-${UBUNTU_VERSION}
ENV DEBIAN_FRONTEND noninteractive

RUN echo "deb http://packages.ros.org/ros-testing/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y wget git && \
    rm -rf /var/lib/apt/lists/*

RUN wget http://packages.ros.org/ros.key -O - | apt-key add -
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y apt-utils && \
    rm -rf /var/lib/apt/lists/*

# FIXME: https://github.com/start-jsk/jsk_apc/pull/2664
ARG ROS_DISTRO
RUN apt-get update && \
    if [ ${ROS_DISTRO} = "noetic" ]; then \
      apt-get install -y \
        python3-catkin-tools \
        python3-rosdep \
        python3-setuptools \
        python3-wstool \
        python3-pip; \
    else \
      apt-get install -y \
        python-catkin-tools \
        python-rosdep \
        python-setuptools \
        python-wstool \
        python-pip; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ARG ROS_DISTRO
RUN if [ ${ROS_DISTRO} = "indigo" ]; then \
      apt-get update && \
      apt-get install -y software-properties-common && \
      add-apt-repository -y ppa:longsleep/python2.7-backports && \
      apt-get update && \
      apt-get dist-upgrade -y; \
    fi && \
    pip install 'pip==9.0.3' && \
    pip install 'setuptools==44.1.0' && \
    pip install 'dlib==19.21.1' && \
    rm -rf /var/lib/apt/lists/*

ARG ROS_DISTRO
RUN cd ~ && \
    mkdir -p ros/${ROS_DISTRO}/src && \
    cd ros/${ROS_DISTRO}/src && \
    wstool init && \
    wstool set start-jsk/jsk_apc https://github.com/start-jsk/jsk_apc.git -v master --git -y && \
    wstool up -j 2 && \
    wstool merge start-jsk/jsk_apc/.travis.rosinstall && \
    wstool merge start-jsk/jsk_apc/.travis.rosinstall.${ROS_DISTRO} && \
    wstool up -j 2

# /opt/ros/${ROS_DISTRO}/share can be changed after rosdep install, so we run it 3 times.
RUN apt-get update && rosdep update --include-eol-distros && \
    for i in $(seq 3); do rosdep install --rosdistro ${ROS_DISTRO} -r -y -i --from-paths /opt/ros/${ROS_DISTRO}/share ~/ros/${ROS_DISTRO}/src; done && \
    rm -rf /var/lib/apt/lists/*

ARG TESTING
ARG ROS_DISTRO
RUN if [ ${TESTING} = "true" ] ; then \
      rm -rf ~/ros; \
    else \
      . /opt/ros/${ROS_DISTRO}/setup.sh && \
      cd ~/ros/${ROS_DISTRO} && \
      if [ ${ROS_DISTRO} = "indigo" ]; then \
        catkin build jsk_apc2015_common jsk_apc2016_common jsk_arc2017_common jsk_2015_05_baxter_apc jsk_2016_01_baxter_apc jsk_arc2017_baxter selective_dualarm_stowing sphand_driver sphand_driver_msgs vl53l0x_mraa_ros baxtergv6_apc2016;\
      else \
        catkin build; \
      fi; \
    fi
