ARG ROS_DISTRO
ARG UBUNTU_VERSION
FROM osrf/ros:${ROS_DISTRO}-desktop-${UBUNTU_VERSION}

RUN echo "deb http://packages.ros.org/ros-testing/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y wget git
RUN wget http://packages.ros.org/ros.key -O - | apt-key add -

# FIXME: https://github.com/start-jsk/jsk_apc/pull/2664
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y \
    python-catkin-tools \
    python-rosdep \
    python-setuptools \
    python-wstool

RUN easy_install -U pip && \
    pip install 'pip<10' && \
    pip install -U setuptools

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

RUN rosdep update --include-eol-distros

# /opt/ros/${ROS_DISTRO}/share can be changed after rosdep install, so we run it 3 times.
RUN for i in $(seq 3); do rosdep install --rosdistro ${ROS_DISTRO} -r -y -i --from-paths /opt/ros/${ROS_DISTRO}/share ~/ros/${ROS_DISTRO}/src; done

ARG TESTING
RUN if [ ${TESTING} = "true" ] ; then \
      rm -rf ~/ros; \
    else \
      . /opt/ros/${ROS_DISTRO}/setup.sh && \
      cd ~/ros/${ROS_DISTRO} && \
      catkin build; \
    fi
