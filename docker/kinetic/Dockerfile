FROM osrf/ros:kinetic-desktop-xenial

ENV ROS_DISTRO=kinetic

RUN echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# FIXME: https://github.com/start-jsk/jsk_apc/pull/2664
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y \
    wget \
    git \
    python-catkin-tools \
    python-rosdep \
    python-setuptools \
    python-wstool

RUN easy_install -U pip && \
    pip install 'pip<10' && \
    pip install -U setuptools

RUN cd ~ && \
    mkdir -p src && \
    cd src && \
    wstool init && \
    wstool set start-jsk/jsk_apc https://github.com/start-jsk/jsk_apc.git -v master --git -y && \
    wstool up -j 2 && \
    wstool merge start-jsk/jsk_apc/.travis.rosinstall && \
    wstool merge start-jsk/jsk_apc/.travis.rosinstall.$ROS_DISTRO && \
    wstool up -j 2
RUN rosdep update --include-eol-distros
# /opt/ros/$ROS_DISTRO/share can be changed after rosdep install, so we run it 3 times.
RUN for i in $(seq 3); do rosdep install --rosdistro $ROS_DISTRO -r -y -i --from-paths /opt/ros/$ROS_DISTRO/share ~/src; done
RUN rm -rf ~/src
