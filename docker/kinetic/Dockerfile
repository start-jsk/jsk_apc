FROM osrf/ros:kinetic-desktop-xenial

RUN echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y \
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
    wstool merge start-jsk/jsk_apc/.travis.rosinstall.kinetic && \
    wstool up -j 2
RUN rosdep update
RUN rosdep install --rosdistro kinetic -r -y -i --from-paths /opt/ros/kinetic/share ~/src
RUN rm -rf ~/src
