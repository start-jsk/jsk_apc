FROM osrf/ros:indigo-desktop-trusty

RUN echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list

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
    wstool up -j 2
RUN rosdep update
# /opt/ros/indigo/share can be changed after rosdep install, so we run it 3 times.
RUN for i in $(seq 3); do rosdep install --rosdistro indigo -r -y -i --from-paths /opt/ros/indigo/share ~/src; done
RUN rm -rf ~/src
