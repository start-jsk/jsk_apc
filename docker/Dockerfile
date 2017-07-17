FROM osrf/ros:indigo-desktop-full

RUN echo "deb http://packages.ros.org/ros-shadow-fixed/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y \
    wget \
    git \
    python-catkin-tools \
    python-rosdep \
    python-setuptools \
    python-wstool

RUN easy_install -U pip && \
    pip install -U pip setuptools

RUN cd ~ && \
    mkdir -p src && \
    cd src && \
    wstool init && \
    wstool set start-jsk/jsk_apc https://github.com/start-jsk/jsk_apc.git -v master --git -y && \
    wstool up -j 2 && \
    wstool merge start-jsk/jsk_apc/.travis.rosinstall && \
    wstool up -j 2

RUN rosdep update
RUN rosdep install --from-path ~/src/jsk-ros-pkg/jsk_recognition -r -y -i
RUN rosdep install --from-path ~/src/jsk-ros-pkg/jsk_robot/jsk_baxter_robot -r -y -i --skip-keys=ros3djs --skip-keys=ros2djs --skip-keys=roslibjs
RUN rosdep install --from-path ~/src/FA-I-sensor/force_proximity_ros -r -y -i
RUN rosdep install --from-path ~/src/start-jsk/jsk_apc -r -y -i

RUN rm -rf ~/src
