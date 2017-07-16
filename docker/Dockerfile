FROM osrf/ros:indigo-desktop-full

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

RUN cd ~ && \
    rm -rf src/start-jsk/jsk_apc && \
    . /opt/ros/indigo/setup.sh && \
    catkin config --blacklist imagesift jsk_recognition_msgs jsk_perception jsk_pcl_ros_utils jsk_pcl_ros resized_image_transport checkerboard_detector fetcheus naoqieus jsk_fetch_startup jsk_nao_startup roseus_remote jsk_robot_startup jsk_robot_utils jsk_pr2_calibration pr2_base_trajectory_action jsk_baxter_web peppereus naoeus jsk_baxter_desktop jsk_pepper_startup jsk_pr2_startup jsk_pr2_desktop && \
    catkin build
