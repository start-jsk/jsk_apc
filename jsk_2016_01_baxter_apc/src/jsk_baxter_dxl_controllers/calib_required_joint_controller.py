import math

import rospy
from dynamixel_controllers.joint_position_controller import JointPositionController

class CalibRequiredJointController(JointPositionController):
    def __init__(self, dxl_io, controller_namespace, port_namespace):
        JointPositionController.__init__(self, dxl_io, controller_namespace, port_namespace)

        self.detect_limit_load = rospy.get_param(self.controller_namespace + '/detect_limit_load', 0.15)

    def initialize(self):
        if not JointPositionController.initialize(self):
            return False

        # Initialize joint position
        self.set_speed(0.0)
        prev_limits = self.__get_angle_limits()       # Backup current angle limits
        self.__set_angle_limits(0, 0)                 # Change to wheel mode
        self.__set_speed_wheel(self.joint_speed if self.flipped else -self.joint_speed)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if abs(self.__get_feedback()['load']) > self.detect_limit_load:
                break
            rate.sleep()

        self.__set_angle_limits(prev_limits['min'], prev_limits['max'])  # Change to previous mode
        self.set_speed(self.joint_speed)
        rospy.sleep(1.0)

        # Remember initial joint position
        diff = self.__get_feedback()['position'] - self.initial_position_raw
        self.initial_position_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/init', self.initial_position_raw)
        self.min_angle_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/min', self.min_angle_raw)
        self.max_angle_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/max', self.max_angle_raw)
        if self.flipped:
            self.min_angle = (self.initial_position_raw - self.min_angle_raw) * self.RADIANS_PER_ENCODER_TICK
            self.max_angle = (self.initial_position_raw - self.max_angle_raw) * self.RADIANS_PER_ENCODER_TICK
        else:
            self.min_angle = (self.min_angle_raw - self.initial_position_raw) * self.RADIANS_PER_ENCODER_TICK
            self.max_angle = (self.max_angle_raw - self.initial_position_raw) * self.RADIANS_PER_ENCODER_TICK

        return (not rospy.is_shutdown())

    def __set_angle_limits(self, min_angle, max_angle):
        self.dxl_io.set_angle_limits(self.motor_id, min_angle, max_angle)

    def __get_angle_limits(self):
        return self.dxl_io.get_angle_limits(self.motor_id)

    def __get_feedback(self):
        return self.dxl_io.get_feedback(self.motor_id)

    def __spd_rad_to_raw_wheel(self, spd_rad):
        if spd_rad < -self.joint_max_speed: spd_rad = -self.joint_max_speed
        elif spd_rad > self.joint_max_speed: spd_rad = self.joint_max_speed
        return int(round(spd_rad / self.VELOCITY_PER_TICK))

    def __set_speed_wheel(self, speed):
        mcv = (self.motor_id, self.__spd_rad_to_raw_wheel(speed))
        self.dxl_io.set_multi_speed([mcv])
