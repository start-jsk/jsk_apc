import rospy
from dynamixel_controllers.joint_position_controller import (
    JointPositionController,
)


class CalibRequiredJointController(JointPositionController):
    def __init__(self, dxl_io, controller_namespace, port_namespace):
        JointPositionController.__init__(self, dxl_io, controller_namespace,
                                         port_namespace)

        self.calib_speed = rospy.get_param(
                self.controller_namespace + '/calib_speed',
                0.1)
        self.calib_torque_limit = rospy.get_param(
                self.controller_namespace + '/calib_torque_limit',
                0.3)
        self.detect_limit_load = rospy.get_param(
                self.controller_namespace + '/detect_limit_load',
                0.15)

    def initialize(self):
        if not JointPositionController.initialize(self):
            return False

        # Initialize joint position

        self.set_speed(0.0)
        # Backup current angle limits
        prev_limits = self.__get_angle_limits()
        # Change to wheel mode
        self.__set_angle_limits(0, 0)
        self.set_torque_limit(self.calib_torque_limit)
        if self.flipped:
            self.__set_speed_wheel(self.calib_speed)
        else:
            self.__set_speed_wheel(-self.calib_speed)
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            init_pos = self.__get_feedback()['position']
            if abs(self.__get_feedback()['load']) > self.detect_limit_load:
                break
            rate.sleep()
        self.__set_speed_wheel(0.0)
        # Change to previous mode
        self.__set_angle_limits(prev_limits['min'], prev_limits['max'])
        self.set_speed(self.joint_speed)
        if self.torque_limit is not None:
            self.set_torque_limit(self.torque_limit)

        # Remember initial joint position
        diff = init_pos - self.initial_position_raw
        self.initial_position_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/init',
                        self.initial_position_raw)
        self.min_angle_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/min',
                        self.min_angle_raw)
        self.max_angle_raw += diff
        rospy.set_param(self.controller_namespace + '/motor/max',
                        self.max_angle_raw)
        if self.flipped:
            self.min_angle = ((self.initial_position_raw -
                               self.min_angle_raw) *
                              self.RADIANS_PER_ENCODER_TICK)
            self.max_angle = ((self.initial_position_raw -
                               self.max_angle_raw) *
                              self.RADIANS_PER_ENCODER_TICK)
        else:
            self.min_angle = ((self.min_angle_raw -
                               self.initial_position_raw) *
                              self.RADIANS_PER_ENCODER_TICK)
            self.max_angle = ((self.max_angle_raw -
                               self.initial_position_raw) *
                              self.RADIANS_PER_ENCODER_TICK)

        return (not rospy.is_shutdown())

    def __set_angle_limits(self, min_angle, max_angle):
        self.dxl_io.set_angle_limits(self.motor_id, min_angle, max_angle)

    def __get_angle_limits(self):
        return self.dxl_io.get_angle_limits(self.motor_id)

    def __get_feedback(self):
        return self.dxl_io.get_feedback(self.motor_id)

    def __spd_rad_to_raw_wheel(self, spd_rad):
        if spd_rad < -self.joint_max_speed:
            spd_rad = -self.joint_max_speed
        elif spd_rad > self.joint_max_speed:
            spd_rad = self.joint_max_speed
        return int(round(spd_rad / self.VELOCITY_PER_TICK))

    def __set_speed_wheel(self, speed):
        mcv = (self.motor_id, self.__spd_rad_to_raw_wheel(speed))
        self.dxl_io.set_multi_speed([mcv])
