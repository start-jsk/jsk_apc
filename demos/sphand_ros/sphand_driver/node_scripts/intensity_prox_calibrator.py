#!/usr/bin/env python

import numpy as np

from sphand_driver_msgs.msg import IntensityProxCalibInfo
from sphand_driver_msgs.msg import IntensityProxCalibInfoArray
from sphand_driver_msgs.msg import ProximityStampedArray
from vl53l0x_mraa_ros.msg import RangingMeasurementDataStampedArray
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse
import rospy


class IntensityProxCalibrator(object):

    def __init__(self):
        self.i_refl_param = rospy.get_param('~i_reflectance_param', None)
        if self.i_refl_param is not None:
            self.i_refl_param = np.array(self.i_refl_param)
        self.i_init_value = rospy.get_param('~i_init_value', None)
        if self.i_init_value is not None:
            self.i_init_value = np.array(self.i_init_value)
        self.i_valid_min = rospy.get_param('~i_valid_min', 50)
        self.i_valid_max = rospy.get_param('~i_valid_max', 1000)
        self.i_valid_max_dist = rospy.get_param('~i_valid_max_dist', 60)
        self.i_height_from_tof = rospy.get_param('~i_height_from_tof', None)
        if self.i_height_from_tof is not None:
            self.i_height_from_tof = np.array(self.i_height_from_tof)
        self.i_queue_size_for_tof = rospy.get_param('~i_queue_size_for_tof', 2)
        self.tof_valid_min = rospy.get_param('~tof_valid_min', 40)
        self.tof_delay_from_i = rospy.get_param('~tof_delay_from_i', 0.0)
        self.tof_tm_tolerance = rospy.get_param('~tof_tm_tolerance', 0.02)
        self.use_i_average = rospy.get_param('~use_i_average', False)
        self.rubber_t = rospy.get_param('~rubber_thickness', None)
        if self.rubber_t is not None:
            self.rubber_t = np.array(self.rubber_t)
        self.i_raw = None
        self.i_diff_from_init = None
        self.tof_dist = None
        self.i_diff_queue = []
        self.i_tms_queue = []

        self.pub_i_calib = rospy.Publisher(
            '~output', IntensityProxCalibInfoArray,
            queue_size=1)
        self.sub_input_i = rospy.Subscriber(
            '~input/intensity', ProximityStampedArray, self._intensity_cb)
        self.sub_input_tof = rospy.Subscriber(
            '~input/tof', RangingMeasurementDataStampedArray, self._tof_cb)
        self.init_srv = rospy.Service(
            '~set_init_proximities', Trigger, self._set_init_proximities)
        self.reset_refl_param_srv = rospy.Service(
            '~reset_reflectance_param', Trigger, self._reset_refl_param)

    def _intensity_cb(self, msg):
        if self.use_i_average:
            self.i_raw = np.array([p.proximity.average
                                   for p in msg.proximities])
        else:
            self.i_raw = np.array([p.proximity.proximity
                                   for p in msg.proximities])
        if self.i_init_value is None:
            rospy.logwarn_throttle(10, 'Init prox is not set, so skipping')
            return
        assert self.i_raw.shape == self.i_init_value.shape
        self.i_diff_from_init = self.i_raw - self.i_init_value
        self.i_diff_queue.append(self.i_diff_from_init)
        self.i_tms_queue.append([p.header.stamp for p in msg.proximities])
        while len(self.i_diff_queue) > self.i_queue_size_for_tof:
            self.i_diff_queue.pop(0)
            self.i_tms_queue.pop(0)
        if self.i_refl_param is None:
            rospy.logwarn_throttle(10, 'Refl. param is not set, so skipping')
            return
        assert self.i_diff_from_init.shape == self.i_refl_param.shape
        diff_plus = self.i_diff_from_init.copy()
        diff_plus = diff_plus.astype(np.float64)
        diff_plus[diff_plus <= 0] = np.inf
        distance = np.sqrt(self.i_refl_param / diff_plus)
        distance[distance == 0] = np.inf

        if self.rubber_t is not None:
            # If distance is under thickness
            init_refl = self.i_init_value * (self.rubber_t ** 2)
            distance = np.where(
                distance < self.rubber_t,
                np.sqrt((self.i_refl_param + init_refl) / self.i_raw),
                distance
            )

        # Create distance combined with ToF output
        tof_d_from_i = self.tof_dist - self.i_height_from_tof
        dist_combined = np.where(
            ((distance == np.inf) |
             ((tof_d_from_i > self.i_valid_max_dist) &
              (self.i_diff_from_init <
               ((self.i_valid_min + self.i_valid_max) / 2.0)))),
            tof_d_from_i,
            distance
        )

        # Publish calibrated info
        pub_msg = IntensityProxCalibInfoArray()
        pub_msg.header.stamp = msg.header.stamp
        for p, dist, diff, refl, init, dist_c in zip(msg.proximities,
                                                     distance,
                                                     self.i_diff_from_init,
                                                     self.i_refl_param,
                                                     self.i_init_value,
                                                     dist_combined):
            info = IntensityProxCalibInfo()
            info.header.stamp = p.header.stamp
            info.distance = dist
            info.diff_from_init = diff
            info.reflectance_param = refl
            info.init_value = init
            info.distance_combined = dist_c
            pub_msg.data.append(info)
        self.pub_i_calib.publish(pub_msg)

    def _tof_cb(self, msg):
        self.tof_dist = np.array([e.data.range_millimeter for e in msg.array],
                                 dtype=np.float64)
        if self.i_diff_from_init is None:
            rospy.logwarn_throttle(
                10, 'Prox diff_from_init is not set, so skipping')
            return
        assert len(self.i_diff_from_init) == len(msg.array)
        if self.i_height_from_tof is None:
            self.i_height_from_tof = np.zeros(len(msg.array))
        if self.i_refl_param is None:
            self.i_refl_param = np.zeros(len(msg.array))
        for i, data_st in enumerate(msg.array):
            for i_diff, i_tms in zip(self.i_diff_queue, self.i_tms_queue):
                if abs((data_st.header.stamp - i_tms[i]).to_sec() -
                       self.tof_delay_from_i) > \
                   self.tof_tm_tolerance:
                    continue
                if i_diff[i] < self.i_valid_min:
                    continue
                if i_diff[i] > self.i_valid_max:
                    continue
                tof_r = data_st.data.range_millimeter
                if tof_r < self.tof_valid_min:
                    continue
                if tof_r > self.i_valid_max_dist + self.i_height_from_tof[i]:
                    continue
                self.i_refl_param[i] = \
                    i_diff[i] * ((tof_r - self.i_height_from_tof[i]) ** 2)

    def _set_init_proximities(self, req):
        is_success = True
        if self.i_raw is not None:
            self.i_init_value = self.i_raw.copy()
        else:
            is_success = False
        return TriggerResponse(success=is_success)

    def _reset_refl_param(self, req):
        is_success = True
        if self.i_raw is not None:
            self.i_refl_param = np.zeros(len(self.i_raw))
        else:
            is_success = False
        return TriggerResponse(success=is_success)


if __name__ == '__main__':
    rospy.init_node('intensity_prox_calibrator')
    app = IntensityProxCalibrator()
    rospy.spin()
