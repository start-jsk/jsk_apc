#!/usr/bin/env python

import serial

from jsk_arc2017_common.msg import WeightStamped
import rospy


class EkEwIDriver(object):

    """Read data from EK-i/EW-i scale.

    Data Sheet: https://www.aandd.co.jp/adhome/pdf/manual/balance/ekew-i.pdf
    """

    def __init__(self):
        super(EkEwIDriver, self).__init__()
        port = rospy.get_param('~port', '/dev/ttyUSB0')
        timeout = rospy.get_param('~timeout', None)
        rospy.loginfo('port=%s', port)
        # EK-i/EW-i series default settings
        self.ser = serial.Serial(
            port, baudrate=2400, bytesize=7, parity=serial.PARITY_EVEN,
            timeout=timeout, writeTimeout=timeout)
        self.pub = rospy.Publisher('~output', WeightStamped, queue_size=1)
        self.pub_raw = rospy.Publisher('~output/weight_raw', WeightStamped, queue_size=1)
        rate = rospy.get_param('~rate', 10)
        self.read_timer = rospy.Timer(rospy.Duration(1. / rate),
                                      self._read_timer_cb)

    def _read_timer_cb(self, event):
        if (self.pub.get_num_connections() == 0) and (self.pub_raw.get_num_connections() == 0):
            return

        try:
            self.ser.write('Q\r\n')
        except SerialTimeoutException:
            rospy.logerr('Serial write timeout')
            rospy.signal_shutdown('Serial write timeout')
            return
        data = self.ser.read(17)
        if len(data) != 17:
            rospy.logerr('Serial read timeout')
            rospy.signal_shutdown('Serial read timeout')
            return
    
        # get raw scale value without checking the mode of the scale
        weight_raw = -1 # unknown
        unit = data[12:15]
        if unit != '  g':
            rospy.logerr('Unsupported unit: %s', unit)
        else:
            weight_raw = float(data[3:12])
        msg = WeightStamped()
        msg.header.stamp = event.current_real
        msg.weight.value = weight_raw
        self.pub_raw.publish(msg)

        # get scale value with checking the mode of the scale
        header = data[:2]
        weight = -1  # unknown
        if header == 'ST':
            # scale mode
            unit = data[12:15]
            if unit != '  g':
                rospy.logerr('Unsupported unit: %s', unit)
            else:
                weight = float(data[3:12])
        elif header == 'QT':
            # number mode
            rospy.logerr('Unsupported mode: %s', header)
        elif header == 'US':
            # unstable
            pass
        elif header == 'OL':
            # scale over
            rospy.logerr('Scale over')

        msg.weight.value = weight
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('ekew_i_driver')
    EkEwIDriver()
    rospy.spin()
