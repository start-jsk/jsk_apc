#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import dynamic_reconfigure.client
import rospy
from jsk_recognition_msgs.msg import Int32Stamped


expected_n_cluster = None
reconfig_n_times = 0
lock = threading.Lock()


def cb_kcluster(msg):
    global expected_n_cluster
    with lock:
        expected_n_cluster = msg.data


def cb(msg):
    global expected_n_cluster
    global reconfig_n_limit, reconfig_n_times
    global default_tolerance
    global sub_kcluster, sub_ncluster
    with lock:
        if reconfig_n_times > reconfig_n_limit:
            sub_kcluster.unregister()
            sub_ncluster.unregister()
            return
        if expected_n_cluster is None:
            return
        cfg = client.get_configuration(timeout=None)
        tol_orig = cfg['tolerance']
        delta = tol_orig * reconfig_eps
        if msg.data == expected_n_cluster:
            print('Expected/Actual n_cluster: {0}, Tolerance: {1}'
                .format(msg.data, tol_orig))
            reconfig_n_times = reconfig_n_limit + 1
            return
        elif msg.data > expected_n_cluster:
            cfg['tolerance'] += delta
        else:
            cfg['tolerance'] -= delta
        if cfg['tolerance'] < 0.001:
            print('Invalid tolerance, resetting')
            cfg['tolerance'] = default_tolerance
        print('''\
Expected n_cluster: {0}
Actual   n_cluster: {1}
tolerance: {2} -> {3}
'''.format(expected_n_cluster, msg.data, tol_orig, cfg['tolerance']))
        client.update_configuration(cfg)
        reconfig_n_times += 1


if __name__ == '__main__':
    rospy.init_node('euclid_k_clutering')
    reconfig_eps = rospy.get_param('~reconfig_eps', 0.2)
    reconfig_n_limit = rospy.get_param('~reconfig_n_limit', 10)
    node_name = rospy.get_param('~node')
    default_tolerance = rospy.get_param('~default_tolerance')
    client = dynamic_reconfigure.client.Client(node_name)
    sub_kcluster = rospy.Subscriber('~k_cluster', Int32Stamped, cb_kcluster)
    sub_ncluster = rospy.Subscriber(
        '{node}/cluster_num'.format(node=node_name), Int32Stamped, cb)
    cfg = {'tolerance': default_tolerance}
    client.update_configuration(cfg)
    rospy.spin()
