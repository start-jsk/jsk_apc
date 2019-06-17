#!/usr/bin/env python

import os.path as osp

import six

import rospy
from std_srvs.srv import Trigger

from dynamic_reconfigure.client import Client


def main():
    # server_name = rospy.resolve_name('~data_collection_server')
    # root_dir = osp.expanduser(rospy.get_param('~root_dir'))
    server_name = '/data_collection_server'
    root_dir = osp.expanduser('~/.ros/instance_occlsegm')

    caller = rospy.ServiceProxy('%s/save_request' % server_name, Trigger)
    client = Client(server_name)

    action = 'n'  # to initialize
    while not rospy.is_shutdown():
        while action not in ['n', 's', 'q']:
            action = six.input(
                'Please select actions: [n] New save_dir, [s] Save, [q] Quit: '
            )
            action = action.strip().lower()
        if action == 'n':
            now = rospy.Time.now()
            save_dir = osp.join(root_dir, str(now.to_nsec()))
            client.update_configuration({'save_dir': save_dir})
            print('-' * 79)
            print('Updated save_dir to: {}'.format(save_dir))
        elif action == 's':
            res = caller.call()
            print('Sent request to data_collection_server')
            print(res)
        else:
            assert action == 'q'
            return
        action = None


if __name__ == '__main__':
    rospy.init_node('data_collection_client')
    main()
