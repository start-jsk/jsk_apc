#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import rospy

from work_order import get_sorted_work_order


def main():
    json_file = rospy.get_param('~json', None)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    work_order = get_sorted_work_order(json_file=json_file)

    flag = {'left': False, 'right': False}
    for bin_, _ in work_order:
        if sum(flag.values()) == len(flag):
            break
        if bin_ in 'cfil':
            rospy.set_param('right_process/target', bin_)
            flag['right'] = True
        else:
            rospy.set_param('left_process/target', bin_)
            flag['left'] = True


if __name__ == '__main__':
    rospy.init_node('setup_target')
    main()
    quit()
