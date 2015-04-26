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

    # we should start from left limb
    for bin_, _ in work_order:
        if bin_ in 'cfil':
            continue  # skip bins for right limb
        rospy.set_param('/target', bin_)
        return


if __name__ == '__main__':
    rospy.init_node('setup_target')
    main()
    quit()
