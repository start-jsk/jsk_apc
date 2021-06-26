#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import rospy

from work_order import get_sorted_work_order
from jsk_rqt_plugins.srv import YesNo


def main():
    json_file = rospy.get_param('~json', None)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    work_order = get_sorted_work_order(json_file=json_file)

    first_orders = dict(left='', right='')
    for bin_, _ in work_order:
        if bin_ in 'abdeghjk' and first_orders['left'] == '':
            first_orders['left'] = bin_
        elif bin_ in 'cfil' and first_orders['right'] == '':
            first_orders['right'] = bin_
        elif first_orders['left'] == '' and first_orders['right'] == '':
            break

    rospy.set_param('/left_process/target', first_orders['left'])
    rospy.set_param('/right_process/target', first_orders['right'])
    rospy.set_param('/left_process/state', 'wait_for_user_input')
    rospy.set_param('/right_process/state', 'wait_for_user_input')

    rospy.set_param('/total_score', 0)

    # wait for rqt_yn_btn
    wait_for_rqt_yn_btn = True
    while wait_for_rqt_yn_btn and (not rospy.is_shutdown()):
        try:
            rospy.wait_for_service('/rqt_yn_btn', timeout=10)
            wait_for_rqt_yn_btn = False
        except rospy.ROSException as e:
            rospy.logerr('timeout for /rqt_yn_btn: {0}'.format(e))
        rospy.sleep(1)
    # get user input using rqt_yn_btn
    cl_yn = rospy.ServiceProxy('/rqt_yn_btn', YesNo)
    start_main = False
    while start_main is False:
        res = cl_yn()
        start_main = res.yes
        if res.yes is False:
            rospy.logerr('please select yes to start')
        rospy.sleep(1)
    rospy.set_param('/left_process/state', 'pick_object')
    rospy.set_param('/right_process/state', 'pick_object')


if __name__ == '__main__':
    rospy.init_node('setup_target')
    main()
