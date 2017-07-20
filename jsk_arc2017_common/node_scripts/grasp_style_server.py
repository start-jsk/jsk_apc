#!/usr/bin/env python

import jsk_arc2017_common
from jsk_arc2017_common.srv import GetGraspStyle
from jsk_arc2017_common.srv import GetGraspStyleResponse
import rospy


class GraspStyleServer(object):

    style_priority = [
        'suction',
        'pinch'
    ]

    def __init__(self):
        super(GraspStyleServer, self).__init__()
        self.object_names = jsk_arc2017_common.get_object_names()
        self.graspability = jsk_arc2017_common.get_object_graspability()
        self.max_trial = rospy.get_param('~max_trial', 2)
        self.service = rospy.Service(
            '~get_grasp_style', GetGraspStyle, self._get_grasp_style)

    def _get_grasp_style(self, req):
        object_name = req.item
        trial_time = req.trial_time
        res = GetGraspStyleResponse()
        res.success = False
        if object_name not in self.object_names:
            res.message = 'there is no object: {}'.format(object_name)
            return res
        elif trial_time > self.max_trial:
            res.message = 'exceeds max trial time: {}'.format(self.max_trial)
            return res
        elif trial_time < 1:
            res.message = 'invalid trial time: {}'.format(trial_time)
            return res
        graspability = self.graspability[object_name]
        sorted_ability = sorted(
            graspability.items(),
            key=lambda x: self.style_priority.index(x[0]))
        sorted_ability = sorted(sorted_ability, key=lambda x: x[1])
        styles = [x[0] for x in sorted_ability if x[1] < 3]
        if len(styles) < trial_time:
            res.message = 'no more chance to grasp'
            return res
        res.style = styles[trial_time-1]
        res.success = True
        return res

if __name__ == "__main__":
    rospy.init_node('grasp_style_server')
    server = GraspStyleServer()
    rospy.spin()
