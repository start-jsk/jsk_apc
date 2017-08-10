#!/usr/bin/env python
# -*- coding:utf-8 -*-

from jsk_arc2017_baxter.srv import CheckCanStart
from jsk_arc2017_baxter.srv import CheckCanStartResponse
from jsk_arc2017_baxter.srv import GetState
from jsk_arc2017_baxter.srv import GetStateResponse
from jsk_arc2017_baxter.srv import UpdateState
from jsk_arc2017_baxter.srv import UpdateStateResponse
import rospy
import threading


class StateServer(threading.Thread):
    def __init__(self):
        super(StateServer, self).__init__(target=self._run_services)
        self.is_pick = rospy.get_param('~is_pick', True)
        self.state = {
            'right': 'init',
            'left': 'init'
        }
        self.lock = threading.Lock()
        self.daemon = True

    def _run_services(self):
        self.services = []
        self.services.append(
            rospy.Service(
                '~left_hand/update_state',
                UpdateState,
                self._update_left_state))
        self.services.append(
            rospy.Service(
                '~right_hand/update_state',
                UpdateState,
                self._update_right_state))
        self.services.append(
            rospy.Service(
                '~left_hand/get_state',
                GetState,
                self._get_left_state))
        self.services.append(
            rospy.Service(
                '~right_hand/get_state',
                GetState,
                self._get_right_state))
        self.services.append(
            rospy.Service(
                '~left_hand/check_can_start',
                CheckCanStart,
                self._check_left_can_start))
        self.services.append(
            rospy.Service(
                '~right_hand/check_can_start',
                CheckCanStart,
                self._check_right_can_start))

    def _check_left_can_start(self, req):
        can_start = self._check_can_start(
            'left', req.start_state, req.wait_state)
        return CheckCanStartResponse(can_start=can_start)

    def _check_right_can_start(self, req):
        can_start = self._check_can_start(
            'right', req.start_state, req.wait_state)
        return CheckCanStartResponse(can_start=can_start)

    def _check_can_start(self, hand, start_state, wait_state):
        self.lock.acquire()
        if hand == 'left':
            opposite_hand = 'right'
        else:
            opposite_hand = 'left'
        state = self.state[hand]
        opposite_state = self.state[opposite_hand]

        can_start = self._get_can_start(state, opposite_state)
        if can_start:
            self.state[hand] = start_state
        else:
            self.state[hand] = wait_state
        self.lock.release()
        return can_start

    def _get_can_start(self, state, opposite_state):
        if self.is_pick:
            # pick task: wait condition
            if state == 'wait-for-opposite-arm-start-picking':
                if opposite_state == 'init' \
                        or opposite_state == 'set-target' \
                        or opposite_state == 'wait-for-user-input' \
                        or opposite_state == 'wait-for-opposite-arm-start-picking':  # NOQA
                    can_start = True
                else:
                    can_start = False
            elif state == 'wait-for-opposite-arm':
                if opposite_state == 'set-target' \
                        or opposite_state == 'recognize-object' \
                        or opposite_state == 'check-can-start-picking' \
                        or opposite_state == 'check-start-picking-fail-count' \
                        or opposite_state == 'pick-object' \
                        or opposite_state == 'change-grasp-style' \
                        or opposite_state == 'verify-object' \
                        or opposite_state == 'set-target-cardboard' \
                        or opposite_state == 'place-object' \
                        or opposite_state == 'return-object':
                    can_start = False
                else:
                    can_start = True
            else:
                can_start = True
        else:
            # stow task: wait condition
            if state == 'wait-for-opposite-arm-start-picking':
                if opposite_state == 'init' \
                        or opposite_state == 'recognize-object' \
                        or opposite_state == 'wait-for-user-input' \
                        or opposite_state == 'wait-for-opposite-arm-start-picking':  # NOQA
                    can_start = True
                else:
                    can_start = False
            elif state == 'wait-for-opposite-arm':
                if opposite_state == 'recognize-object' \
                        or opposite_state == 'check-can-start-picking' \
                        or opposite_state == 'check-start-picking-fail-count' \
                        or opposite_state == 'pick-object' \
                        or opposite_state == 'change-grasp-style' \
                        or opposite_state == 'verify-object' \
                        or opposite_state == 'set-target-bin' \
                        or opposite_state == 'return-object':
                    can_start = False
                else:
                    can_start = True
            else:
                can_start = True
        return can_start

    def _update_left_state(self, req):
        is_updated = self._update_state(req.state, 'left')
        return UpdateStateResponse(updated=is_updated)

    def _update_right_state(self, req):
        is_updated = self._update_state(req.state, 'right')
        return UpdateStateResponse(updated=is_updated)

    def _update_state(self, state, hand):
        self.lock.acquire()
        is_updated = True
        try:
            self.state[hand] = state
        except Exception:
            is_updated = False
        self.lock.release()
        return is_updated

    def _get_left_state(self, req):
        self.lock.acquire()
        state = self.state['left']
        self.lock.release()
        return GetStateResponse(state=state)

    def _get_right_state(self, req):
        self.lock.acquire()
        state = self.state['right']
        self.lock.release()
        return GetStateResponse(state=state)


if __name__ == "__main__":
    rospy.init_node('state_server')
    state_server = StateServer()
    state_server.start()
    rospy.spin()
