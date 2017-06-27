#!/usr/bin/env python

import datetime

from jsk_arc2017_common.srv import UpdateJSON
from jsk_arc2017_common.srv import UpdateJSONResponse
import json
import os
import os.path as osp
import rospy
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse
import threading


class JSONSaver(threading.Thread):

    def __init__(self):
        super(JSONSaver, self).__init__(target=self._run_services)
        json_dir = rospy.get_param('~json_dir', None)
        output_dir = rospy.get_param('~output_dir', None)

        if json_dir is None:
            rospy.logerr('must set json dir path to ~json_dir')
            return
        if output_dir is None:
            rospy.logerr('must set output dir path to ~output_dir')
            return
        now = datetime.datetime.now()
        output_dir = osp.join(output_dir, now.strftime('%Y%m%d_%H%M%S'))
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        location_path = osp.join(json_dir, 'item_location_file.json')
        self.output_json_path = osp.join(
            output_dir, 'item_location_file.json')
        with open(location_path) as location_f:
            data = json.load(location_f)
        self.bin_contents = {}
        for bin_ in data['bins']:
            self.bin_contents[bin_['bin_id']] = bin_['contents']
        self.tote_contents = data['tote']['contents']

        self.lock = threading.Lock()
        self.daemon = True

    def _run_services(self):
        self.services = []
        self.services.append(rospy.Service(
            '~update_json', UpdateJSON, self._update))
        self.services.append(rospy.Service(
            '~save_json', Trigger, self._save))

    def _update(self, req):
        is_updated = self._update_location(req)
        is_saved = self._save_json()
        is_updated = is_saved and is_updated
        return UpdateJSONResponse(updated=is_updated)

    def _save(self, req):
        is_saved = self._save_json()
        return TriggerResponse(success=is_saved)

    def _save_json(self):
        separators = (',', ': ')
        self.lock.acquire()
        is_saved = True
        location = {
            'bins': [
                {
                    'bin_id': 'A',
                    'contents': self.bin_contents['A']
                },
                {
                    'bin_id': 'B',
                    'contents': self.bin_contents['B']
                },
                {
                    'bin_id': 'C',
                    'contents': self.bin_contents['C']
                },
            ],
            'boxes': [],
            'tote': {
                'contents': self.tote_contents,
            }
        }
        try:
            with open(self.output_json_path, 'w+') as f:
                json.dump(
                    location, f, sort_keys=True,
                    indent=4, separators=separators)
        except Exception:
            rospy.logerr('could not save json in {}'
                         .format(self.output_json_path))
            is_saved = False
        self.lock.release()
        return is_saved

    def _update_location(self, req):
        item = req.item
        src = req.src
        dst = req.dst
        is_updated = True
        self.lock.acquire()
        if src in 'ABC':
            try:
                self.bin_contents[src].remove(item)
                self.bin_contents[dst].append(item)
            except Exception:
                rospy.logerr('{0} does not exist in bin {1}'.format(item, src))
                is_updated = False
        else:
            try:
                self.tote_contents.remove(item)
                self.bin_contents[dst].append(item)
            except Exception:
                rospy.logerr('{} does not exist in tote'.format(item))
                is_updated = False
        self.lock.release()
        return is_updated


if __name__ == '__main__':
    rospy.init_node('json_saver')
    json_saver = JSONSaver()
    json_saver.start()
    rospy.spin()
