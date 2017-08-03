#!/usr/bin/env python

import datetime

from jsk_arc2017_common.msg import Content
from jsk_arc2017_common.msg import ContentArray
from jsk_arc2017_common.srv import UpdateJSON
from jsk_arc2017_common.srv import UpdateJSONResponse
import json
import os
import os.path as osp
import rospy
import shutil
from std_msgs.msg import String
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
        if osp.exists(location_path):
            shutil.copy(location_path, self.output_json_path)
            with open(location_path) as location_f:
                data = json.load(location_f)
        else:
            rospy.logerr(
                'item_location_file.json does not exists in {}', location_path)
        self.bin_contents = {}
        for bin_ in data['bins']:
            self.bin_contents[bin_['bin_id']] = bin_['contents']
        self.tote_contents = data['tote']['contents']

        self.cardboard_contents = {}
        self.cardboard_ids = {}

        # this is for pick task
        # order file is only used in pick task
        order_path = osp.join(json_dir, 'order_file.json')
        if osp.exists(order_path):
            output_order_path = osp.join(output_dir, 'order_file.json')
            shutil.copy(order_path, output_order_path)

            order_path = osp.join(json_dir, 'order_file.json')
            with open(order_path) as order_f:
                orders = json.load(order_f)['orders']

            for order in orders:
                size_id = order['size_id']
                if len(order['contents']) == 2:
                    cardboard_id = 'A'
                elif len(order['contents']) == 3:
                    cardboard_id = 'B'
                else:  # len(order['contents']) == 5
                    cardboard_id = 'C'
                self.cardboard_ids[cardboard_id] = size_id

            cardboard_contents = {}
            for box in data['boxes']:
                size_id = box['size_id']
                cardboard_contents[size_id] = box['contents']
            for key in 'ABC':
                size_id = self.cardboard_ids[key]
                self.cardboard_contents[key] = cardboard_contents[size_id]

        # publish stamped json_dir
        self.pub = rospy.Publisher('~output/json_dir', String, queue_size=1)
        self.pub_bin = rospy.Publisher(
            '~output/bin_contents',
            ContentArray,
            queue_size=1)
        rate = rospy.get_param('~rate', 1)
        self.timer_pub = rospy.Timer(rospy.Duration(1. / rate), self._cb_pub)

        self.lock = threading.Lock()
        self.daemon = True

    def _cb_pub(self, event):
        self.pub.publish(String(data=osp.dirname(self.output_json_path)))
        contents_msg = ContentArray()
        contents = []
        for bin_ in 'ABC':
            msg = Content()
            msg.bin = bin_
            msg.items = self.bin_contents[bin_]
            contents.append(msg)
        contents_msg.header.stamp = rospy.Time.now()
        contents_msg.contents = contents
        self.pub_bin.publish(contents_msg)

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
        boxes = []
        if len(self.cardboard_contents.keys()) > 0:
            for key in 'ABC':
                boxes.append({
                    'size_id': self.cardboard_ids[key],
                    'contents': self.cardboard_contents[key]
                })
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
            'boxes': boxes,
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
        is_updated = True
        self.lock.acquire()
        item = req.item
        src = req.src
        dst = req.dst
        if src[:3] == 'bin':
            src = src[4]
            try:
                self.bin_contents[src].remove(item)
            except Exception:
                rospy.logerr('{0} does not exist in bin {1}'.format(item, src))
                self.lock.release()
                return False
        elif src[:9] == 'cardboard':
            src = src[10]
            try:
                self.cardboard_contents[src].remove(item)
            except Exception:
                rospy.logerr('{0} does not exist in bin {1}'.format(item, src))
                self.lock.release()
                return False
        elif src == 'tote':
            try:
                self.tote_contents.remove(item)
            except Exception:
                rospy.logerr('{} does not exist in tote'.format(item))
                self.lock.release()
                return False
        else:
            rospy.logerr('Invalid src request {}', src)
            is_updated = False

        if dst[:3] == 'bin':
            dst = dst[4]
            self.bin_contents[dst].append(item)
        elif dst[:9] == 'cardboard':
            dst = dst[10]
            self.cardboard_contents[dst].append(item)
        elif dst == 'tote':
            self.tote_contents.append(item)
        else:
            rospy.logerr('Invalid dst request {}', dst)
            is_updated = False

        self.lock.release()
        return is_updated


if __name__ == '__main__':
    rospy.init_node('json_saver')
    json_saver = JSONSaver()
    json_saver.start()
    rospy.spin()
