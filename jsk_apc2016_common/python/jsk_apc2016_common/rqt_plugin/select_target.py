#!/usr/bin/env python
#
import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QDialog


class SelectTargetWidget(QDialog):
    def __init__(self):
        super(SelectTargetWidget, self).__init__()
        rp = rospkg.RosPack()
        ui_file = os.path.join(
                rp.get_path('jsk_apc2016_common'),
                'resource',
                'select_target.ui'
                )
        loadUi(ui_file, self)

        work_order_list = rospy.get_param("/work_order")
        self.work_order = {}
        for order in work_order_list:
            self.work_order[order['bin']] = order['item']
        self.init_work_order = self.work_order.copy()
        self.bin_contents = {}
        for bin_, items in rospy.get_param('/bin_contents').items():
            bin_ = bin_.split('_')[1].lower()
            self.bin_contents[bin_] = items
        self.bin_dict = {
                'a': self.bin_A,
                'b': self.bin_B,
                'c': self.bin_C,
                'd': self.bin_D,
                'e': self.bin_E,
                'f': self.bin_F,
                'g': self.bin_G,
                'h': self.bin_H,
                'i': self.bin_I,
                'j': self.bin_J,
                'k': self.bin_K,
                'l': self.bin_L
                }

        self.setObjectName('SelectTargetUI')

        for bin_ in 'abcdefghijkl':
            self.bin_dict[bin_].addItems(self.bin_contents[bin_])
            self.bin_dict[bin_].currentIndexChanged.connect(
                    self._select_target(bin_)
                    )
        self.update.accepted.connect(self._update_param)
        self.update.rejected.connect(self._reset_param)
        self._get_init_index()
        self._set_init_index()

    def _select_target(self, bin_):
        def _select_target_curried(index):
            self.work_order[bin_] = self.bin_contents[bin_][index]
        return _select_target_curried

    def _get_init_index(self):
        self.init_index = {}
        for bin_ in 'abcdefghijkl':
            self.init_index[bin_] = self.bin_contents[bin_].index(
                    self.init_work_order[bin_]
                    )

    def _set_init_index(self):
        for bin_ in 'abcdefghijkl':
            self.bin_dict[bin_].setCurrentIndex(self.init_index[bin_])

    def _update_param(self):
        self._set_param(self.work_order)
        self.show()

    def _reset_param(self):
        self._set_param(self.init_work_order)
        self._set_init_index()
        self.show()

    def _set_param(self, work_order):
        work_order_list = []
        for bin_, item in work_order.items():
            work_order_list.append({'bin': bin_, 'item': item})
        rospy.set_param('/work_order', work_order_list)


class SelectTarget(Plugin):
    def __init__(self, context):
        super(SelectTarget, self).__init__(context)
        self.setObjectName('SelectTarget')
        self._widget = SelectTargetWidget()
        context.add_widget(self._widget)
