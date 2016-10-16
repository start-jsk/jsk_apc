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
        self.init_work_order = self.work_order
        self.bin_contents = rospy.get_param('/bin_contents')
        self.bin_dict = {
                'bin_A': self.bin_A,
                'bin_B': self.bin_B,
                'bin_C': self.bin_C,
                'bin_D': self.bin_D,
                'bin_E': self.bin_E,
                'bin_F': self.bin_F,
                'bin_G': self.bin_G,
                'bin_H': self.bin_H,
                'bin_I': self.bin_I,
                'bin_J': self.bin_J,
                'bin_K': self.bin_K,
                'bin_L': self.bin_L
                }

        self.setObjectName('SelectTargetUI')

        for bin_ in 'abcdefghijkl':
            bin_ = 'bin_' + bin_.upper()
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
            bin_ = 'bin_' + bin_.upper()
            self.init_index[bin_] = self.bin_contents[bin_].index(
                    self.init_work_order[bin_]
                    )

    def _set_init_index(self):
        for bin_ in 'abcdefghijkl':
            bin_ = 'bin_' + bin_.upper()
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
