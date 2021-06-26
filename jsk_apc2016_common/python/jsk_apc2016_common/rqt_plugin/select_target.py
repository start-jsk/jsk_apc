#!/usr/bin/env python
#
import os
import rospy
import rospkg

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QDialog, QPixmap
from jsk_apc2016_common.srv import UpdateTarget


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
        self.image_path = os.path.join(
                rp.get_path('jsk_apc2016_common'),
                'models',
                )
        work_order_list = rospy.get_param("~work_order")
        self.work_order = {}
        for order in work_order_list:
            self.work_order[order['bin']] = order['item']
        self.init_work_order = self.work_order.copy()
        self.prev_work_order = self.work_order.copy()
        self.bin_contents = {}
        for bin_, items in rospy.get_param('~bin_contents').items():
            bin_ = bin_.split('_')[1].lower()
            self.bin_contents[bin_] = items
            # for user to have robot stop going that bin
            self.bin_contents[bin_].append('<no target>')
        self.bin_dict = {
                'a': {'combo': self.bin_A, 'image': self.bin_A_image},
                'b': {'combo': self.bin_B, 'image': self.bin_B_image},
                'c': {'combo': self.bin_C, 'image': self.bin_C_image},
                'd': {'combo': self.bin_D, 'image': self.bin_D_image},
                'e': {'combo': self.bin_E, 'image': self.bin_E_image},
                'f': {'combo': self.bin_F, 'image': self.bin_F_image},
                'g': {'combo': self.bin_G, 'image': self.bin_G_image},
                'h': {'combo': self.bin_H, 'image': self.bin_H_image},
                'i': {'combo': self.bin_I, 'image': self.bin_I_image},
                'j': {'combo': self.bin_J, 'image': self.bin_J_image},
                'k': {'combo': self.bin_K, 'image': self.bin_K_image},
                'l': {'combo': self.bin_L, 'image': self.bin_L_image}
                }
        self.setObjectName('SelectTargetUI')

        for bin_ in 'abcdefghijkl':
            self.bin_dict[bin_]['combo'].addItems(self.bin_contents[bin_])
            self.bin_dict[bin_]['combo'].currentIndexChanged.connect(
                    self._select_target(bin_)
                    )
        self.update.accepted.connect(self._update_param)
        self.update.rejected.connect(self._reset_param)
        self._set_init_target()

    def _select_target(self, bin_):
        def _select_target_curried(index):
            self.work_order[bin_] = self.bin_contents[bin_][index]
            self._show_image(bin_, index)
        return _select_target_curried

    def _set_init_target(self):
        for bin_ in 'abcdefghijkl':
            init_index = self.bin_contents[bin_].index(
                    self.init_work_order[bin_]
                    )
            self.bin_dict[bin_]['combo'].setCurrentIndex(init_index)
            self._show_image(bin_, init_index)

    def _show_image(self, bin_, index):
        target_name = self.bin_contents[bin_][index]
        target_image_path = os.path.join(
                self.image_path,
                target_name,
                'image.jpg'
                )
        target_pixmap = QPixmap(target_image_path).scaled(100, 100)
        self.bin_dict[bin_]['image'].setPixmap(target_pixmap)
        self.bin_dict[bin_]['image'].show()

    def _update_param(self):
        self._set_param(self.work_order)
        self.show()

    def _reset_param(self):
        self._set_param(self.init_work_order)
        self._set_init_target()
        self.show()

    def _set_param(self, work_order):
        for bin_ in 'abcdefghijkl':
            if work_order[bin_] != self.prev_work_order[bin_]:
                rospy.wait_for_service('~service')
                try:
                    update_target = rospy.ServiceProxy(
                            '~service',
                            UpdateTarget
                            )
                    if work_order[bin_] == '<no target>':
                        response = update_target(bin=bin_, target='')
                    else:
                        response = update_target(bin=bin_, target=work_order[bin_])
                    if response.success is False:
                        rospy.logerr(
                                "{} : Failed to update target object"
                                .format(bin_)
                                )
                except rospy.ServiceException as e:
                    rospy.logerr("Service call failed. {}".format(e))
        self.prev_work_order = work_order.copy()


class SelectTarget(Plugin):
    def __init__(self, context):
        super(SelectTarget, self).__init__(context)
        self.setObjectName('SelectTarget')
        self._widget = SelectTargetWidget()
        context.add_widget(self._widget)
