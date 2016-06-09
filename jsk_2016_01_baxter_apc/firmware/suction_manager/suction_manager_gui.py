#!/usr/bin/env python

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QWidget

import os
import rospy
import rospkg
import rosservice


class SuctionManagerGUIWidget(QWidget):
    def __init__(self):
        super(SuctionManagerGUIWidget, self).__init__()
        rp = rospkg.RosPack()
        ui_file = os.path.join(rp.get_path('jsk_2016_01_baxter_apc'),
            'resource', 'suction_manager_gui')
        loadUi(ui_file, self)
        self.setObjectName('SuctionManagerGUIUi')
        self.yes_button.clicked[bool].connect(self._clicked_yes_button)
        self.no_button.clicked[bool].connect(self._clicked_no_button)
        self.yes_button.setEnabled(False)
        self.no_button.setEnabled(False)
        self.yes = None
        service_name = rospy.get_namespace() + 'rqt_yn_btn'
        if service_name in rosservice.get_service_list():
            rospy.logwarn('{} is already advertised'.format(service_name))
            return
        self.srv = rospy.Service('rqt_yn_btn', YesNo, self._handle_yn_btn)

    def _clicked_yes_button(self):
        """Handle events of being clicked yes button."""
        self.yes = True

    def _clicked_no_button(self):
        """Handle events of being clicked no button."""
        self.yes = False

    def _handle_yn_btn(self, req):
        """Callback function of service,
        and handle enable/disable of the buttons."""
        self.yes = None  # initialize
        self.yes_button.setEnabled(True)
        self.no_button.setEnabled(True)
        while self.yes is None:  # wait for user input
            rospy.sleep(1.)
        self.yes_button.setEnabled(False)
        self.no_button.setEnabled(False)
        return YesNoResponse(yes=self.yes)

    def __del__(self):
        self.srv.shutdown()


class SuctionManagerGUI(Plugin):
    def __init__(self, context):
        super(SuctionManagerGUI, self).__init__(context)
        self.setObjectName('SuctionManagerGUI')
        self._widget = SuctionManagerGUIWidget()
        context.add_widget(self._widget)

