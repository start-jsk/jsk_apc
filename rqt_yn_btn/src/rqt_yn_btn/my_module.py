import os
import rospy

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtGui import QWidget

from rqt_yn_btn.srv import YesOrNo, YesOrNoResponse


class MyPlugin(Plugin):
    def __init__(self, context):
        super(MyPlugin, self).__init__(context)
        # Give QObjects reasonable names
        self.setObjectName('MyPlugin')
        # Service initialize
        s = rospy.Service('/semi/rqt_yn_btn', YesOrNo, self.handle_yn_btn)
        # Create QWidget
        self._widget = QWidget()
        ui_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'MyPlugin.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)
        # Give QObjects reasonable names
        self._widget.setObjectName('MyPluginUi')
        # Add button
        self._widget.yes_button.clicked[bool].connect(self._handle_yes_clicked)
        self._widget.no_button.clicked[bool].connect(self._handle_no_clicked)
        self._widget.yes_button.setEnabled(False)
        self._widget.no_button.setEnabled(False)
        # Add widget to the user interface
        context.add_widget(self._widget)
        # set property
        self.yes = None

    def handle_yn_btn(self, req):
        """Callback function of service,
        and handle enable/disable of the buttons."""
        self.yes = None  # initialize
        self._widget.yes_button.setEnabled(True)
        self._widget.no_button.setEnabled(True)
        while self.yes is None:  # wait for user input
            rospy.sleep(1.)

        self._widget.yes_button.setEnabled(False)
        self._widget.no_button.setEnabled(False)

        return YesOrNoResponse(yes=self.yes)

    def _handle_yes_clicked(self):
        """Handle events of being clicked yes button."""
        self.yes = True

    def _handle_no_clicked(self):
        """Handle events of being clicked no button."""
        self.yes = False

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        pass

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        pass

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog
