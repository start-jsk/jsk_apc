#!/usr/bin/env python

import sys
from rqt_gui.main import Main
from jsk_2016_01_baxter_apc.suction_manager_gui import SuctionManagerGUI

plugin = 'SuctionManagerGUI'
main = Main(filename=plugin)
sys.exit(main.main(standalone=plugin))
