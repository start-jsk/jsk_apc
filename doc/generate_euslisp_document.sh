#!/bin/bash

if [ ! -e /tmp/documentation.l ]; then
  wget https://raw.githubusercontent.com/euslisp/EusLisp/Euslisp-9.18/lib/llib/documentation.l -O /tmp/documentation.l
fi

roseus /tmp/documentation.l "(progn
(setq filename \"package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l\")
(load filename)
(setq output-filename \"/tmp/baxter_interface.md\")
(make-document filename output-filename)
(exit)"
echo -e "euslisp/jsk_2015_05_baxter_apc/baxter-interface.l\n=================================================\n" > jsk_2015_05_baxter_apc/euslisp/baxter_interface.rst
pandoc --from=markdown --to=rst /tmp/baxter_interface.md >> jsk_2015_05_baxter_apc/euslisp/baxter_interface.rst

roseus /tmp/documentation.l "(progn
(setq filename \"package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter.l\")
(load filename)
(setq output-filename \"/tmp/baxter.md\")
(make-document filename output-filename)
(exit)"
echo -e "euslisp/jsk_2015_05_baxter_apc/baxter.l\n=======================================\n" > jsk_2015_05_baxter_apc/euslisp/baxter.rst
pandoc --from=markdown --to=rst /tmp/baxter.md >> jsk_2015_05_baxter_apc/euslisp/baxter.rst

roseus /tmp/documentation.l "(progn
(setq filename \"package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l\")
(load filename)
(setq output-filename \"/tmp/util.md\")
(make-document filename output-filename)
(exit)"
echo -e "euslisp/jsk_2015_05_baxter_apc/util.l\n=====================================\n" > jsk_2015_05_baxter_apc/euslisp/util.rst
pandoc --from=markdown --to=rst /tmp/util.md >> jsk_2015_05_baxter_apc/euslisp/util.rst
