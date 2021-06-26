#!/usr/bin/env python
from check_common import *


def ask_yesno():
    answer=""
    while not ( answer in ["y", "Y", "n", "N"]):
        print(OKGREEN + "Next? [y/n]", ENDC)
        answer = raw_input()
        if not ( answer in ["y", "Y", "n", "N"]):
            print(WARNING + "Please type y/n !", ENDC)
    if answer in ["n", "N"]:
        fail_print("You canceled....")
        exit(-1)

def show_description(show_what_next, sub, command):
    index_print("=========================================")
    print(OKGREEN + show_what_next, ENDC)
    print(OKGREEN + sub, ENDC)
    print()
    print(command)
    print()


if __name__ == "__main__":
    index_print("Let's Start APC")
    show_description("1 : Check Baxter Boot",
                     "    Please type below",
                     "    rosrun jsk_2015_05_baxter_apc check_after_boot.py")
    ask_yesno()

    show_description("2 : Launch baxter.launch",
                     "    Please type below",
                     "    roslaunch jsk_2015_05_baxter_apc baxter.launch")
    ask_yesno()

    show_description("3 : Check Baxter.launch",
                     "    Please type below",
                     "    rosrun jsk_2015_05_baxter_apc check_after_baxter.py")
    ask_yesno()
    
    show_description("4 : Launch setup.launch",
                     "    Please type below",
                     "    roslaunch jsk_2015_05_baxter_apc setup.launch")
    ask_yesno()

    show_description("5 : Check setup.launch",
                     "    Please type below",
                     "    rosrun jsk_2015_05_baxter_apc check_after_setup.py")
    ask_yesno()

    show_description("7 : Install apc.json",
                     "    Please copy",
                     "    cp apc.json `rospack find jsk_2015_05_baxter_apc`/data/apc.json")
    ask_yesno()

    show_description("8 : Check apc.json",
                     "    Please type below",
                     "    rosrun jsk_2015_05_baxter_apc check_json.py")
    ask_yesno()

    show_description("9 : Launch main.launch",
                     "    Please type below",
                     "    roslaunch jsk_2015_05_baxter_apc main.launch")
    ask_yesno()

    show_description("10 : Push the Button !",
                     "    Let's Start!!",
                     "    Good Luck!!!")
