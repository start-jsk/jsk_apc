#/bin/sh
#
rostopic pub /semi/move_larm2target_bin std_msgs/String 'f' -1
sleep 7
rostopic pub /semi/move_larm2target_bin std_msgs/String 'e' -1
sleep 7
rostopic pub /semi/move_larm2target_bin std_msgs/String 'd' -1
sleep 7
rostopic pub /semi/move_larm2target_bin std_msgs/String 'c' -1
sleep 7
rostopic pub /semi/move_larm2target_bin std_msgs/String 'b' -1
sleep 7
rostopic pub /semi/move_larm2target_bin std_msgs/String 'a' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'g' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'h' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'i' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'j' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'k' -1
sleep 7
rostopic pub /semi/move_rarm2target_bin std_msgs/String 'l' -1
sleep 7

