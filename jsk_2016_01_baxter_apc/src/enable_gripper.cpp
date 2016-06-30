#include <vector>
#include <map>

#include <ros/ros.h>
#include <baxter_core_msgs/AssemblyState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

ros::Publisher g_gripper_enable_pub;
ros::Publisher g_vacuum_right_pub;
ros::Publisher g_vacuum_left_pub;
bool g_robot_enabled;
std::map<std::string, ros::Time> g_last_command_time;
std::map<std::string, bool> g_servo_on;

void robot_stateCb(const baxter_core_msgs::AssemblyState::ConstPtr& sub_msg)
{
  std_msgs::Bool pub_msg;
  pub_msg.data = sub_msg->enabled;
  g_gripper_enable_pub.publish(pub_msg);
  if (!pub_msg.data)
  {
    g_vacuum_right_pub.publish(pub_msg);
    g_vacuum_left_pub.publish(pub_msg);
  }
  if (g_robot_enabled == false && sub_msg->enabled == true)
  {
    g_last_command_time["right"] = ros::Time::now();
    g_last_command_time["left"] = ros::Time::now();
    ROS_INFO("%s: Enabled grippers", ros::this_node::getName().c_str());
  }
  else if (g_robot_enabled != sub_msg->enabled)
    ROS_INFO("%s: Disabled grippers", ros::this_node::getName().c_str());
  g_robot_enabled = sub_msg->enabled;
}

void servo_angleCb(const std_msgs::Float32::ConstPtr& msg, const std::string& side)
{
  g_last_command_time[side] = ros::Time::now();
}

void servo_torqueCb(const std_msgs::Bool::ConstPtr& msg, const std::string& side)
{
  g_last_command_time[side] = ros::Time::now();
  g_servo_on[side] = msg->data;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "enable_gripper");
  ros::NodeHandle n;
  g_gripper_enable_pub = n.advertise<std_msgs::Bool>("gripper_front/enable", 10);
  g_vacuum_right_pub = n.advertise<std_msgs::Bool>("vacuum_gripper/limb/right", 10);
  g_vacuum_left_pub = n.advertise<std_msgs::Bool>("vacuum_gripper/limb/left", 10);
  std::map<std::string, ros::Publisher> servo_torque_pub;
  servo_torque_pub["right"] = n.advertise<std_msgs::Bool>("gripper_front/limb/right/servo/torque", 10);
  servo_torque_pub["left"] = n.advertise<std_msgs::Bool>("gripper_front/limb/left/servo/torque", 10);
  ros::Subscriber robot_state_sub = n.subscribe<baxter_core_msgs::AssemblyState>("robot/state", 10, robot_stateCb);
  ros::Subscriber right_servo_torque_sub = n.subscribe<std_msgs::Bool>("gripper_front/limb/right/servo/torque", 10,
                                                                       boost::bind(&servo_torqueCb, _1, "right"));
  ros::Subscriber left_servo_torque_sub =
      n.subscribe<std_msgs::Bool>("gripper_front/limb/left/servo/torque", 10, boost::bind(&servo_torqueCb, _1, "left"));
  ros::Subscriber right_servo_angle_sub = n.subscribe<std_msgs::Float32>("gripper_front/limb/right/servo/angle", 10,
                                                                         boost::bind(&servo_angleCb, _1, "right"));
  ros::Subscriber left_servo_angle_sub = n.subscribe<std_msgs::Float32>("gripper_front/limb/left/servo/angle", 10,
                                                                        boost::bind(&servo_angleCb, _1, "left"));
  g_robot_enabled = false;
  g_last_command_time["right"] = ros::Time::now();
  g_last_command_time["left"] = ros::Time::now();
  g_servo_on["right"] = false;
  g_servo_on["left"] = false;

  const int wait_min = 15;
  std::vector<std::string> sides;
  sides.push_back("right");
  sides.push_back("left");
  ros::Rate r(100);
  while (ros::ok())
  {
    ros::spinOnce();
    for (std::vector<std::string>::iterator side = sides.begin(); side != sides.end(); ++side)
    {
      if (g_robot_enabled && g_servo_on[*side] &&
          (ros::Time::now() - g_last_command_time[*side]) > ros::Duration(wait_min * 60))
      {
        std_msgs::Bool msg;
        msg.data = false;
        servo_torque_pub[*side].publish(msg);
        ROS_INFO("%s: Power off %s gripper's servo because no command came to it in this %d minutes",
                 ros::this_node::getName().c_str(), (*side).c_str(), wait_min);
      }
    }
    r.sleep();
  }

  return 0;
}
