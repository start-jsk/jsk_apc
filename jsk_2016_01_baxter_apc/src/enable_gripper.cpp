#include <ros/ros.h>
#include <baxter_core_msgs/AssemblyState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

ros::Publisher gripper_enable_pub;
bool robot_enabled;
ros::Time right_last_command_time;
bool right_servo_on;

void robot_stateCb(const baxter_core_msgs::AssemblyState::ConstPtr& sub_msg)
{
  std_msgs::Bool pub_msg;
  pub_msg.data = sub_msg->enabled;
  gripper_enable_pub.publish(pub_msg);
  if (robot_enabled == false && sub_msg->enabled == true)
  {
    right_last_command_time = ros::Time::now();
    ROS_INFO("%s: Enabled grippers", ros::this_node::getName().c_str());
  }
  else if (robot_enabled != sub_msg->enabled)
    ROS_INFO("%s: Disabled grippers", ros::this_node::getName().c_str());
  robot_enabled = sub_msg->enabled;
}

void right_servo_angleCb(const std_msgs::Float32::ConstPtr& msg)
{
  right_last_command_time = ros::Time::now();
}

void right_servo_torqueCb(const std_msgs::Bool::ConstPtr& msg)
{
  right_last_command_time = ros::Time::now();
  right_servo_on = msg->data;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "enable_gripper");
  ros::NodeHandle n;
  gripper_enable_pub = n.advertise<std_msgs::Bool>("gripper_front/enable", 10);
  ros::Publisher right_servo_torque_pub = n.advertise<std_msgs::Bool>("gripper_front/limb/right/servo/torque", 10);
  ros::Subscriber robot_state_sub = n.subscribe<baxter_core_msgs::AssemblyState>("robot/state", 10, robot_stateCb);
  ros::Subscriber right_servo_torque_sub =
      n.subscribe<std_msgs::Bool>("gripper_front/limb/right/servo/torque", 10, right_servo_torqueCb);
  ros::Subscriber right_servo_angle_sub =
      n.subscribe<std_msgs::Float32>("gripper_front/limb/right/servo/angle", 10, right_servo_angleCb);
  robot_enabled = false;
  right_last_command_time = ros::Time::now();
  right_servo_on = false;

  ros::Rate r(100);
  while (ros::ok())
  {
    ros::spinOnce();
    const int wait_min = 15;
    if (robot_enabled && right_servo_on && (ros::Time::now() - right_last_command_time) > ros::Duration(wait_min * 60))
    {
      std_msgs::Bool msg;
      msg.data = false;
      right_servo_torque_pub.publish(msg);
      ROS_INFO("%s: Power off right gripper's servo becase no command came to right gripper in this %d minutes",
               ros::this_node::getName().c_str(), wait_min);
    }
    r.sleep();
  }

  return 0;
}
