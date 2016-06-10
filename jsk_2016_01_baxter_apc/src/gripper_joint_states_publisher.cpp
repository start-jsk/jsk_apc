#include <map>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float32.h>

std::map<std::string, double> g_servo_angle;

void angle_stateCb(const std_msgs::Float32::ConstPtr& sub_msg, const std::string& side)
{
  g_servo_angle[side] = sub_msg->data;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_joint_states_publisher");
  ros::NodeHandle n;
  ros::Publisher servo_angle_pub = n.advertise<sensor_msgs::JointState>("robot/joint_states", 100);
  ros::Subscriber right_servo_angle_sub = n.subscribe<std_msgs::Float32>("gripper_front/limb/right/servo/angle/state",
                                                                         10, boost::bind(&angle_stateCb, _1, "right"));
  ros::Subscriber left_servo_angle_sub = n.subscribe<std_msgs::Float32>("gripper_front/limb/left/servo/angle/state", 10,
                                                                        boost::bind(&angle_stateCb, _1, "left"));

  std::map<std::string, std::string> side_to_jt;
  side_to_jt["right"] = "right_gripper_vacuum_pad_joint";
  side_to_jt["left"] = "left_gripper_vacuum_pad_joint";
  ros::Rate r(30);
  while (ros::ok())
  {
    ros::spinOnce();
    if (g_servo_angle.size() == side_to_jt.size())
    {
      sensor_msgs::JointState pub_msg;

      pub_msg.header.stamp = ros::Time::now();
      for (std::map<std::string, std::string>::iterator pair = side_to_jt.begin(); pair != side_to_jt.end(); ++pair)
      {
        pub_msg.name.push_back(pair->second);
        pub_msg.position.push_back(g_servo_angle[pair->first]);
        pub_msg.velocity.push_back(0);
        pub_msg.effort.push_back(0);
      }
      servo_angle_pub.publish(pub_msg);
    }
    r.sleep();
  }

  return 0;
}
