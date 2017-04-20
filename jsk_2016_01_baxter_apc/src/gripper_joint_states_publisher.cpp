#include <map>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float32.h>

std::map<std::string, double> g_servo_angle;
std::map<std::string, std::string> g_side_to_jt;
sensor_msgs::JointState g_joint_states_except_gripper;

void angle_stateCb(const std_msgs::Float32::ConstPtr& sub_msg, const std::string& side)
{
  g_servo_angle[side] = sub_msg->data;
}

void joint_statesCb(const sensor_msgs::JointState::ConstPtr& joint_states)
{
  for (std::map<std::string, std::string>::iterator pair = g_side_to_jt.begin(); pair != g_side_to_jt.end(); ++pair)
  {
    for (int i = 0; i < joint_states->name.size(); i++)
    {
      if (joint_states->name[i] == pair->second)
        return;
    }
  }

  g_joint_states_except_gripper = (*joint_states);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_joint_states_publisher");
  ros::NodeHandle n;
  ros::Publisher servo_angle_pub = n.advertise<sensor_msgs::JointState>("robot/joint_states", 100);
  ros::Subscriber joint_states_sub = n.subscribe("robot/joint_states", 10, joint_statesCb);

  std::vector<std::string> limb;
  if (!ros::param::get("~limb", limb))
  {
    // Set default value
    limb.push_back("right");
    limb.push_back("left");
    ros::param::set("~limb", limb);
  }
  std::vector<ros::Subscriber> servo_angle_subs;
  for (std::vector<std::string>::iterator l = limb.begin(); l != limb.end(); ++l)
  {
    if (*l == "right" || *l == "left")
    {
      servo_angle_subs.push_back(n.subscribe<std_msgs::Float32>("gripper_front/limb/" + *l + "/servo/angle/state", 10,
                                                                boost::bind(&angle_stateCb, _1, *l)));
      g_side_to_jt[*l] = *l + "_gripper_vacuum_pad_joint";
    }
  }

  ros::Rate r(30);
  while (ros::ok())
  {
    ros::spinOnce();
    if (g_servo_angle.size() == g_side_to_jt.size())
    {
      sensor_msgs::JointState pub_msg;

      pub_msg = g_joint_states_except_gripper;
      pub_msg.header.stamp = ros::Time::now();
      for (std::map<std::string, std::string>::iterator pair = g_side_to_jt.begin(); pair != g_side_to_jt.end(); ++pair)
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
