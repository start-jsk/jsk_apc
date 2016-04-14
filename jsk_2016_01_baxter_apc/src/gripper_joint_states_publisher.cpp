#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float32.h>

ros::Publisher servo_angle_pub;
void angle_stateCb(const std_msgs::Float32::ConstPtr& sub_msg)
{
    sensor_msgs::JointState pub_msg;

    pub_msg.name.push_back("right_gripper_vacuum_pad_joint");

    pub_msg.header.stamp = ros::Time::now();
    pub_msg.position.resize(1);
    pub_msg.velocity.resize(1);
    pub_msg.effort.resize(1);
    pub_msg.position[0] = sub_msg->data;
    pub_msg.velocity[0] = 0;
    pub_msg.effort[0] = 0;

    servo_angle_pub.publish(pub_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gripper_joint_states_publisher");
    ros::NodeHandle n;
    servo_angle_pub = n.advertise<sensor_msgs::JointState>("robot/joint_states", 100);
    ros::Subscriber servo_angle_sub = n.subscribe<std_msgs::Float32>("gripper_front/limb/right/servo/angle/state", 10, angle_stateCb);

    ros::spin();

    return 0;
}
