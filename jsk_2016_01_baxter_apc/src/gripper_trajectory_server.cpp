#include <ros/ros.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/server/simple_action_server.h>
#include <std_msgs/Float32.h>
#include "minjerk.h"
#include <boost/shared_ptr.hpp>

class GripperAction
{
protected:
  ros::NodeHandle nh_;
  // member variables must be initialized in the order they're declared in.
  // http://stackoverflow.com/questions/12222417/why-should-i-initialize-member-variables-in-the-order-theyre-declared-in
  std::string side_;
  std::string ns_name_;
  actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> as_;
  // create messages that are used to publisssh feedback/result
  control_msgs::FollowJointTrajectoryFeedback feedback_;
  control_msgs::FollowJointTrajectoryResult result_;

  ros::Publisher gripper_servo_angle_pub_;
  ros::Subscriber gripper_servo_angle_sub_;

  float angle_state_;
  MinJerk minjerk;
public:
  GripperAction(std::string name, std::string side);
  void executeCB(const control_msgs::FollowJointTrajectoryGoalConstPtr& goal);
  void angle_stateCB(const std_msgs::Float32::ConstPtr& angle);
};

GripperAction::GripperAction(std::string name, std::string side):
  side_(side),
  ns_name_(name),
  as_(nh_, ns_name_+side_ + "follow_joint_trajectory", boost::bind(&GripperAction::executeCB, this, _1), false)
{
  angle_state_ = 0;

  gripper_servo_angle_pub_ = nh_.advertise<std_msgs::Float32>(ns_name_ + side_ + "/servo/angle", 10);
  gripper_servo_angle_sub_ = nh_.subscribe<std_msgs::Float32>(ns_name_ + side_ + "/servo/angle/state", 10,
                                                              &GripperAction::angle_stateCB, this);
  as_.start();
}

void GripperAction::angle_stateCB(const std_msgs::Float32::ConstPtr& angle)
{
  angle_state_ = angle->data;
}

void GripperAction::executeCB(const control_msgs::FollowJointTrajectoryGoalConstPtr& goal)
{
  bool success = true;
  std::string node_name(ros::this_node::getName());

  ROS_INFO("%s: Executing requested joint trajectory for %s gripper", node_name.c_str(), side_.c_str());

  // Wait for the specified execution time, if not provided use now
  ros::Time start_time = goal->trajectory.header.stamp;
  if (start_time == ros::Time(0, 0))
    start_time = ros::Time::now();
  ros::Duration wait = start_time - ros::Time::now();
  if (wait > ros::Duration(0))
    wait.sleep();
  ros::Time start_segment_time = start_time;
  float prev_rad = angle_state_;

  // setup interpolator
  std::vector<double> position_list, velocity_list, acceleration_list;
  std::vector<double> time_list;
  position_list.push_back(angle_state_);
  velocity_list.push_back(0);
  acceleration_list.push_back(0);
  for (int i = 0; i < goal->trajectory.points.size(); i++)
  {
    position_list.push_back(goal->trajectory.points[i].positions[0]);
    time_list.push_back(goal->trajectory.points[i].time_from_start.toSec());
    //
    double velocity = 0;
    if ( i <  goal->trajectory.points.size() - 1 ) {
      double d0 = position_list[i+1] - position_list[i];
      double d1 = goal->trajectory.points[i+1].positions[0] - goal->trajectory.points[i].positions[0];
      double t0 = time_list[i] - ((i>0)?time_list[i-1]:0);
      double t1 = goal->trajectory.points[i+1].time_from_start.toSec() - goal->trajectory.points[i].time_from_start.toSec();
      double v0 = d0/t0;
      double v1 = d1/t1;
      if ( v0 * v1 >= 0 ) {
        velocity = 0.5 * (v0 + v1);
      } else {
        velocity = 0;
      }
    }
    velocity_list.push_back(velocity);
    acceleration_list.push_back(0);
  }
  minjerk.Reset(position_list, velocity_list, acceleration_list, time_list);
  minjerk.StartInterpolation();

  // Loop until end of trajectory
  ros::Rate feedback_rate(100); // 10 msec
  ros::Time now, prev = ros::Time::now();
  do
  {
    if (as_.isPreemptRequested())
    {
      ROS_INFO("%s: Preempted for %s gripper", node_name.c_str(), side_.c_str());
      as_.setPreempted();
      success = false;
      break;
    }

    if (!ros::ok())
    {
      ROS_INFO("%s: Aborted for %s gripper", node_name.c_str(), side_.c_str());
      as_.setAborted();
      return;
    }

    feedback_rate.sleep();

    now = ros::Time::now();
    std_msgs::Float32 angle_msg;
    double rad = minjerk.PassTime((now-prev).toSec());;
    angle_msg.data = rad;
    gripper_servo_angle_pub_.publish(angle_msg);

    control_msgs::FollowJointTrajectoryFeedback feedback_;

    feedback_.joint_names.push_back(goal->trajectory.joint_names[0]);
    feedback_.desired.positions.resize(1);
    feedback_.actual.positions.resize(1);
    feedback_.error.positions.resize(1);

    feedback_.desired.positions[0] = rad;
    feedback_.actual.positions[0] = rad;
    feedback_.error.positions[0] = 0;

    feedback_.desired.time_from_start = now - start_time;
    feedback_.actual.time_from_start = now - start_time;
    feedback_.error.time_from_start = now - start_time;
    feedback_.header.stamp = now;
    as_.publishFeedback(feedback_);

    prev = now;
    prev_rad = rad;
  } while (ros::ok() &&
           now < (start_time + goal->trajectory.points[goal->trajectory.points.size()-1].time_from_start));

  angle_state_ = prev_rad; // just in case anlge_stae is not updated;

  control_msgs::FollowJointTrajectoryResult result_;
  if (success)
  {
    ROS_INFO("%s: Succeeded for %s gripper", node_name.c_str(), side_.c_str());
    result_.error_code = result_.SUCCESSFUL;
    as_.setSucceeded(result_);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_joint_trajectory_action_server");
  std::vector<std::string> limb;
  if (!ros::param::get("~limb", limb))
  {
    // Set default value
    limb.push_back("right");
    limb.push_back("left");
    ros::param::set("~limb", limb);
  }
  std::vector<boost::shared_ptr<GripperAction> > servers;
  for (std::vector<std::string>::iterator l = limb.begin(); l != limb.end(); ++l)
  {
    if (*l == "right" || *l == "left")
    {
      servers.push_back(boost::make_shared<GripperAction>("gripper_front/limb/", *l + "/"));
    }
  }
  ros::spin();
  return 0;
}
