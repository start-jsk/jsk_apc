#include <map>

#include <ros/ros.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/server/simple_action_server.h>
#include <std_msgs/Float32.h>

std::map<std::string, ros::Publisher> g_gripper_servo_angle_pub;

class GripperAction
{
protected:
  ros::NodeHandle nh_;
  // NodeHandle instance mustbe created before this line,

  actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> as_;
  std::string action_name_, side_;
  // create messages that are used to publisssh feedback/result
  control_msgs::FollowJointTrajectoryFeedback feedback_;
  control_msgs::FollowJointTrajectoryResult result_;

public:
  GripperAction(std::string name, std::string side);
  void executeCB(const control_msgs::FollowJointTrajectoryGoalConstPtr& goal);
};

GripperAction::GripperAction(std::string name, std::string side):
    as_(nh_, name, boost::bind(&GripperAction::executeCB, this, _1), false),
    action_name_(name),
    side_(side)
{
  as_.start();
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
  float prev_rad = 0;

  // Loop until end of trajectory
  for (int i = 0; i < goal->trajectory.points.size(); i++)
  {
    ros::Duration segment_time = goal->trajectory.points[i].time_from_start;
    if ( i > 0 )
      segment_time = segment_time -  goal->trajectory.points[i-1].time_from_start;
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

    float rad = goal->trajectory.points[i].positions[0];
    std_msgs::Float32 angle_msg;

    // Publish feedbacks until next point
    ros::Rate feedback_rate(100); // 10 msec
    ros::Time now;
    //
    float velocity = 0;
    if ( i < goal->trajectory.points.size() - 1) {
      float d0 = rad - prev_rad;
      float d1 = goal->trajectory.points[i+1].positions[0];
      float t0 = segment_time.toSec();
      float t1 = goal->trajectory.points[i+1].time_from_start.toSec();
      float v0 = d0/t0;
      float v1 = d1/t1;
      if ( v0 * v1 >= 0 ) {
        velocity = 0.5 * (v0 + v1);
      } else {
        velocity = 0;
      }
    }
    static float x = prev_rad;
    static float v = 0;
    static float a = 0;
    float gx = rad;
    float gv = velocity;
    float ga = 0;
    float target_t = segment_time.toSec();
    float A=(gx-(x+v*target_t+(a/2.0)*target_t*target_t))/(target_t*target_t*target_t);
    float B=(gv-(v+a*target_t))/(target_t*target_t);
    float C=(ga-a)/target_t;

    float a0=x;
    float a1=v;
    float a2=a/2.0;
    float a3=10*A-4*B+0.5*C;
    float a4=(-15*A+7*B-C)/target_t;
    float a5=(6*A-3*B+0.5*C)/(target_t*target_t);
    do
    {
      // see seqplay::playPattern in https://github.com/fkanehiro/hrpsys-base/blob/master/rtc/SequencePlayer/seqplay.cpp
      now = ros::Time::now();
      // hoff arbib code is copied from https://github.com/fkanehiro/hrpsys-base/blob/master/rtc/SequencePlayer/interpolator.cpp
      float t = (now - start_segment_time).toSec();
      // angle_msg.data = (rad - prev_rad) * (now - start_segment_time).toSec()/segment_time.toSec() + prev_rad; // linear
      angle_msg.data = a0+a1*t+a2*t*t+a3*t*t*t+a4*t*t*t*t+a5*t*t*t*t*t;

      g_gripper_servo_angle_pub[side_].publish(angle_msg);
      feedback_rate.sleep();

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
      feedback_.header.stamp = ros::Time::now();
      as_.publishFeedback(feedback_);

    } while (ros::ok() &&
             now < (start_time + goal->trajectory.points[i].time_from_start));

    start_segment_time = now;
    prev_rad = rad;
    x = gx;
    v = gv;
    a = ga;
  }

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
  ros::NodeHandle n;
  g_gripper_servo_angle_pub["right"] = n.advertise<std_msgs::Float32>("gripper_front/limb/right/servo/angle", 10);
  g_gripper_servo_angle_pub["left"] = n.advertise<std_msgs::Float32>("gripper_front/limb/left/servo/angle", 10);
  GripperAction right_server("gripper_front/limb/right/follow_joint_trajectory", "right");
  GripperAction left_server("gripper_front/limb/left/follow_joint_trajectory", "left");
  ros::spin();
  return 0;
}
