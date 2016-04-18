#include <ros/ros.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/server/simple_action_server.h>
#include <std_msgs/Float32.h>

typedef actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> Server;

ros::Publisher servo_angle;

void execute(const control_msgs::FollowJointTrajectoryGoalConstPtr& goal, Server* as_)
{
    bool success = true;

    ROS_INFO("gripper_front/limb/right/servo/angle: Executing requested joint trajectory");

    for (int i = 0; i < goal->trajectory.points.size(); i++) {

        if (as_->isPreemptRequested()) {
            ROS_INFO("gripper_front/limb/right/follow_joint_trajectory: Preempted");
            as_->setPreempted();
            success = false;
            break;
        }

        if (!ros::ok()) {
            ROS_INFO("gripper_front/limb/right/follow_joint_trajectory: Aborted");
            as_->setAborted();
            return;
        }

        float rad = goal->trajectory.points[i].positions[0];
        std_msgs::Float32 angle_msg;
        angle_msg.data = rad;
        servo_angle.publish(angle_msg);

        control_msgs::FollowJointTrajectoryFeedback feedback_;
        
        feedback_.joint_names.push_back(goal->trajectory.joint_names[0]);
        feedback_.desired.positions.resize(1);
        feedback_.actual.positions.resize(1);
        feedback_.error.positions.resize(1);

        feedback_.desired.positions[0] = rad;
        feedback_.actual.positions[0] = rad;
        feedback_.error.positions[0] = 0;

        ros::Duration wait = (goal->trajectory.header.stamp + goal->trajectory.points[i].time_from_start) - ros::Time::now();
        if (wait > ros::Duration(0))
            wait.sleep();
        feedback_.header.stamp = ros::Time::now();
        as_->publishFeedback(feedback_);
    }

    control_msgs::FollowJointTrajectoryResult result_;
    if (success) {
        ROS_INFO("gripper_front/limb/right/servo/angle: Succeeded");
        result_.error_code = result_.SUCCESSFUL;
        as_->setSucceeded(result_);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gripper_controll_servoer");
    ros::NodeHandle n;
    servo_angle = n.advertise<std_msgs::Float32>("gripper_front/limb/right/servo/angle", 10);
    Server server(n, "gripper_front/limb/right/follow_joint_trajectory", boost::bind(&execute, _1, &server), false);
    server.start();
    ros::spin();
    return 0;
}
