#include <vector>
#include <map>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
// We probably don't need this because limiters are on dynamixel controllers
//#include <joint_limits_interface/joint_limits_interface.h>
#include <hardware_interface/robot_hw.h>
#include <transmission_interface/simple_transmission.h>
#include <transmission_interface/transmission_interface.h>
#include <controller_manager/controller_manager.h>

#include <std_msgs/Float64.h>
#include <dynamixel_msgs/JointState.h>

class GripperRosControl : public hardware_interface::RobotHW
{
private:
  ros::NodeHandle nh_;

  hardware_interface::JointStateInterface jnt_state_interface_;
  hardware_interface::PositionJointInterface pos_jnt_interface_;

  const std::vector<std::string> actr_names_;
  const std::vector<std::string> jnt_names_;
  const std::vector<std::string> controller_names_;

  // Actuator raw data
  std::map<std::string, double> actr_curr_pos_;
  std::map<std::string, double> actr_curr_vel_;
  std::map<std::string, double> actr_curr_eff_;
  std::map<std::string, double> actr_cmd_pos_;

  // Joint raw data
  std::map<std::string, double> jnt_curr_pos_;
  std::map<std::string, double> jnt_curr_vel_;
  std::map<std::string, double> jnt_curr_eff_;
  std::map<std::string, double> jnt_cmd_pos_;

  // For transmission between actuator and joint
  transmission_interface::ActuatorToJointStateInterface actr_to_jnt_state_;
  transmission_interface::JointToActuatorPositionInterface jnt_to_actr_pos_;
  std::vector<boost::shared_ptr<transmission_interface::SimpleTransmission> > trans_;
  std::map<std::string, transmission_interface::ActuatorData> actr_curr_data_;
  std::map<std::string, transmission_interface::ActuatorData> actr_cmd_data_;
  std::map<std::string, transmission_interface::JointData> jnt_curr_data_;
  std::map<std::string, transmission_interface::JointData> jnt_cmd_data_;

  // ROS publishers
  std::map<std::string, ros::Publisher> actr_cmd_pub_;

  // ROS subscribers
  std::map<std::string, ros::Subscriber> actr_state_sub_;

  std::map<std::string, dynamixel_msgs::JointState> received_actr_states_;

  // For multi-threaded spinning
  boost::shared_ptr<ros::AsyncSpinner> subscriber_spinner_;
  ros::CallbackQueue subscriber_queue_;

public:
  GripperRosControl(const std::vector<std::string>& actr_names, const std::vector<std::string>& jnt_names,
                    const std::vector<std::string>& controller_names, const std::vector<double>& reducers)
    : actr_names_(actr_names)
    , jnt_names_(jnt_names)
    , controller_names_(controller_names)
  {
    for (int i = 0; i < jnt_names_.size(); i++)
    {
      std::string actr = actr_names_[i];
      std::string jnt = jnt_names_[i];

      // Get transmission
      boost::shared_ptr<transmission_interface::SimpleTransmission> t_ptr(
          new transmission_interface::SimpleTransmission(reducers[i]));
      trans_.push_back(t_ptr);

      // Initialize and wrap raw current data
      actr_curr_data_[actr].position.push_back(&actr_curr_pos_[actr]);
      actr_curr_data_[actr].velocity.push_back(&actr_curr_vel_[actr]);
      actr_curr_data_[actr].effort.push_back(&actr_curr_eff_[actr]);
      jnt_curr_data_[jnt].position.push_back(&jnt_curr_pos_[jnt]);
      jnt_curr_data_[jnt].velocity.push_back(&jnt_curr_vel_[jnt]);
      jnt_curr_data_[jnt].effort.push_back(&jnt_curr_eff_[jnt]);

      // Initialize and wrap raw command data
      actr_cmd_data_[actr].position.push_back(&actr_cmd_pos_[actr]);
      jnt_cmd_data_[jnt].position.push_back(&jnt_cmd_pos_[jnt]);

      // Register transmissions to each interface
      actr_to_jnt_state_.registerHandle(transmission_interface::ActuatorToJointStateHandle(
          actr + "_trans", trans_[i].get(), actr_curr_data_[actr], jnt_curr_data_[jnt]));
      jnt_to_actr_pos_.registerHandle(transmission_interface::JointToActuatorPositionHandle(
          jnt + "_trans", trans_[i].get(), actr_cmd_data_[actr], jnt_cmd_data_[jnt]));

      // Connect and register the joint state handle
      hardware_interface::JointStateHandle state_handle(jnt, &jnt_curr_pos_[jnt], &jnt_curr_vel_[jnt],
                                                        &jnt_curr_eff_[jnt]);
      jnt_state_interface_.registerHandle(state_handle);

      // Connect and register the joint position handle
      hardware_interface::JointHandle pos_handle(jnt_state_interface_.getHandle(jnt), &jnt_cmd_pos_[jnt]);
      pos_jnt_interface_.registerHandle(pos_handle);

      // ROS publishers and subscribers
      actr_cmd_pub_[actr] = nh_.advertise<std_msgs::Float64>("dxl/" + controller_names_[i] + "/command", 5);
      actr_state_sub_[actr] =
          nh_.subscribe("dxl/" + controller_names_[i] + "/state", 1, &GripperRosControl::actrStateCallback, this);
    }

    // Register interfaces
    registerInterface(&jnt_state_interface_);
    registerInterface(&pos_jnt_interface_);

    // Start spinning
    nh_.setCallbackQueue(&subscriber_queue_);
    subscriber_spinner_.reset(new ros::AsyncSpinner(1, &subscriber_queue_));
    subscriber_spinner_->start();
  }

  void cleanup()
  {
    subscriber_spinner_->stop();
  }

  void read()
  {
    // Update actuator current state
    for (int i = 0; i < actr_names_.size(); i++)
    {
      actr_curr_pos_[actr_names_[i]] = received_actr_states_[actr_names_[i]].current_pos;
      actr_curr_vel_[actr_names_[i]] = received_actr_states_[actr_names_[i]].velocity;
    }

    // Propagate current actuator state to joints
    actr_to_jnt_state_.propagate();
  }

  void write()
  {
    // Propagate joint commands to actuators
    jnt_to_actr_pos_.propagate();

    // Publish command to actuator
    for (int i = 0; i < actr_names_.size(); i++)
    {
      std_msgs::Float64 msg;
      msg.data = actr_cmd_pos_[actr_names_[i]];
      actr_cmd_pub_[actr_names_[i]].publish(msg);
    }
  }

  void actrStateCallback(const dynamixel_msgs::JointStateConstPtr& dxl_actr_state)
  {
    received_actr_states_[dxl_actr_state->name] = *dxl_actr_state;
  }
};  // end class GripperRosControl

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_v5_ros_control_node");

  std::vector<std::string> actr_names;
  std::vector<std::string> jnt_names;
  std::vector<std::string> controller_names;
  std::vector<double> reducers;
  int rate_hz;

  if (!(ros::param::get("~actuator_names", actr_names) && ros::param::get("~joint_names", jnt_names) &&
        ros::param::get("~controller_names", controller_names) && ros::param::get("~mechanical_reduction", reducers) &&
        ros::param::get("~control_rate", rate_hz)))
  {
    ROS_ERROR("Couldn't get necessary parameters");
    return 0;
  }

  GripperRosControl gripper(actr_names, jnt_names, controller_names, reducers);
  controller_manager::ControllerManager cm(&gripper);

  // For non-realtime spinner thread
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // Control loop
  ros::Rate rate(rate_hz);
  ros::Time prev_time = ros::Time::now();

  while (ros::ok())
  {
    const ros::Time now = ros::Time::now();
    const ros::Duration elapsed_time = now - prev_time;

    gripper.read();
    cm.update(now, elapsed_time);
    gripper.write();
    prev_time = now;

    rate.sleep();
  }
  spinner.stop();
  gripper.cleanup();

  return 0;
}
