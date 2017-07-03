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

#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt16.h>
#include <dynamixel_msgs/JointState.h>
#include <baxter_core_msgs/AssemblyState.h>
#include <dynamixel_controllers/TorqueEnable.h>

class GripperRosControl : public hardware_interface::RobotHW
{
private:
  ros::NodeHandle nh_;

  hardware_interface::JointStateInterface jnt_state_interface_;
  hardware_interface::PositionJointInterface pos_jnt_interface_;

  const std::vector<std::string> actr_names_;
  const std::vector<std::string> jnt_names_;
  const std::vector<std::string> controller_names_;
  const std::vector<std::string> flex_names_;
  const std::vector<int> flex_thre_;
  const std::vector<double> wind_offset_flex_;

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
  ros::Publisher vacuum_pub_;

  // ROS subscribers
  std::map<std::string, ros::Subscriber> actr_state_sub_;
  std::map<std::string, ros::Subscriber> flex_sub_;
  ros::Subscriber robot_state_sub_;

  std::map<std::string, dynamixel_msgs::JointState> received_actr_states_;
  std::vector<int> received_flex_;
  std::vector<bool> is_flexion_;
  std::vector<int> flex_dec_cnt_;
  bool is_gripper_enabled_;

  // ROS service clients
  std::map<std::string, ros::ServiceClient> torque_enable_client_;

  // For multi-threaded spinning
  boost::shared_ptr<ros::AsyncSpinner> subscriber_spinner_;
  ros::CallbackQueue subscriber_queue_;

public:
  GripperRosControl(const std::vector<std::string>& actr_names, const std::vector<std::string>& jnt_names,
                    const std::vector<std::string>& controller_names, const std::vector<double>& reducers,
                    const std::vector<std::string>& flex_names, const std::vector<int>& flex_thre,
                    const std::vector<double>& wind_offset_flex)
    : actr_names_(actr_names)
    , jnt_names_(jnt_names)
    , controller_names_(controller_names)
    , flex_names_(flex_names)
    , flex_thre_(flex_thre)
    , wind_offset_flex_(wind_offset_flex)
    , is_gripper_enabled_(true)
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

    // Publisher for vacuum
    vacuum_pub_ = nh_.advertise<std_msgs::Bool>("vacuum", 10);

    received_flex_ = std::vector<int>(flex_names_.size(), 0);
    is_flexion_ = std::vector<bool>(flex_names_.size(), false);
    flex_dec_cnt_ = std::vector<int>(flex_names_.size(), 0);
    // Subscribers for flex
    for (int i = 0; i < flex_names_.size(); i++)
    {
      flex_sub_[flex_names_[i]] = nh_.subscribe<std_msgs::UInt16>("flex/" + flex_names_[i] + "/state", 1,
                                                        boost::bind(&GripperRosControl::flexCallback, this, _1, i));
    }

    // Subscriber for robot state
    robot_state_sub_ = nh_.subscribe("/robot/state", 1, &GripperRosControl::robotStateCallback, this);

    // Service clients for torque enabling
    for (std::vector<std::string>::const_iterator itr = controller_names_.begin(); itr != controller_names_.end();
         ++itr)
    {
      torque_enable_client_[*itr] =
          nh_.serviceClient<dynamixel_controllers::TorqueEnable>("dxl/" + *itr + "/torque_enable");
    }

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

      // If fingers flex, add offset angle to finger tendon winder
      if (actr_names_[i].find("finger_tendon_winder") != std::string::npos)
      {
        for (int j = 0; j < flex_names_.size(); j++)
        {
          if (is_flexion_[j])
          {
            actr_curr_pos_[actr_names_[i]] -= wind_offset_flex_[j];
          }
        }
      }
    }

    // Propagate current actuator state to joints
    actr_to_jnt_state_.propagate();
  }

  void write()
  {
    if (is_gripper_enabled_)
    {
      // Propagate joint commands to actuators
      jnt_to_actr_pos_.propagate();

      // Publish command to actuator
      for (int i = 0; i < actr_names_.size(); i++)
      {
        // If fingers flex, add offset angle to finger tendon winder
        if (actr_names_[i].find("finger_tendon_winder") != std::string::npos)
        {
          for (int j = 0; j < flex_names_.size(); j++)
          {
            if (is_flexion_[j])
            {
              actr_cmd_pos_[actr_names_[i]] += wind_offset_flex_[j];
            }
          }
        }

        std_msgs::Float64 msg;
        msg.data = actr_cmd_pos_[actr_names_[i]];
        actr_cmd_pub_[actr_names_[i]].publish(msg);
      }
    }
    else
    {
      // Switch off vacuum
      std_msgs::Bool vacuum;
      vacuum.data = false;
      vacuum_pub_.publish(vacuum);

      // Gripper servo off
      dynamixel_controllers::TorqueEnable srv;
      srv.request.torque_enable = false;
      for (std::vector<std::string>::const_iterator itr = controller_names_.begin(); itr != controller_names_.end();
           ++itr)
      {
        torque_enable_client_[*itr].call(srv);
      }
    }
  }

  bool isGripperEnabled()
  {
    return is_gripper_enabled_;
  }

  void actrStateCallback(const dynamixel_msgs::JointStateConstPtr& dxl_actr_state)
  {
    received_actr_states_[dxl_actr_state->name] = *dxl_actr_state;
  }

  void flexCallback(const std_msgs::UInt16ConstPtr& flex, const int& idx)
  {
    received_flex_[idx] = flex->data;
    if (received_flex_[idx] > flex_thre_[idx])
    {
      is_flexion_[idx] = true;
      flex_dec_cnt_[idx] = 0;
    }
    else
    {
      flex_dec_cnt_[idx]++;
      if (flex_dec_cnt_[idx] > 2)
      {
        is_flexion_[idx] = false;
        flex_dec_cnt_[idx] = 0;
      }
    }
  }

  void robotStateCallback(const baxter_core_msgs::AssemblyStateConstPtr& state)
  {
    is_gripper_enabled_ = state->enabled;
  }
};  // end class GripperRosControl

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_v6_ros_control_node");

  std::vector<std::string> actr_names;
  std::vector<std::string> jnt_names;
  std::vector<std::string> controller_names;
  std::vector<double> reducers;
  int rate_hz;
  std::vector<std::string> flex_names;
  std::vector<int> flex_thre;
  std::vector<double> wind_offset_flex;

  if (!(ros::param::get("~actuator_names", actr_names) && ros::param::get("~joint_names", jnt_names) &&
        ros::param::get("~controller_names", controller_names) && ros::param::get("~mechanical_reduction", reducers) &&
        ros::param::get("~control_rate", rate_hz) && ros::param::get("~flex_names", flex_names) &&
        ros::param::get("~flex_thresholds", flex_thre) && ros::param::get("~wind_offset_flex", wind_offset_flex)))
  {
    ROS_ERROR("Couldn't get necessary parameters");
    return 0;
  }

  GripperRosControl gripper(actr_names, jnt_names, controller_names, reducers, flex_names, flex_thre, wind_offset_flex);
  controller_manager::ControllerManager cm(&gripper);

  // For non-realtime spinner thread
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // Control loop
  ros::Rate rate(rate_hz);
  ros::Time prev_time = ros::Time::now();
  bool prev_gripper_enabled = true;

  while (ros::ok())
  {
    const ros::Time now = ros::Time::now();
    const ros::Duration elapsed_time = now - prev_time;
    const bool gripper_enabled = gripper.isGripperEnabled();

    gripper.read();
    cm.update(now, elapsed_time, !prev_gripper_enabled && gripper_enabled);
    gripper.write();
    prev_time = now;
    prev_gripper_enabled = gripper_enabled;

    rate.sleep();
  }
  spinner.stop();
  gripper.cleanup();

  return 0;
}
