// GPIO
#include "mraa.hpp"

// C++
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <limits>
#include <map>
#include <vector>

// ROS base
#include <ros/ros.h>
#include <ros/callback_queue.h>

// ros_control
#include <controller_manager/controller_manager.h>
#include <hardware_interface/robot_hw.h>
#include <transmission_interface/transmission_interface_loader.h>

// ROS msg and srv
#include <baxter_core_msgs/AssemblyState.h>
#include <dynamixel_controllers/TorqueEnable.h>
#include <dynamixel_msgs/JointState.h>
#include <force_proximity_ros/ProximityStamped.h>
#include <force_proximity_ros/ProximityArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt16.h>

class PressureSensorDriver
{
private:
  mraa::Spi spi_;
  uint16_t dig_T1_;
  int16_t dig_T2_;
  int16_t dig_T3_;
  uint16_t dig_P1_;
  int16_t dig_P2_;
  int16_t dig_P3_;
  int16_t dig_P4_;
  int16_t dig_P5_;
  int16_t dig_P6_;
  int16_t dig_P7_;
  int16_t dig_P8_;
  int16_t dig_P9_;

public:
  PressureSensorDriver(const int spi_bus = 2, const int spi_cs = 0, const uint32_t max_speed = 8000000)
    : spi_(spi_bus, spi_cs)
  {
    spi_.frequency(max_speed);
  }

  ~PressureSensorDriver()
  {
  }

  void initBME()
  {
    uint8_t tx[2];
    tx[0] = 0xF5 & 0x7F;
    tx[1] = 0x20;
    spi_.write(tx, 2);
    tx[0] = 0xF4 & 0x7F;
    tx[1] = 0x27;
    spi_.write(tx, 2);
  }

  void readTrim()
  {
    uint8_t tx[25] = {};
    uint8_t rx[25];
    tx[0] = 0x88 | 0x80;
    spi_.transfer(tx, rx, 25);

    dig_T1_ = (rx[2] << 8) | rx[1];
    dig_T2_ = (rx[4] << 8) | rx[3];
    dig_T3_ = (rx[6] << 8) | rx[5];
    dig_P1_ = (rx[8] << 8) | rx[7];
    dig_P2_ = (rx[10] << 8) | rx[9];
    dig_P3_ = (rx[12] << 8) | rx[11];
    dig_P4_ = (rx[14] << 8) | rx[13];
    dig_P5_ = (rx[16] << 8) | rx[15];
    dig_P6_ = (rx[18] << 8) | rx[17];
    dig_P7_ = (rx[20] << 8) | rx[19];
    dig_P8_ = (rx[22] << 8) | rx[21];
    dig_P9_ = (rx[24] << 8) | rx[23];
  }

  bool init()
  {
    initBME();
    readTrim();

    return true;
  }

  void readRawPressureAndTemperature(uint32_t* pres_raw, uint32_t* temp_raw)
  {
    uint8_t data[8];
    uint8_t tx[9] = {};
    uint8_t rx[9];
    tx[0] = 0xF7 | 0x80;
    spi_.transfer(tx, rx, 9);
    *pres_raw = rx[1];
    *pres_raw = ((*pres_raw) << 8) | rx[2];
    *pres_raw = ((*pres_raw) << 4) | (rx[3] >> 4);
    *temp_raw = rx[4];
    *temp_raw = ((*temp_raw) << 8) | rx[5];
    *temp_raw = ((*temp_raw) << 4) | (rx[6] >> 4);
  }

  int32_t calibTemperature(const int32_t temp_raw)
  {
    int32_t var1, var2, T;
    var1 = ((((temp_raw >> 3) - ((int32_t)dig_T1_ << 1))) * ((int32_t)dig_T2_)) >> 11;
    var2 =
        (((((temp_raw >> 4) - ((int32_t)dig_T1_)) * ((temp_raw >> 4) - ((int32_t)dig_T1_))) >> 12) * ((int32_t)dig_T3_)) >>
        14;

    return (var1 + var2);
  }

  uint32_t calibPressure(const int32_t pres_raw, const int32_t t_fine)
  {
    int32_t var1, var2;
    var1 = (((int32_t)t_fine) >> 1) - (int32_t)64000;
    var2 = (((var1 >> 2) * (var1 >> 2)) >> 11) * ((int32_t)dig_P6_);
    var2 = var2 + ((var1 * ((int32_t)dig_P5_)) << 1);
    var2 = (var2 >> 2) + (((int32_t)dig_P4_) << 16);
    var1 = (((dig_P3_ * (((var1 >> 2) * (var1 >> 2)) >> 13)) >> 3) + ((((int32_t)dig_P2_) * var1) >> 1)) >> 18;
    var1 = ((((32768 + var1)) * ((int32_t)dig_P1_)) >> 15);
    if (var1 == 0)
    {
      return 0;
    }
    uint32_t P = (((uint32_t)(((int32_t)1048576) - pres_raw) - (var2 >> 12))) * 3125;
    if (P < 0x80000000)
    {
      P = (P << 1) / ((uint32_t)var1);
    }
    else
    {
      P = (P / (uint32_t)var1) * 2;
    }
    var1 = (((int32_t)dig_P9_) * ((int32_t)(((P >> 3) * (P >> 3)) >> 13))) >> 12;
    var2 = (((int32_t)(P >> 2)) * ((int32_t)dig_P8_)) >> 13;
    P = (uint32_t)((int32_t)P + ((var1 + var2 + dig_P7_) >> 4));
    return P;
  }

  double getPressure()
  {
    uint32_t pres_raw, temp_raw;
    readRawPressureAndTemperature(&pres_raw, &temp_raw);
    return ((double)calibPressure(pres_raw, calibTemperature(temp_raw)) / 100.0);
  }
};  // end class PressureSensorDriver

class FlexSensorDriver
{
private:
  mraa::Spi spi_;
  const int sensor_num_;

public:
  FlexSensorDriver(const int sensor_num = 2, const int spi_bus = 2, const int spi_cs = 1,
                   const uint32_t max_speed = 1000000)
    : spi_(spi_bus, spi_cs), sensor_num_(sensor_num)
  {
    spi_.frequency(max_speed);
  }

  ~FlexSensorDriver()
  {
  }

  void getFlex(std::vector<uint16_t>* flex)
  {
    flex->clear();
    uint8_t tx[3] = {};
    uint8_t rx[3];
    for (int sensor_no = 0; sensor_no < sensor_num_; sensor_no++)
    {
      tx[0] = (0x18 | sensor_no) << 3;
      spi_.transfer(tx, rx, 3);
      uint16_t value = (rx[0] & 0x01) << 11;
      value |= rx[1] << 3;
      value |= rx[2] >> 5;
      flex->push_back(value);
    }
  }
};  // end class FlexSensorDriver

class Pca9547Mraa
{
private:
  mraa::I2c* i2c_;

public:
  Pca9547Mraa()
  {
  }

  ~Pca9547Mraa()
  {
  }

  void init(mraa::I2c* i2c)
  {
    i2c_ = i2c;
  }

  mraa::Result setChannel(const uint8_t mux_addr, const int8_t ch)
  {
    uint8_t tx;

    if (ch == -1)
    {
      // No channel selected
      tx &= 0xF7;
    }
    else if (0 <= ch && ch <= 7)
    {
      // Channel 0~7 selected
      tx = (uint8_t)ch | 0x08;
    }
    else
    {
      ROS_ERROR("I2C Multiplexer PCA9547 has no channel %d", ch);
      return mraa::ERROR_UNSPECIFIED;
    }

    i2c_->address(mux_addr);
    return i2c_->writeByte(tx);
  }
};  // end class Pca9547Mraa

class Vcnl4040Mraa
{
private:
  // Constants
  enum
  {
    // Command Registers
    PS_CONF1 = 0x03,
    PS_CONF3 = 0x04,
    PS_DATA_L = 0x08,
  };
  mraa::I2c* i2c_;
  uint8_t i2c_addr_;
  // Sensitivity of touch/release detection
  int sensitivity_;
  // exponential average weight parameter / cut-off frequency for high-pass filter
  double ea_;
  // low-pass filtered proximity reading
  double average_value_;
  // FA-II value
  double fa2_;

public:
  Vcnl4040Mraa()
    : sensitivity_(1000)
    , ea_(0.3)
    , average_value_(std::numeric_limits<double>::quiet_NaN())
    , fa2_(0)
  {
  }

  ~Vcnl4040Mraa()
  {
  }

  // Read from two Command Registers of VCNL4040
  uint16_t readCommandRegister(const uint8_t command_code)
  {
    i2c_->address(i2c_addr_);
    uint16_t rx = i2c_->readWordReg(command_code);
    return rx;
  }

  // Write to two Command Registers of VCNL4040
  mraa::Result writeCommandRegister(const uint8_t command_code, const uint8_t low_data, const uint8_t high_data)
  {
    uint16_t data = ((uint16_t)high_data << 8) | low_data;

    i2c_->address(i2c_addr_);
    return i2c_->writeWordReg(command_code, data);
  }

  // Configure VCNL4040
  void init(mraa::I2c* i2c, const uint8_t i2c_addr)
  {
    i2c_ = i2c;
    i2c_addr_ = i2c_addr;

    // Set PS_CONF3 and PS_MS
    uint8_t conf3 = 0x00;
    // uint8_t ms = 0x00;  // IR LED current to 50mA
    uint8_t ms = 0x01;  // IR LED current to 75mA
    // uint8_t ms = 0x02;  // IR LED current to 100mA
    // uint8_t ms = 0x06;  // IR LED current to 180mA
    // uint8_t ms = 0x07;  // IR LED current to 200mA
    writeCommandRegister(PS_CONF3, conf3, ms);
  }

  void startSensing()
  {
    // Clear PS_SD to turn on proximity sensing
    // uint8_t conf1 = 0x00;  // Clear PS_SD bit to begin reading
    uint8_t conf1 = 0x0E;  // Integrate 8T, Clear PS_SD bit to begin reading
    // uint8_t conf2 = 0x00;  // Clear PS to 12-bit
    uint8_t conf2 = 0x08;  // Set PS to 16-bit
    writeCommandRegister(PS_CONF1, conf1, conf2);
  }

  void stopSensing()
  {
    // Set PS_SD to turn off proximity sensing
    uint8_t conf1 = 0x01;  // Set PS_SD bit to stop reading
    uint8_t conf2 = 0x00;
    writeCommandRegister(PS_CONF1, conf1, conf2);
  }

  uint16_t getRawProximity()
  {
    return readCommandRegister(PS_DATA_L);
  }

  void getProximityStamped(force_proximity_ros::ProximityStamped* prox_st)
  {
    uint16_t raw = getRawProximity();
    // Record time of reading sensor
    prox_st->header.stamp = ros::Time::now();
    prox_st->proximity.proximity = raw;
    if (std::isnan(average_value_))
    {
      average_value_ = raw;
    }
    prox_st->proximity.average = average_value_;
    prox_st->proximity.fa2derivative = average_value_ - raw - fa2_;
    fa2_ = average_value_ - raw;
    prox_st->proximity.fa2 = fa2_;
    if (fa2_ < -sensitivity_)
    {
      prox_st->proximity.mode = "T";
    }
    else if (fa2_ > sensitivity_)
    {
      prox_st->proximity.mode = "R";
    }
    else
    {
      prox_st->proximity.mode = "0";
    }
    average_value_ = ea_ * raw + (1 - ea_) * average_value_;
  }
};  // end class Vcnl4040Mraa

class I2cSensorDriver
{
private:
  // Constants
  enum
  {
    // 7-bit unshifted I2C address of Multiplexer (PCA9547)
    MUX_ADDR = 0x70,
    // 7-bit unshifted I2C address of VCNL4040
    VCNL4040_ADDR = 0x60,
  };
  const int sensor_num_;
  mraa::I2c i2c_;
  Pca9547Mraa pca9547;
  std::vector<Vcnl4040Mraa> vcnl4040_array;

public:
  I2cSensorDriver(const int sensor_num = 1, const int i2c_bus = 0)
    : sensor_num_(sensor_num), i2c_(i2c_bus), vcnl4040_array(sensor_num, Vcnl4040Mraa())
  {
  }

  ~I2cSensorDriver()
  {
  }

  bool init()
  {
    pca9547.init(&i2c_);
    for (int sensor_no = 0; sensor_no < sensor_num_; sensor_no++)
    {
      pca9547.setChannel(MUX_ADDR, sensor_no);
      vcnl4040_array[sensor_no].init(&i2c_, VCNL4040_ADDR);
      vcnl4040_array[sensor_no].startSensing();
    }

    return true;
  }

  void getProximityArray(force_proximity_ros::ProximityArray* proximity_array)
  {
    // Fill necessary data of proximity
    proximity_array->proximities.clear();
    force_proximity_ros::ProximityStamped prox_st;
    for (int sensor_no = 0; sensor_no < sensor_num_; sensor_no++)
    {
      pca9547.setChannel(MUX_ADDR, sensor_no);
      vcnl4040_array[sensor_no].getProximityStamped(&prox_st);
      proximity_array->proximities.push_back(prox_st.proximity);
    }

    // Record time of reading last sensor
    proximity_array->header.stamp = prox_st.header.stamp;
  }
};  // end class I2cSensorDriver

class GripperLoop : public hardware_interface::RobotHW
{
private:
  ros::NodeHandle nh_;

  // Transmission loader
  transmission_interface::RobotTransmissions robot_transmissions_;
  boost::scoped_ptr<transmission_interface::TransmissionInterfaceLoader> transmission_loader_;

  // Actuator interface to transmission loader
  hardware_interface::ActuatorStateInterface actr_state_interface_;
  hardware_interface::PositionActuatorInterface pos_actr_interface_;

  // Actuator raw data
  const std::vector<std::string> actr_names_;
  std::vector<double> actr_curr_pos_;
  std::vector<double> actr_curr_vel_;
  std::vector<double> actr_curr_eff_;
  std::vector<double> actr_cmd_pos_;

  // Actuator interface to other nodes
  const std::vector<std::string> controller_names_;
  std::vector<ros::Publisher> actr_cmd_pub_;
  std::vector<ros::Subscriber> actr_state_sub_;
  std::map<std::string, dynamixel_msgs::JointState> received_actr_states_;
  std::vector<ros::ServiceClient> torque_enable_client_;
  std::map<std::string, double> pad_lim_conf_;

  // E-stop interface
  ros::Subscriber robot_state_sub_;
  ros::Publisher vacuum_pub_;
  bool is_gripper_enabled_;

  // Pressure sensor
  PressureSensorDriver pres_sen_;
  ros::Publisher pressure_pub_;

  // Flex sensor
  const std::vector<std::string> flex_names_;
  const std::vector<int> flex_thre_;
  const std::vector<double> wind_offset_flex_;
  FlexSensorDriver flex_sen_;
  std::vector<uint16_t> received_flex_;
  std::map<std::string, ros::Publisher> flex_pub_;

  // Proximity sensor
  I2cSensorDriver prox_sen_;
  ros::Publisher prox_pub_;

  // For multi-threaded spinning
  boost::shared_ptr<ros::AsyncSpinner> subscriber_spinner_;
  ros::CallbackQueue subscriber_queue_;

public:
  GripperLoop(const std::vector<std::string>& actr_names, const std::vector<std::string> controller_names,
              const std::vector<std::string>& flex_names, const std::vector<int>& flex_thre,
              const std::vector<double>& wind_offset_flex, const int prox_num,
              const std::map<std::string, double> pad_lim_conf)
    : actr_names_(actr_names)
    , controller_names_(controller_names)
    , flex_names_(flex_names)
    , flex_sen_(flex_names.size())
    , flex_thre_(flex_thre)
    , wind_offset_flex_(wind_offset_flex)
    , prox_sen_(prox_num)
    , pad_lim_conf_(pad_lim_conf)
    , is_gripper_enabled_(true)
  {
    // Register actuator interfaces to transmission loader
    actr_curr_pos_.resize(actr_names_.size(), 0);
    actr_curr_vel_.resize(actr_names_.size(), 0);
    actr_curr_eff_.resize(actr_names_.size(), 0);
    actr_cmd_pos_.resize(actr_names_.size(), 0);
    for (int i = 0; i < actr_names_.size(); i++)
    {
      hardware_interface::ActuatorStateHandle state_handle(actr_names_[i], &actr_curr_pos_[i], &actr_curr_vel_[i], &actr_curr_eff_[i]);
      actr_state_interface_.registerHandle(state_handle);

      hardware_interface::ActuatorHandle position_handle(state_handle, &actr_cmd_pos_[i]);
      pos_actr_interface_.registerHandle(position_handle);
    }
    registerInterface(&actr_state_interface_);
    registerInterface(&pos_actr_interface_);

    // Initialize transmission loader
    try
    {
      transmission_loader_.reset(new transmission_interface::TransmissionInterfaceLoader(this, &robot_transmissions_));
    }
    catch (const std::invalid_argument& ex)
    {
      ROS_ERROR_STREAM("Failed to create transmission interface loader. " << ex.what());
      return;
    }
    catch (const pluginlib::LibraryLoadException& ex)
    {
      ROS_ERROR_STREAM("Failed to create transmission interface loader. " << ex.what());
      return;
    }
    catch (...)
    {
      ROS_ERROR_STREAM("Failed to create transmission interface loader. ");
      return;
    }

    // Load URDF from parameter
    std::string urdf_string;
    ros::param::get("/robot_description", urdf_string);
    while (urdf_string.empty() && ros::ok())
    {
      ROS_INFO_STREAM_ONCE("Waiting for robot_description");
      ros::param::get("/robot_description", urdf_string);
      ros::Duration(0.1).sleep();
    }

    // Extract transmission infos from URDF
    transmission_interface::TransmissionParser parser;
    std::vector<transmission_interface::TransmissionInfo> infos;
    if (!parser.parse(urdf_string, infos))
    {
      ROS_ERROR("Error parsing URDF");
      return;
    }

    // Load transmissions composed of target actuators
    BOOST_FOREACH (const transmission_interface::TransmissionInfo& info, infos)
    {
      if (std::find(actr_names_.begin(), actr_names_.end(), info.actuators_[0].name_) != actr_names_.end())
      {
        BOOST_FOREACH (const transmission_interface::ActuatorInfo& actuator, info.actuators_)
        {
          if (std::find(actr_names_.begin(), actr_names_.end(), actuator.name_) == actr_names_.end())
          {
            ROS_ERROR_STREAM("Error loading transmission: " << info.name_);
            ROS_ERROR_STREAM("Cannot find " << actuator.name_ << " in target actuator list");
            return;
          }
        }
        if (!transmission_loader_->load(info))
        {
          ROS_ERROR_STREAM("Error loading transmission: " << info.name_);
          return;
        }
        else
        {
          ROS_INFO_STREAM("Loaded transmission: " << info.name_);
        }
      }
    }

    // Initialize actuator interfaces to other nodes
    BOOST_FOREACH (const std::string controller, controller_names_)
    {
      actr_cmd_pub_.push_back(nh_.advertise<std_msgs::Float64>("dxl/" + controller + "/command", 5));
      actr_state_sub_.push_back(
          nh_.subscribe("dxl/" + controller + "/state", 1, &GripperLoop::actrStateCallback, this));
      torque_enable_client_.push_back(
          nh_.serviceClient<dynamixel_controllers::TorqueEnable>("dxl/" + controller + "/torque_enable"));
    }

    // Initialize E-stop interfaces
    robot_state_sub_ = nh_.subscribe("/robot/state", 1, &GripperLoop::robotStateCallback, this);
    vacuum_pub_ = nh_.advertise<std_msgs::Bool>("vacuum", 10);

    // Initialize pressure sensor
    pres_sen_.init();
    pressure_pub_ = nh_.advertise<std_msgs::Float64>("pressure/state", 1);

    // Initialize proximity sensor
    prox_sen_.init();
    prox_pub_ = nh_.advertise<force_proximity_ros::ProximityArray>("proximity_array", 1);

    // Initialize flex sensor
    for (int i = 0; i < flex_names_.size(); i++)
    {
      flex_pub_[flex_names_[i]] = nh_.advertise<std_msgs::UInt16>("flex/" + flex_names_[i] + "/state", 1);
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
    // Get and publish pressure
    std_msgs::Float64 pressure;
    pressure.data = pres_sen_.getPressure();
    pressure_pub_.publish(pressure);

    // Get and publish flex
    flex_sen_.getFlex(&received_flex_);
    for (int i = 0; i < flex_names_.size(); i++)
    {
      std_msgs::UInt16 value;
      value.data = received_flex_[i];
      flex_pub_[flex_names_[i]].publish(value);
    }

    // Get and publish proximity
    force_proximity_ros::ProximityArray proximity_array;
    prox_sen_.getProximityArray(&proximity_array);
    prox_pub_.publish(proximity_array);

    // Update actuator current state
    for (int i = 0; i < actr_names_.size(); i++)
    {
      actr_curr_pos_[i] = received_actr_states_[actr_names_[i]].current_pos;
      actr_curr_vel_[i] = received_actr_states_[actr_names_[i]].velocity;

      // If fingers flex, add offset angle to finger tendon winder
      if (actr_names_[i].find("finger_tendon_winder") != std::string::npos)
      {
        for (int j = 0; j < flex_names_.size(); j++)
        {
          if (received_flex_[j] > flex_thre_[j])
          {
            actr_curr_pos_[i] -= wind_offset_flex_[j];
          }
        }
      }
    }

    // Propagate current actuator state to joints
    if (robot_transmissions_.get<transmission_interface::ActuatorToJointStateInterface>())
    {
      robot_transmissions_.get<transmission_interface::ActuatorToJointStateInterface>()->propagate();
    }
  }

  void write()
  {
    if (is_gripper_enabled_)
    {
      // Propagate joint commands to actuators
      if (robot_transmissions_.get<transmission_interface::JointToActuatorPositionInterface>())
      {
        robot_transmissions_.get<transmission_interface::JointToActuatorPositionInterface>()->propagate();
      }

      // Publish command to actuator
      for (int i = 0; i < actr_names_.size(); i++)
      {
        // If fingers flex, add offset angle to finger tendon winder
        if (actr_names_[i].find("finger_tendon_winder") != std::string::npos)
        {
          for (int j = 0; j < flex_names_.size(); j++)
          {
            if (received_flex_[j] > flex_thre_[j])
            {
              actr_cmd_pos_[i] += wind_offset_flex_[j];
            }
          }
        }

        // If prismatic joint is drawed back, limit vacuum pad joint to avoid collision
        if (actr_names_[i].find("vacuum_pad_tendon_winder") != std::string::npos)
        {
          int prismatic_idx =
              std::distance(controller_names_.begin(), std::find(controller_names_.begin(), controller_names_.end(),
                                                                 "prismatic_joint_controller"));
          if (actr_curr_pos_[prismatic_idx] < pad_lim_conf_["prismatic_joint_threshold"] &&
              actr_cmd_pos_[i] > pad_lim_conf_["upper_angle_limit"])
          {
            actr_cmd_pos_[i] = pad_lim_conf_["upper_angle_limit"];
          }
        }

        std_msgs::Float64 msg;
        msg.data = actr_cmd_pos_[i];
        actr_cmd_pub_[i].publish(msg);
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
      for (int i = 0; i < controller_names_.size(); i++)
      {
        torque_enable_client_[i].call(srv);
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

  void robotStateCallback(const baxter_core_msgs::AssemblyStateConstPtr& state)
  {
    is_gripper_enabled_ = state->enabled;
  }
};  // end class GripperLoop

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_v7_loop_node");

  std::vector<std::string> actr_names;
  std::vector<std::string> controller_names;
  std::vector<std::string> flex_names;
  std::vector<int> flex_thre;
  std::vector<double> wind_offset_flex;
  int prox_num;
  int rate_hz;
  std::map<std::string, double> pad_lim_conf;

  if (!(ros::param::get("~actuator_names", actr_names) && ros::param::get("~controller_names", controller_names) &&
        ros::param::get("~flex_names", flex_names) && ros::param::get("~flex_thresholds", flex_thre) &&
        ros::param::get("~wind_offset_flex", wind_offset_flex) && ros::param::get("~proximity_sensor_num", prox_num) &&
        ros::param::get("~control_rate", rate_hz) && ros::param::get("~vacuum_pad_motion_limit_config", pad_lim_conf)))
  {
    ROS_ERROR("Couldn't get necessary parameters");
    return 0;
  }

  GripperLoop gripper(actr_names, controller_names, flex_names, flex_thre, wind_offset_flex, prox_num, pad_lim_conf);
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
