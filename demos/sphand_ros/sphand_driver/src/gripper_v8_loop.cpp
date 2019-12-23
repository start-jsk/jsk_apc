// GPIO
#include "mraa.hpp"

// C++
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
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

// vl53l0x_mraa_ros
#include <vl53l0x_mraa_ros/vl53l0x_mraa.h>

// ROS msg and srv
#include <baxter_core_msgs/AssemblyState.h>
#include <dynamixel_controllers/TorqueEnable.h>
#include <dynamixel_msgs/JointState.h>
#include <force_proximity_ros/ProximityStamped.h>
#include <sphand_driver_msgs/ProximityStampedArray.h>
#include <sphand_driver_msgs/TurnOffSensors.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt16.h>
#include <std_srvs/SetBool.h>
#include <vl53l0x_mraa_ros/RangingMeasurementDataStamped.h>
#include <vl53l0x_mraa_ros/RangingMeasurementDataStampedArray.h>

// Temporary class to use /dev/spidev2.1 (see https://github.com/pazeshun/sphand_ros/issues/15)
// XXX: Don't substitute this class for mraa::Spi pointer as mraa::Spi's destructor doesn't have virtual
// https://qiita.com/ashdik/items/3cb3ee76137d176982f7
// http://cpp.aquariuscode.com/inheritance-use-case
// XXX: Because this is temporary, only write and transfer are implemented
class SpiWithCS1 : public mraa::Spi
{
private:
  const int bus_;
  const int cs_;
  mraa::Gpio cs1_pin_;

public:
  SpiWithCS1(int bus, int cs)
    : bus_(bus)
    , cs_(cs)
    , cs1_pin_(26)  // Pin number is MRAA Number in https://iotdk.intel.com/docs/master/mraa/up.html
    , mraa::Spi(bus, cs)
  {
    // Small delay to allow udev rules to execute. Without this, GPIO writing just after reboot fails.
    // See https://wiki.up-community.org/MRAA/UPM
    usleep(100 * 1000);
    cs1_pin_.dir(mraa::DIR_OUT);
    cs1_pin_.write(1);
  }

  ~SpiWithCS1()
  {
    cs1_pin_.write(1);
  }

  uint8_t* write(uint8_t* txBuf, int length)
  {
    // Initialize SPI_CS1
    mraa::Result res_gpio;
    if (cs_ == 1)
    {
      res_gpio = cs1_pin_.write(0);
    }
    else
    {
      res_gpio = cs1_pin_.write(1);
    }
    if (res_gpio != mraa::SUCCESS)
    {
      std::cout << "Error in mraa::Gpio::write in SpiWithCS1::write" << std::endl;
      return NULL;
    }

    // Main function
    uint8_t* res = mraa::Spi::write(txBuf, length);

    // Finalize SPI_CS1
    if (cs_ == 1)
    {
      if (cs1_pin_.write(1) != mraa::SUCCESS)
      {
        std::cout << "Error in mraa::Gpio::write in SpiWithCS1::write" << std::endl;
        return NULL;
      }
    }

    return res;
  }

  mraa::Result transfer(uint8_t* txBuf, uint8_t* rxBuf, int length)
  {
    // Initialize SPI_CS1
    mraa::Result res_gpio;
    if (cs_ == 1)
    {
      res_gpio = cs1_pin_.write(0);
    }
    else
    {
      res_gpio = cs1_pin_.write(1);
    }
    if (res_gpio != mraa::SUCCESS)
    {
      std::cout << "Error in mraa::Gpio::write in SpiWithCS1::transfer" << std::endl;
      return res_gpio;
    }

    // Main function
    mraa::Result res = mraa::Spi::transfer(txBuf, rxBuf, length);

    // Finalize SPI_CS1
    if (cs_ == 1)
    {
      res_gpio = cs1_pin_.write(1);
      if (res_gpio != mraa::SUCCESS)
      {
        std::cout << "Error in mraa::Gpio::write in SpiWithCS1::transfer" << std::endl;
        return res_gpio;
      }
    }

    return res;
  }
};

class PressureSensorDriver
{
private:
  SpiWithCS1 spi_;
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
    uint8_t* recv = spi_.write(tx, 2);
    if (recv == NULL)
    {
      free(recv);
      throw std::invalid_argument("Error in SpiWithCS1::write");
    }
    free(recv);
    tx[0] = 0xF4 & 0x7F;
    tx[1] = 0x27;
    recv = spi_.write(tx, 2);
    if (recv == NULL)
    {
      free(recv);
      throw std::invalid_argument("Error in SpiWithCS1::write");
    }
    free(recv);
  }

  void readTrim()
  {
    uint8_t tx[25] = {};
    uint8_t rx[25];
    tx[0] = 0x88 | 0x80;
    if (spi_.transfer(tx, rx, 25) != mraa::SUCCESS)
    {
      throw std::invalid_argument("Error in SpiWithCS1::transfer");
    }

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

  void init()
  {
    initBME();
    readTrim();
  }

  void readRawPressureAndTemperature(uint32_t* pres_raw, uint32_t* temp_raw)
  {
    uint8_t data[8];
    uint8_t tx[9] = {};
    uint8_t rx[9];
    tx[0] = 0xF7 | 0x80;
    if (spi_.transfer(tx, rx, 9) != mraa::SUCCESS)
    {
      throw std::invalid_argument("Error in SpiWithCS1::transfer");
    }
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
  SpiWithCS1 spi_;
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
      if (spi_.transfer(tx, rx, 3) != mraa::SUCCESS)
      {
        throw std::invalid_argument("Error in SpiWithCS1::transfer");
      }
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

  bool setChannel(const uint8_t mux_addr, const int8_t ch)
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
      return false;
    }

    mraa::Result result = i2c_->address(mux_addr);
    if (result != mraa::SUCCESS)
    {
      return false;
    }
    return (i2c_->writeByte(tx) == mraa::SUCCESS);
  }
};  // end class Pca9547Mraa

class Pca9546Mraa
{
private:
  mraa::I2c* i2c_;

public:
  Pca9546Mraa()
  {
  }

  ~Pca9546Mraa()
  {
  }

  void init(mraa::I2c* i2c)
  {
    i2c_ = i2c;
  }

  bool setChannel(const uint8_t mux_addr, const int8_t ch)
  {
    uint8_t tx;

    if (ch == -1)
    {
      // No channel selected
      tx = 0;
    }
    else if (0 <= ch && ch <= 3)
    {
      // Channel 0~3 selected
      tx = 1 << (uint8_t)ch;
    }
    else
    {
      ROS_ERROR("I2C Multiplexer PCA9546 has no channel %d", ch);
      return false;
    }

    mraa::Result result = i2c_->address(mux_addr);
    if (result != mraa::SUCCESS)
    {
      return false;
    }
    return (i2c_->writeByte(tx) == mraa::SUCCESS);
  }
};  // end class Pca9546Mraa

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
    ID_L = 0x0C,
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
  mraa::Result readCommandRegister(const uint8_t command_code, uint16_t* data)
  {
    mraa::Result result = i2c_->address(i2c_addr_);
    if (result != mraa::SUCCESS)
    {
      return result;
    }
    try
    {
      *data = i2c_->readWordReg(command_code);
    }
    catch (std::invalid_argument& err)
    {
      return mraa::ERROR_UNSPECIFIED;
    }
    return mraa::SUCCESS;
  }

  // Write to two Command Registers of VCNL4040
  mraa::Result writeCommandRegister(const uint8_t command_code, const uint8_t low_data, const uint8_t high_data)
  {
    uint16_t data = ((uint16_t)high_data << 8) | low_data;

    mraa::Result result = i2c_->address(i2c_addr_);
    if (result != mraa::SUCCESS)
    {
      return result;
    }
    return i2c_->writeWordReg(command_code, data);
  }

  bool ping()
  {
    uint16_t data;
    return (readCommandRegister(ID_L, &data) == mraa::SUCCESS && data == 0x0186);
  }

  // Configure VCNL4040
  bool init(mraa::I2c* i2c, const uint8_t i2c_addr)
  {
    i2c_ = i2c;
    i2c_addr_ = i2c_addr;

    if (!ping())
    {
      return false;
    }

    // Set PS_CONF3 and PS_MS
    uint8_t conf3 = 0x00;
    // uint8_t ms = 0x00;  // IR LED current to 50mA
    // uint8_t ms = 0x01;  // IR LED current to 75mA
    // uint8_t ms = 0x02;  // IR LED current to 100mA
    uint8_t ms = 0x06;  // IR LED current to 180mA
    // uint8_t ms = 0x07;  // IR LED current to 200mA
    return (writeCommandRegister(PS_CONF3, conf3, ms) == mraa::SUCCESS);
  }

  bool startSensing()
  {
    // Clear PS_SD to turn on proximity sensing
    // uint8_t conf1 = 0x00;  // Clear PS_SD bit to begin reading
    uint8_t conf1 = 0x0E;  // Integrate 8T, Clear PS_SD bit to begin reading
    // uint8_t conf2 = 0x00;  // Clear PS to 12-bit
    uint8_t conf2 = 0x08;  // Set PS to 16-bit
    return (writeCommandRegister(PS_CONF1, conf1, conf2) == mraa::SUCCESS);
  }

  bool stopSensing()
  {
    // Set PS_SD to turn off proximity sensing
    uint8_t conf1 = 0x01;  // Set PS_SD bit to stop reading
    uint8_t conf2 = 0x00;
    return (writeCommandRegister(PS_CONF1, conf1, conf2) == mraa::SUCCESS);
  }

  bool getRawProximity(uint16_t* data)
  {
    return (readCommandRegister(PS_DATA_L, data) == mraa::SUCCESS);
  }

  bool getProximityStamped(force_proximity_ros::ProximityStamped* prox_st)
  {
    uint16_t raw;
    if (!getRawProximity(&raw))
    {
      return false;
    }
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

    return true;
  }
};  // end class Vcnl4040Mraa

class I2cSensorDriver
{
private:
  // Constants
  enum
  {
    // 7-bit unshifted I2C address of VCNL4040
    VCNL4040_ADDR = 0x60,
    // 7-bit unshifted I2C address of VL53L0X
    VL53L0X_ADDR = 0x29,
    // Max number of VL53L0X readable in a cycle
    MAX_TOF_IN_A_CYCLE = 4,
  };
  std::vector<std::vector<std::map<std::string, int> > > i2c_mux_;
  mraa::I2c i2c_;
  Pca9547Mraa pca9547_;
  Pca9546Mraa pca9546_;
  std::vector<Vcnl4040Mraa> vcnl4040_array_;
  boost::scoped_array<Vl53l0xMraa> vl53l0x_array_;
  // FIXME: If using std::vector, node is shut down when Vl53l0xMraa::begin() is called
  vl53l0x_mraa_ros::RangingMeasurementDataStampedArray current_tof_array_;
  std::vector<uint64_t> i_turned_off_;
  std::vector<uint64_t> tof_turned_off_;
  std::vector<std::vector<uint64_t> > tof_read_order_;
  int tof_read_idx_;
  bool is_tof_start_stop_req_;

public:
  I2cSensorDriver(const std::vector<std::vector<std::map<std::string, int> > > i2c_mux, const int i2c_bus = 0)
    : i2c_mux_(i2c_mux)
    , i2c_(i2c_bus)
    , vcnl4040_array_(i2c_mux.size(), Vcnl4040Mraa())
    , vl53l0x_array_(new Vl53l0xMraa[i2c_mux.size()])
  {
    // I2C is slow in old UP Board (https://forum.up-community.org/discussion/2402/i2c-400khz-and-pullup-resistors),
    // but this may be fixed in new UP Board (https://github.com/pazeshun/sphand_ros/issues/14#issuecomment-543105693)
    i2c_.frequency(mraa::I2C_FAST);
    current_tof_array_.array.resize(i2c_mux_.size());
    // Initialize tof_read_order_
    turnOffTof(std::vector<uint64_t>());
  }

  ~I2cSensorDriver()
  {
  }

  void setMultiplexers(std::vector<std::map<std::string, int> >& mux_infos,
                       std::vector<std::map<std::string, int> > prev_infos = std::vector<std::map<std::string, int> >())
  {
    bool res;
    for (int mux_no = 0; mux_no < mux_infos.size(); mux_no++)
    {
      std::map<std::string, int>& mux_info = mux_infos[mux_no];
      if (prev_infos.size() > mux_no && prev_infos[mux_no] == mux_info)
      {
        continue;
      }
      if (mux_info["type"] == 9547)
      {
        res = pca9547_.setChannel(mux_info["address"], mux_info["channel"]);
      }
      else if (mux_info["type"] == 9546)
      {
        res = pca9546_.setChannel(mux_info["address"], mux_info["channel"]);
      }
      if (!res)
      {
        std::ostringstream ss;
        ss << "Cannot set to channel " << mux_info["channel"] << " on Multiplexer " << std::hex << mux_info["address"];
        throw std::invalid_argument(ss.str());
      }
    }
  }

  void init()
  {
    pca9547_.init(&i2c_);
    pca9546_.init(&i2c_);
    for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
    {
      if (sensor_no == 0)
      {
        setMultiplexers(i2c_mux_[sensor_no]);
      }
      else
      {
        setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[sensor_no - 1]);
      }

      // Initialize VCNL4040
      if (!vcnl4040_array_[sensor_no].init(&i2c_, VCNL4040_ADDR))
      {
        std::ostringstream ss;
        ss << "Failed to init VCNL4040 No. " << sensor_no;
        throw std::invalid_argument(ss.str());
      }

      // Boot VL53L0X
      if (!vl53l0x_array_[sensor_no].begin(&i2c_, false, VL53L0X_ADDR))
      {
        ROS_ERROR("Failed to boot VL53L0X No. %d, try to reset it", sensor_no);
        // Reset VL53L0X
        if (!vl53l0x_array_[sensor_no].resetDevice())
        {
          std::ostringstream ss;
          ss << "Failed to reset VL53L0X No. " << sensor_no;
          throw std::invalid_argument(ss.str());
        }
        ROS_INFO("Reset VL53L0X No.%d", sensor_no);
        // Reboot VL53L0X
        if (!vl53l0x_array_[sensor_no].begin(&i2c_, false, VL53L0X_ADDR))
        {
          std::ostringstream ss;
          ss << "Failed to reboot VL53L0X No. " << sensor_no;
          throw std::invalid_argument(ss.str());
        }
        ROS_INFO("Rebooted VL53L0X No.%d", sensor_no);
      }
      if (!vl53l0x_array_[sensor_no].setMeasurementTimingBudget(20000))
      {
        std::ostringstream ss;
        ss << "Failed to set MeasurementTimingBudget in VL53L0X No. " << sensor_no;
        throw std::invalid_argument(ss.str());
      }
      if (vl53l0x_array_[sensor_no].setDeviceModeToContinuousRanging() != VL53L0X_ERROR_NONE)
      {
        std::ostringstream ss;
        ss << "Failed to set device mode to continuous ranging in VL53L0X No. " << sensor_no;
        throw std::invalid_argument(ss.str());
      }
      is_tof_start_stop_req_ = true;
    }
  }

  void cleanup()
  {
    // Stop ToF sensing
    int prev_sen_no = -1;
    for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
    {
      if (prev_sen_no < 0)
      {
        setMultiplexers(i2c_mux_[sensor_no]);
      }
      else
      {
        setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[prev_sen_no]);
      }
      if (vl53l0x_array_[sensor_no].stopMeasurement() != VL53L0X_ERROR_NONE ||
          vl53l0x_array_[sensor_no].waitStopCompleted() != VL53L0X_ERROR_NONE)
      {
        ROS_ERROR("Failed to stop measurement in VL53L0X No. %d", sensor_no);
      }
      prev_sen_no = sensor_no;
    }
  }

  void resetTof()
  {
    for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
    {
      if (sensor_no == 0)
      {
        setMultiplexers(i2c_mux_[sensor_no]);
      }
      else
      {
        setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[sensor_no - 1]);
      }
      if (!vl53l0x_array_[sensor_no].resetDevice())
      {
        std::ostringstream ss;
        ss << "Failed to reset VL53L0X No. " << sensor_no;
        throw std::invalid_argument(ss.str());
      }
    }
  }

  void getProximityArrays(sphand_driver_msgs::ProximityStampedArray* intensity_array,
                          vl53l0x_mraa_ros::RangingMeasurementDataStampedArray* tof_array)
  {
    bool is_i2c_error = false;
    do
    {
      try
      {
        if (is_i2c_error)
        {
          ros::Duration(0.5).sleep();
          ROS_INFO("Try to re-initialize all I2C sensors");
          // If begin() in init() has once been called against unconnected VL53L0X,
          // begin() sometimes fails to reconnect with VL53L0X.
          // range_status in the sensor response becomes unexpected value.
          // Calling resetTof() before begin() prevents it.
          resetTof();
          init();
          ROS_INFO("Re-initialized all I2C sensors");
        }

        // Start & stop ToF sensing
        int prev_sen_no = -1;
        if (is_tof_start_stop_req_)
        {
          for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
          {
            if (prev_sen_no < 0)
            {
              setMultiplexers(i2c_mux_[sensor_no]);
            }
            else
            {
              setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[prev_sen_no]);
            }

            if (std::find(tof_turned_off_.begin(), tof_turned_off_.end(), sensor_no) == tof_turned_off_.end())
            {
              if (vl53l0x_array_[sensor_no].startMeasurement() != VL53L0X_ERROR_NONE)
              {
                std::ostringstream ss;
                ss << "Failed to start measurement in VL53L0X No. " << sensor_no;
                throw std::invalid_argument(ss.str());
              }
            }
            else
            {
              if (vl53l0x_array_[sensor_no].stopMeasurement() != VL53L0X_ERROR_NONE ||
                  vl53l0x_array_[sensor_no].waitStopCompleted() != VL53L0X_ERROR_NONE)
              {
                std::ostringstream ss;
                ss << "Failed to stop measurement in VL53L0X No. " << sensor_no;
                throw std::invalid_argument(ss.str());
              }
            }
            prev_sen_no = sensor_no;
          }
          is_tof_start_stop_req_ = false;
        }

        // Start intensity sensing & wait for sensor values ready
        ros::Time sensor_ready_tm = ros::Time::now() + ros::Duration(0.005);
        for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
        {
          if (std::find(i_turned_off_.begin(), i_turned_off_.end(), sensor_no) != i_turned_off_.end())
          {
            continue;
          }
          if (prev_sen_no < 0)
          {
            setMultiplexers(i2c_mux_[sensor_no]);
          }
          else
          {
            setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[prev_sen_no]);
          }
          if (!vcnl4040_array_[sensor_no].startSensing())
          {
            std::ostringstream ss;
            ss << "Failed to start sensing on VCNL4040 No. " << sensor_no;
            throw std::invalid_argument(ss.str());
          }
          prev_sen_no = sensor_no;
        }
        ros::Duration sensor_wait_dur = sensor_ready_tm - ros::Time::now();
        if (sensor_wait_dur.toSec() > 0)
        {
          sensor_wait_dur.sleep();
        }

        // Fill necessary data of proximity
        intensity_array->proximities.clear();
        tof_array->array.clear();
        force_proximity_ros::ProximityStamped intensity_st;
        VL53L0X_RangingMeasurementData_t tof_data;
        vl53l0x_mraa_ros::RangingMeasurementDataStamped tof_data_st;
        for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
        {
          if (sensor_no == 0)
          {
            setMultiplexers(i2c_mux_[sensor_no]);
          }
          else
          {
            setMultiplexers(i2c_mux_[sensor_no], i2c_mux_[sensor_no - 1]);
          }

          // Intensity
          if (!vcnl4040_array_[sensor_no].getProximityStamped(&intensity_st))
          {
            std::ostringstream ss;
            ss << "Failed to read data of VCNL4040 No. " << sensor_no;
            throw std::invalid_argument(ss.str());
          }
          if (!vcnl4040_array_[sensor_no].stopSensing())
          {
            std::ostringstream ss;
            ss << "Failed to stop sensing on VCNL4040 No. " << sensor_no;
            throw std::invalid_argument(ss.str());
          }
          intensity_array->proximities.push_back(intensity_st);

          // ToF
          const std::vector<uint64_t>& tof_read_target = tof_read_order_[tof_read_idx_];
          if (std::find(tof_read_target.begin(), tof_read_target.end(), sensor_no) != tof_read_target.end())
          {
            if (vl53l0x_array_[sensor_no].measurementPollForCompletion() != VL53L0X_ERROR_NONE)
            {
              std::ostringstream ss;
              ss << "Failed to read data of VL53L0X No. " << sensor_no;
              throw std::invalid_argument(ss.str());
            }
            if (vl53l0x_array_[sensor_no].getRangingMeasurementData(&tof_data) != VL53L0X_ERROR_NONE)
            {
              std::ostringstream ss;
              ss << "Failed to read data of VL53L0X No. " << sensor_no;
              throw std::invalid_argument(ss.str());
            }
            if (tof_data.RangeStatus > 5)
            {
              std::ostringstream ss;
              ss << "Odd RangeStatus: " << tof_data.RangeStatus << " from VL53L0X No. " << sensor_no;
              throw std::invalid_argument(ss.str());
            }
            tof_data_st.header.stamp = ros::Time::now();
            tof_data_st.data.timestamp = tof_data.TimeStamp;
            tof_data_st.data.measurement_time_usec = tof_data.MeasurementTimeUsec;
            tof_data_st.data.range_millimeter = tof_data.RangeMilliMeter;
            tof_data_st.data.range_d_max_millimeter = tof_data.RangeDMaxMilliMeter;
            tof_data_st.data.signal_rate_rtn_megacps = tof_data.SignalRateRtnMegaCps;
            tof_data_st.data.ambient_rate_rtn_megacps = tof_data.AmbientRateRtnMegaCps;
            tof_data_st.data.effective_spad_rtn_count = tof_data.EffectiveSpadRtnCount;
            tof_data_st.data.zone_id = tof_data.ZoneId;
            tof_data_st.data.range_fractional_part = tof_data.RangeFractionalPart;
            tof_data_st.data.range_status = tof_data.RangeStatus;
            current_tof_array_.array[sensor_no] = tof_data_st;
          }
          tof_array->array.push_back(current_tof_array_.array[sensor_no]);
        }
        tof_read_idx_++;
        if (tof_read_idx_ >= tof_read_order_.size())
        {
          tof_read_idx_ = 0;
        }

        // Record time of reading last sensor
        intensity_array->header.stamp = intensity_st.header.stamp;
        tof_array->header.stamp = tof_data_st.header.stamp;
        is_i2c_error = false;
      }
      catch (std::invalid_argument& err)
      {
        ROS_ERROR("%s", err.what());
        is_i2c_error = true;
      }
    } while (is_i2c_error);
  }

  bool turnOffIntensity(const std::vector<uint64_t> i_turned_off)
  {
    for (int i = 0; i < i_turned_off.size(); i++)
    {
      if (i_turned_off[i] < 0 || i2c_mux_.size() <= i_turned_off[i])
      {
        return false;
      }
    }
    i_turned_off_ = i_turned_off;
    return true;
  }

  bool turnOffTof(const std::vector<uint64_t> tof_turned_off)
  {
    for (int i = 0; i < tof_turned_off.size(); i++)
    {
      if (tof_turned_off[i] < 0 || i2c_mux_.size() <= tof_turned_off[i])
      {
        return false;
      }
    }
    tof_turned_off_ = tof_turned_off;

    // Prepare tof_read_order_
    tof_read_order_.clear();
    tof_read_order_.push_back(std::vector<uint64_t>());
    for (int sensor_no = 0; sensor_no < i2c_mux_.size(); sensor_no++)
    {
      if (std::find(tof_turned_off_.begin(), tof_turned_off_.end(), sensor_no) == tof_turned_off_.end())
      {
        tof_read_order_[0].push_back(sensor_no);
      }
    }
    // Each element of tof_read_order_ has size no more than MAX_TOF_IN_A_CYCLE
    while (tof_read_order_.back().size() > MAX_TOF_IN_A_CYCLE)
    {
      std::vector<uint64_t> last = tof_read_order_.back();
      tof_read_order_.pop_back();
      std::vector<uint64_t> first_four(last.begin(), last.begin() + MAX_TOF_IN_A_CYCLE);
      tof_read_order_.push_back(first_four);
      std::vector<uint64_t> rest(last.begin() + MAX_TOF_IN_A_CYCLE, last.end());
      tof_read_order_.push_back(rest);
    }
    tof_read_idx_ = 0;
    is_tof_start_stop_req_ = true;

    return true;
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
  bool is_pad_limited_;

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
  bool is_flex_reflex_enabled_;
  ros::ServiceServer enable_flex_reflex_srv_;

  // Proximity sensor
  I2cSensorDriver i2c_sen_;
  ros::Publisher intensity_prox_pub_;
  ros::Publisher tof_prox_pub_;
  int cnt_for_i2c_init_;
  enum
  {
    I2C_INIT_SKIP_CNT = 50,
  };
  ros::ServiceServer turn_off_i_srv_;
  ros::ServiceServer turn_off_tof_srv_;

  // For multi-threaded spinning
  boost::shared_ptr<ros::AsyncSpinner> subscriber_spinner_;
  ros::CallbackQueue subscriber_queue_;

public:
  GripperLoop(const std::vector<std::string>& actr_names, const std::vector<std::string> controller_names,
              const std::vector<std::string>& flex_names, const std::vector<int>& flex_thre,
              const std::vector<double>& wind_offset_flex, const std::map<std::string, double> pad_lim_conf,
              const std::vector<std::vector<std::map<std::string, int> > > i2c_mux)
    : actr_names_(actr_names)
    , controller_names_(controller_names)
    , flex_names_(flex_names)
    , flex_sen_(flex_names.size())
    , flex_thre_(flex_thre)
    , wind_offset_flex_(wind_offset_flex)
    , i2c_sen_(i2c_mux)
    , pad_lim_conf_(pad_lim_conf)
    , is_pad_limited_(false)
    , is_gripper_enabled_(true)
    , is_flex_reflex_enabled_(true)
    , cnt_for_i2c_init_(0)
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
    // FIXME: Initializing sensor here spends much time and causes the following error from dynamixel controller:
    //        ValueError: cannot convert float NaN to integer
    // i2c_sen_.init();
    intensity_prox_pub_ = nh_.advertise<sphand_driver_msgs::ProximityStampedArray>("intensity_proximities", 1);
    tof_prox_pub_ = nh_.advertise<vl53l0x_mraa_ros::RangingMeasurementDataStampedArray>("tof_proximities", 1);
    turn_off_i_srv_ = nh_.advertiseService("turn_off_intensity", &GripperLoop::turnOffIntensity, this);
    turn_off_tof_srv_ = nh_.advertiseService("turn_off_tof", &GripperLoop::turnOffTof, this);

    // Initialize flex sensor
    for (int i = 0; i < flex_names_.size(); i++)
    {
      flex_pub_[flex_names_[i]] = nh_.advertise<std_msgs::UInt16>("flex/" + flex_names_[i] + "/state", 1);
    }
    enable_flex_reflex_srv_ = nh_.advertiseService("enable_flex_reflex", &GripperLoop::enableFlexReflex, this);

    // Start spinning
    nh_.setCallbackQueue(&subscriber_queue_);
    subscriber_spinner_.reset(new ros::AsyncSpinner(1, &subscriber_queue_));
    subscriber_spinner_->start();
  }

  void cleanup()
  {
    i2c_sen_.cleanup();
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
    sphand_driver_msgs::ProximityStampedArray intensity_array;
    vl53l0x_mraa_ros::RangingMeasurementDataStampedArray tof_array;
    // FIXME: Temporarily initialize sensor here to avoid dynamixel error
    //        Some skips are needed
    if (cnt_for_i2c_init_ >= I2C_INIT_SKIP_CNT)
    {
      if (cnt_for_i2c_init_ == I2C_INIT_SKIP_CNT)
      {
        try
        {
          i2c_sen_.init();
        }
        catch (std::invalid_argument& err)
        {
          throw std::invalid_argument(std::string(err.what()) + ". Check I2C cable connection");
        }
      }
      i2c_sen_.getProximityArrays(&intensity_array, &tof_array);
      intensity_prox_pub_.publish(intensity_array);
      tof_prox_pub_.publish(tof_array);
    }
    if (cnt_for_i2c_init_ <= I2C_INIT_SKIP_CNT)
    {
      cnt_for_i2c_init_++;
    }

    // Update actuator current state
    for (int i = 0; i < actr_names_.size(); i++)
    {
      actr_curr_pos_[i] = received_actr_states_[actr_names_[i]].current_pos;
      actr_curr_vel_[i] = received_actr_states_[actr_names_[i]].velocity;

      // If fingers flex, add offset angle to finger tendon winder.
      // If this addition works when is_flex_reflex_enabled_ is false, fingers suddenly open when motion is cancelled
      if (is_flex_reflex_enabled_ && actr_names_[i].find("finger_tendon_winder") != std::string::npos)
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
        if (is_flex_reflex_enabled_ && actr_names_[i].find("finger_tendon_winder") != std::string::npos)
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
            if (!is_pad_limited_)
            {
              ROS_WARN("Vacuum pad joint becomes to be limited: command to pad joint motor: %lf, current angle of "
                       "prismatic joint motor: %lf",
                       actr_cmd_pos_[i], actr_curr_pos_[prismatic_idx]);
            }
            actr_cmd_pos_[i] = pad_lim_conf_["upper_angle_limit"];
            is_pad_limited_ = true;
          }
          else if (is_pad_limited_)
          {
            ROS_WARN("Vacuum pad joint becomes to be unlimited: command to pad joint motor: %lf, current angle of "
                     "prismatic joint motor: %lf",
                     actr_cmd_pos_[i], actr_curr_pos_[prismatic_idx]);
            is_pad_limited_ = false;
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

  bool turnOffIntensity(sphand_driver_msgs::TurnOffSensors::Request& req,
                        sphand_driver_msgs::TurnOffSensors::Response& res)
  {
    if (i2c_sen_.turnOffIntensity(req.sensor_ids))
    {
      res.success = true;
    }
    else
    {
      res.success = false;
      res.message = "Specified sensor id doesn't exist";
    }
    return true;
  }

  bool turnOffTof(sphand_driver_msgs::TurnOffSensors::Request& req,
                  sphand_driver_msgs::TurnOffSensors::Response& res)
  {
    if (i2c_sen_.turnOffTof(req.sensor_ids))
    {
      res.success = true;
    }
    else
    {
      res.success = false;
      res.message = "Specified sensor id doesn't exist";
    }
    return true;
  }

  bool enableFlexReflex(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res)
  {
    is_flex_reflex_enabled_ = req.data;
    res.success = true;
    return true;
  }
};  // end class GripperLoop

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_v8_loop_node");

  std::vector<std::string> actr_names;
  std::vector<std::string> controller_names;
  std::vector<std::string> flex_names;
  std::vector<int> flex_thre;
  std::vector<double> wind_offset_flex;
  int rate_hz;
  std::map<std::string, double> pad_lim_conf;
  XmlRpc::XmlRpcValue i2c_mux_param;

  if (!(ros::param::get("~actuator_names", actr_names) && ros::param::get("~controller_names", controller_names) &&
        ros::param::get("~flex_names", flex_names) && ros::param::get("~flex_thresholds", flex_thre) &&
        ros::param::get("~wind_offset_flex", wind_offset_flex) && ros::param::get("~control_rate", rate_hz) &&
        ros::param::get("~vacuum_pad_motion_limit_config", pad_lim_conf) &&
        ros::param::get("~i2c_multiplexers_to_access_each_sensor", i2c_mux_param)))
  {
    ROS_ERROR("Couldn't get necessary parameters");
    return 0;
  }

  // Parse i2c_mux_param
  if (i2c_mux_param.getType() != XmlRpc::XmlRpcValue::TypeArray)
  {
    ROS_ERROR("i2c_multiplexers_to_access_each_sensor is not array");
    return 0;
  }
  std::vector<std::vector<std::map<std::string, int> > > i2c_mux;
  for (int i = 0; i < i2c_mux_param.size(); i++)
  {
    if (i2c_mux_param[i].getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      ROS_ERROR("i2c_multiplexers_to_access_each_sensor[%d] is not array", i);
      return 0;
    }
    std::vector<std::map<std::string, int> > mux_infos;
    for (int j = 0; j < i2c_mux_param[i].size(); j++)
    {
      if (i2c_mux_param[i][j].getType() != XmlRpc::XmlRpcValue::TypeStruct)
      {
        ROS_ERROR("i2c_multiplexers_to_access_each_sensor[%d][%d] is not dictionary", i, j);
        return 0;
      }
      XmlRpc::XmlRpcValue& mux_info_param = i2c_mux_param[i][j];
      std::map<std::string, int> mux_info;
      for (std::map<std::string, XmlRpc::XmlRpcValue>::iterator itr = mux_info_param.begin();
           itr != mux_info_param.end(); ++itr)
      {
        mux_info[itr->first] = itr->second;
      }
      mux_infos.push_back(mux_info);
    }
    i2c_mux.push_back(mux_infos);
  }

  GripperLoop gripper(actr_names, controller_names, flex_names, flex_thre, wind_offset_flex, pad_lim_conf, i2c_mux);
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
