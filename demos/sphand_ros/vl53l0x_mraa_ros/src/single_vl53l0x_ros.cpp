#include "vl53l0x_mraa_ros/vl53l0x_mraa.h"
#include <ros/ros.h>
#include <vl53l0x_mraa_ros/RangingMeasurementDataStamped.h>

Vl53l0xMraa lox = Vl53l0xMraa();

int main(int argc, char** argv)
{
  ros::init(argc, argv, "single_vl53l0x_ros");
  ros::NodeHandle nh("~");
  ros::Publisher pub = nh.advertise<vl53l0x_mraa_ros::RangingMeasurementDataStamped>("data_stamped", 10);

  ROS_INFO("VL53L0X test");
  mraa::I2c i2c(0);
  if (!lox.begin(&i2c))
  {
    ROS_FATAL("Failed to boot VL53L0X");
    return 0;
  }
  ROS_INFO("VL53L0X API Simple Ranging example");
  ros::Rate rate(10);
  while (ros::ok())
  {
    VL53L0X_RangingMeasurementData_t measure;
    lox.getSingleRangingMeasurement(&measure, false);  // pass in 'true' to get debug data printout!
    if (measure.RangeStatus != 4)
    {
      // phase failures have incorrect data
      ROS_INFO("Distance (mm): %d", measure.RangeMilliMeter);
    }
    else
    {
      ROS_INFO("Out of range");
    }

    // Publish result
    vl53l0x_mraa_ros::RangingMeasurementDataStamped data_st;
    data_st.data.timestamp = measure.TimeStamp;
    data_st.data.measurement_time_usec = measure.MeasurementTimeUsec;
    data_st.data.range_millimeter = measure.RangeMilliMeter;
    data_st.data.range_d_max_millimeter = measure.RangeDMaxMilliMeter;
    data_st.data.signal_rate_rtn_megacps = measure.SignalRateRtnMegaCps;
    data_st.data.ambient_rate_rtn_megacps = measure.AmbientRateRtnMegaCps;
    data_st.data.effective_spad_rtn_count = measure.EffectiveSpadRtnCount;
    data_st.data.zone_id = measure.ZoneId;
    data_st.data.range_fractional_part = measure.RangeFractionalPart;
    data_st.data.range_status = measure.RangeStatus;
    data_st.header.stamp = ros::Time::now();
    pub.publish(data_st);

    rate.sleep();
  }

  return 0;
}
