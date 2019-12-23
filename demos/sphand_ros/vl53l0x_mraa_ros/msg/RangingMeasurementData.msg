# Based on VL53L0X_RangingMeasurementData_t in include/vl53l0x_mraa_ros/vl53l0x_def.h

# Currently not implemented
# cf. src/vl53l0x_mraa/core/vl53l0x_api.cpp
uint32 timestamp

uint32 measurement_time_usec

# Mainly this value is used
uint16 range_millimeter

uint16 range_d_max_millimeter
uint32 signal_rate_rtn_megacps
uint32 ambient_rate_rtn_megacps
uint16 effective_spad_rtn_count

# Currently not implemented
# cf. src/vl53l0x_mraa/core/vl53l0x_api.cpp
uint8 zone_id

uint8 range_fractional_part

uint8 range_status
uint8 RANGE_VALID = 0
uint8 SIGMA_FAIL = 1
uint8 SIGNAL_FAIL = 2
uint8 MIN_RANGE_FAIL = 3
uint8 PHASE_FAIL = 4
uint8 HARDWARE_FAIL = 5
uint8 NO_UPDATE = 255
