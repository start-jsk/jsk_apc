// Based on Adafruit_VL53L0X.cpp in https://github.com/adafruit/Adafruit_VL53L0X

#include "vl53l0x_mraa_ros/vl53l0x_mraa.h"

#define VERSION_REQUIRED_MAJOR  1 ///< Required sensor major version
#define VERSION_REQUIRED_MINOR  0 ///< Required sensor minor version
#define VERSION_REQUIRED_BUILD  1 ///< Required sensor build

#define STR_HELPER( x ) #x ///< a string helper
#define STR( x )        STR_HELPER(x) ///< string helper wrapper

/**************************************************************************/
/*! 
    @brief  Setups the I2C interface and hardware
    @param  i2c_addr Optional I2C address the sensor can be found on. Default is 0x29
    @param debug Optional debug flag. If true, debug information will print out via stdout during setup. Defaults to false.
    @param  i2c Optional I2C bus the sensor is located on. Default is Wire
    @returns True if device is set up, false on any failure
*/
/**************************************************************************/
bool Vl53l0xMraa::begin(mraa::I2c *i2c, bool debug, uint8_t i2c_addr) {
  int32_t   status_int;
  int32_t   init_done         = 0;

  uint32_t  refSpadCount;
  uint8_t   isApertureSpads;
  uint8_t   VhvSettings;
  uint8_t   PhaseCal;

  // Initialize Comms
  pMyDevice->I2cDevAddr      =  VL53L0X_I2C_ADDR;  // default
  pMyDevice->comms_type      =  1;
  pMyDevice->comms_speed_khz =  400;
  pMyDevice->i2c = i2c;

  // VL53L0X_i2c_init();

  // unclear if this is even needed:
  if( VL53L0X_IMPLEMENTATION_VER_MAJOR != VERSION_REQUIRED_MAJOR ||
      VL53L0X_IMPLEMENTATION_VER_MINOR != VERSION_REQUIRED_MINOR ||
      VL53L0X_IMPLEMENTATION_VER_SUB != VERSION_REQUIRED_BUILD )  {
      if( debug ) {
          std::cout << "Found" << STR(VL53L0X_IMPLEMENTATION_VER_MAJOR) << "." << STR(VL53L0X_IMPLEMENTATION_VER_MINOR)
                    << "." << STR(VL53L0X_IMPLEMENTATION_VER_SUB) << " rev " << STR(VL53L0X_IMPLEMENTATION_VER_REVISION)
                    << std::endl;
          std::cout << "Requires " << STR(VERSION_REQUIRED_MAJOR) << "." << STR(VERSION_REQUIRED_MINOR) << "."
                    << STR(VERSION_REQUIRED_BUILD) << std::endl;
      }

      Status = VL53L0X_ERROR_NOT_SUPPORTED;

      return false;
  }

  Status = VL53L0X_DataInit( &MyDevice );         // Data initialization

  if (! setAddress(i2c_addr) ) {
    return false;
  }

  Status = VL53L0X_GetDeviceInfo( &MyDevice, &DeviceInfo );

  if( Status == VL53L0X_ERROR_NONE )  {
      if( debug ) {
          std::cout << "VL53L0X Info:" << std::endl;
          std::cout << "Device Name: " << DeviceInfo.Name << ", Type: " << DeviceInfo.Type
                    << ", ID: " << DeviceInfo.ProductId << std::endl;

          std::cout << "Rev Major: " << DeviceInfo.ProductRevisionMajor << ", Minor: " << DeviceInfo.ProductRevisionMinor
                    << std::endl;
      }

      if( ( DeviceInfo.ProductRevisionMinor != 1 ) && ( DeviceInfo.ProductRevisionMinor != 1 ) ) {
          if( debug ) {
            std::cout << "Error expected cut 1.1 but found " << DeviceInfo.ProductRevisionMajor << ','
                      << DeviceInfo.ProductRevisionMinor << std::endl;
          }

          Status = VL53L0X_ERROR_NOT_SUPPORTED;
      }
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      if( debug ) {
          std::cout << "VL53L0X: StaticInit" << std::endl;
      }

      Status = VL53L0X_StaticInit( pMyDevice ); // Device Initialization
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      if( debug ) {
          std::cout << "VL53L0X: PerformRefSpadManagement" << std::endl;
      }

      Status = VL53L0X_PerformRefSpadManagement( pMyDevice, &refSpadCount, &isApertureSpads ); // Device Initialization

      if( debug ) {
          std::cout << "refSpadCount = " << refSpadCount << ", isApertureSpads = " << isApertureSpads << std::endl;
      }
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      if( debug ) {
          std::cout << "VL53L0X: PerformRefCalibration" << std::endl;
      }

      Status = VL53L0X_PerformRefCalibration( pMyDevice, &VhvSettings, &PhaseCal );           // Device Initialization
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      // no need to do this when we use VL53L0X_PerformSingleRangingMeasurement
      if( debug ) {
          std::cout << "VL53L0X: SetDeviceMode" << std::endl;
      }

      Status = VL53L0X_SetDeviceMode( pMyDevice, VL53L0X_DEVICEMODE_SINGLE_RANGING );        // Setup in single ranging mode
  }

  // Enable/Disable Sigma and Signal check
  if( Status == VL53L0X_ERROR_NONE ) {
      Status = VL53L0X_SetLimitCheckEnable( pMyDevice, VL53L0X_CHECKENABLE_SIGMA_FINAL_RANGE, 1 );
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      Status = VL53L0X_SetLimitCheckEnable( pMyDevice, VL53L0X_CHECKENABLE_SIGNAL_RATE_FINAL_RANGE, 1 );
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      Status = VL53L0X_SetLimitCheckEnable( pMyDevice, VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, 1 );
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      Status = VL53L0X_SetLimitCheckValue( pMyDevice, VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, (FixPoint1616_t)( 1.5 * 0.023 * 65536 ) );
  }

  if( Status == VL53L0X_ERROR_NONE ) {
      return true;
  } else {
      if( debug ) {
          std::cout << "VL53L0X Error: " << Status << std::endl;
      }

      return false;
  }
}

/**************************************************************************/
/*! 
    @brief  Change the I2C address of the sensor
    @param  newAddr the new address to set the sensor to
    @returns True if address was set successfully, False otherwise
*/
/**************************************************************************/
bool Vl53l0xMraa::setAddress(uint8_t newAddr) {
  newAddr &= 0x7F;

  Status = VL53L0X_SetDeviceAddress(pMyDevice, newAddr * 2); // 7->8 bit

  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 10 * 1000 * 1000;
  nanosleep(&ts, NULL);

  if( Status == VL53L0X_ERROR_NONE ) {
    pMyDevice->I2cDevAddr = newAddr;  // 7 bit addr
    return true;
  }
  return false;
}

/**************************************************************************/
/*! 
    @brief  get a ranging measurement from the device
    @param  RangingMeasurementData the pointer to the struct the data will be stored in
    @param debug Optional debug flag. If true debug information will print via stdout during execution. Defaults to false.
    @returns True if address was set successfully, False otherwise
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::getSingleRangingMeasurement( VL53L0X_RangingMeasurementData_t *RangingMeasurementData, bool debug )
{
    VL53L0X_Error   Status = VL53L0X_ERROR_NONE;
    FixPoint1616_t  LimitCheckCurrent;


    /*
     *  Step  4 : Test ranging mode
     */

    if( Status == VL53L0X_ERROR_NONE ) {
        if( debug ) {
            std::cout << "VL53L0X: PerformSingleRangingMeasurement" << Status << std::endl;
        }
        Status = VL53L0X_PerformSingleRangingMeasurement( pMyDevice, RangingMeasurementData );

        if( debug ) {
            printRangeStatus( RangingMeasurementData );
        }

        if( debug ) {
            VL53L0X_GetLimitCheckCurrent( pMyDevice, VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, &LimitCheckCurrent );

            std::cout << "RANGE IGNORE THRESHOLD: " << (float)LimitCheckCurrent / 65536.0 << std::endl;

            std::cout << "Measured distance: " << RangingMeasurementData->RangeMilliMeter << std::endl;
        }
    }

    return Status;
}



/**************************************************************************/
/*! 
    @brief  print a ranging measurement out via stdout in a human-readable format
    @param pRangingMeasurementData a pointer to the ranging measurement data
*/
/**************************************************************************/
void Vl53l0xMraa::printRangeStatus( VL53L0X_RangingMeasurementData_t* pRangingMeasurementData )
{
    char buf[ VL53L0X_MAX_STRING_LENGTH ];
    uint8_t RangeStatus;

    /*
     * New Range Status: data is valid when pRangingMeasurementData->RangeStatus = 0
     */

    RangeStatus = pRangingMeasurementData->RangeStatus;

    VL53L0X_GetRangeStatusString( RangeStatus, buf );

    std::cout << "Range Status: " << RangeStatus << " : " << buf << std::endl;
}

/**************************************************************************/
/*!
    @brief  Set the timing budget of one measurement
    @param  ms the timing budget in microseconds
    @returns True if the timing budget was set successfully, False otherwise
*/
/**************************************************************************/
bool Vl53l0xMraa::setMeasurementTimingBudget(uint32_t ms)
{
    Status = VL53L0X_SetMeasurementTimingBudgetMicroSeconds(pMyDevice, ms);
    return (Status == VL53L0X_ERROR_NONE);
}

/**************************************************************************/
/*!
    @brief  get a ranging measurement from the device fastly by skipping ClearInterruptMask
    @param  RangingMeasurementData the pointer to the struct the data will be stored in
    @param debug Optional debug flag. If true debug information will print via stdout during execution. Defaults to false.
    @returns True if address was set successfully, False otherwise
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::getSingleRangingMeasurementFast(VL53L0X_RangingMeasurementData_t *RangingMeasurementData, bool debug)
{
    VL53L0X_Error   Status = VL53L0X_ERROR_NONE;
    FixPoint1616_t  LimitCheckCurrent;

    Status = VL53L0X_SetDeviceMode(pMyDevice, VL53L0X_DEVICEMODE_SINGLE_RANGING);
    if( Status == VL53L0X_ERROR_NONE ) {
        if( debug ) {
            std::cout << "VL53L0X: PerformSingleRangingMeasurement" << Status << std::endl;
        }
        Status = VL53L0X_PerformSingleMeasurement(pMyDevice);
        if ( Status == VL53L0X_ERROR_NONE ) {
            Status = VL53L0X_GetRangingMeasurementData(pMyDevice, RangingMeasurementData);
        }

        if( debug ) {
            printRangeStatus( RangingMeasurementData );
        }

        if( debug ) {
            VL53L0X_GetLimitCheckCurrent( pMyDevice, VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, &LimitCheckCurrent );

            std::cout << "RANGE IGNORE THRESHOLD: " << (float)LimitCheckCurrent / 65536.0 << std::endl;

            std::cout << "Measured distance: " << RangingMeasurementData->RangeMilliMeter << std::endl;
        }
    }

    return Status;
}

/**************************************************************************/
/*!
    @brief  Reset device and wait for the booting up
    @returns True if device is reseted successfully, False otherwise
*/
/**************************************************************************/
bool Vl53l0xMraa::resetDevice()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_ResetDevice(pMyDevice);

    if (Status == VL53L0X_ERROR_NONE)
    {
      return true;
    }

    return false;
}

// getSingleRangingMeasurementFast() = setDeviceModeToSingleRanging() + startMeasurement() +
//         measurementPollForCompletion() + setPalStateToIdle() + getRangingMeasurementData()
// In Single Ranging, startMeasurement() = startSingleRangingWithoutWaitForStop() + waitForSingleRangingToStop()
// Continuous Ranging without interrupt: setDeviceModeToContinuousRanging() + startMeasurement() +
//         (measurementPollForCompletion() + getRangingMeasurementData()) * n + stopMeasurement() + waitStopCompleted()

/**************************************************************************/
/*!
    @brief Set device mode to single ranging (measurement)
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::setDeviceModeToSingleRanging()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_SetDeviceMode(pMyDevice, VL53L0X_DEVICEMODE_SINGLE_RANGING);

    return Status;
}

/**************************************************************************/
/*!
    @brief Set device mode to continuous ranging (measurement)
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::setDeviceModeToContinuousRanging()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_SetDeviceMode(pMyDevice, VL53L0X_DEVICEMODE_CONTINUOUS_RANGING);

    return Status;
}

/**************************************************************************/
/*!
    @brief Only start measurement
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::startMeasurement()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_StartMeasurement(pMyDevice);

    return Status;
}

/**************************************************************************/
/*!
    @brief Only start single ranging without waiting for it to stop (part of startMeasurement())
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::startSingleRangingWithoutWaitForStop()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_WrByte(pMyDevice, 0x80, 0x01);
    Status = VL53L0X_WrByte(pMyDevice, 0xFF, 0x01);
    Status = VL53L0X_WrByte(pMyDevice, 0x00, 0x00);
    Status = VL53L0X_WrByte(pMyDevice, 0x91, PALDevDataGet(pMyDevice, StopVariable));
    Status = VL53L0X_WrByte(pMyDevice, 0x00, 0x01);
    Status = VL53L0X_WrByte(pMyDevice, 0xFF, 0x00);
    Status = VL53L0X_WrByte(pMyDevice, 0x80, 0x00);

    Status = VL53L0X_WrByte(pMyDevice, VL53L0X_REG_SYSRANGE_START, 0x01);

    return Status;
}

/**************************************************************************/
/*!
    @brief Wait for single ranging to stop (part of startMeasurement())
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::waitForSingleRangingToStop()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;
    uint8_t Byte;
    uint8_t StartStopByte = VL53L0X_REG_SYSRANGE_MODE_START_STOP;
    uint32_t LoopNb;

    Byte = StartStopByte;
    /* Wait until start bit has been cleared */
    LoopNb = 0;
    do
    {
        if (LoopNb > 0)
        {
            Status = VL53L0X_RdByte(pMyDevice, VL53L0X_REG_SYSRANGE_START, &Byte);
        }
        LoopNb = LoopNb + 1;
    } while (((Byte & StartStopByte) == StartStopByte) && (Status == VL53L0X_ERROR_NONE) &&
             (LoopNb < VL53L0X_DEFAULT_MAX_LOOP));
    if (LoopNb >= VL53L0X_DEFAULT_MAX_LOOP)
    {
        Status = VL53L0X_ERROR_TIME_OUT;
    }

    return Status;
}

/**************************************************************************/
/*!
    @brief Wait for measurement data ready by polling on the ranging status
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::measurementPollForCompletion()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_measurement_poll_for_completion(pMyDevice);

    return Status;
}

/**************************************************************************/
/*!
    @brief Set state of the PAL for this device to idle
*/
/**************************************************************************/
void Vl53l0xMraa::setPalStateToIdle()
{
    PALDevDataSet(pMyDevice, PalState, VL53L0X_STATE_IDLE);
}

/**************************************************************************/
/*!
    @brief Get state of the PAL for this device
*/
/**************************************************************************/
VL53L0X_State Vl53l0xMraa::getPalState()
{
    VL53L0X_State State = VL53L0X_STATE_IDLE;

    VL53L0X_GetPalState(pMyDevice, &State);

    return State;
}

/**************************************************************************/
/*!
    @brief Get a ranging measurement from the device without starting measurement
    @param RangingMeasurementData the pointer to the struct the data will be stored in
    @param debug Optional debug flag. If true debug information will print via stdout during execution. Defaults to false.
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::getRangingMeasurementData(VL53L0X_RangingMeasurementData_t *RangingMeasurementData, bool debug)
{
    VL53L0X_Error   Status = VL53L0X_ERROR_NONE;
    FixPoint1616_t  LimitCheckCurrent;

    Status = VL53L0X_GetRangingMeasurementData(pMyDevice, RangingMeasurementData);

    if (debug)
    {
        printRangeStatus(RangingMeasurementData);
        VL53L0X_GetLimitCheckCurrent(pMyDevice, VL53L0X_CHECKENABLE_RANGE_IGNORE_THRESHOLD, &LimitCheckCurrent);
        std::cout << "RANGE IGNORE THRESHOLD: " << (float)LimitCheckCurrent / 65536.0 << std::endl;
        std::cout << "Measured distance: " << RangingMeasurementData->RangeMilliMeter << std::endl;
    }

    return Status;
}

/**************************************************************************/
/*!
    @brief Stop measurement
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::stopMeasurement()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;

    Status = VL53L0X_StopMeasurement(pMyDevice);

    return Status;
}

/**************************************************************************/
/*!
    @brief Wait for stopping measurement to complete by polling
    @returns Error status of this device
*/
/**************************************************************************/
VL53L0X_Error Vl53l0xMraa::waitStopCompleted()
{
    VL53L0X_Error Status = VL53L0X_ERROR_NONE;
    uint32_t StopCompleted = 0;
    uint32_t LoopNb;

    // Wait until it finished
    // use timeout to avoid deadlock
    LoopNb = 0;
    do
    {
        Status = VL53L0X_GetStopCompletedStatus(pMyDevice, &StopCompleted);
        if ((StopCompleted == 0x00) || Status != VL53L0X_ERROR_NONE)
        {
            break;
        }
        LoopNb = LoopNb + 1;
        VL53L0X_PollingDelay(pMyDevice);
    } while (LoopNb < VL53L0X_DEFAULT_MAX_LOOP);

    if (LoopNb >= VL53L0X_DEFAULT_MAX_LOOP)
    {
        Status = VL53L0X_ERROR_TIME_OUT;
    }

    return Status;
}
