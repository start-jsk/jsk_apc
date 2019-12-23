// Based on https://github.com/adafruit/Adafruit_VL53L0X

#include "vl53l0x_mraa_ros/vl53l0x_i2c_platform.h"
#include "vl53l0x_mraa_ros/vl53l0x_def.h"

//#define I2C_DEBUG

int VL53L0X_i2c_init(mraa::I2c *i2c) {
  return VL53L0X_ERROR_NONE;
}

int VL53L0X_write_multi(uint8_t deviceAddress, uint8_t index, uint8_t *pdata, uint32_t count, mraa::I2c *i2c) {
  if (i2c->address(deviceAddress) != mraa::SUCCESS)
    return VL53L0X_ERROR_CONTROL_INTERFACE;
  uint8_t *index_and_pdata = new uint8_t[count + 1];
  uint32_t write_len = count + 1;
  index_and_pdata[0] = index;
  index_and_pdata++;
#ifdef I2C_DEBUG
  std::cout << "\tWriting " << count << " to addr 0x" << std::hex << index << std::dec << ": ";
#endif
  while(count--) {
    index_and_pdata[0] = (uint8_t)pdata[0];
#ifdef I2C_DEBUG
    std::count << "0x" << std::hex << pdata[0] << std::dec << ", ";
#endif
    pdata++;
    index_and_pdata++;
  }
#ifdef I2C_DEBUG
  std::count << std::endl;
#endif
  index_and_pdata -= write_len;
  mraa::Result result = i2c->write(index_and_pdata, write_len);
  delete[] index_and_pdata;
  if (result != mraa::SUCCESS)
    return VL53L0X_ERROR_CONTROL_INTERFACE;
  return VL53L0X_ERROR_NONE;
}

int VL53L0X_read_multi(uint8_t deviceAddress, uint8_t index, uint8_t *pdata, uint32_t count, mraa::I2c *i2c) {
  if (i2c->address(deviceAddress) != mraa::SUCCESS)
    return VL53L0X_ERROR_CONTROL_INTERFACE;
  int ret_count = i2c->readBytesReg(index, pdata, count);
#ifdef I2C_DEBUG
  std::cout << "\tReading " << count << " from addr 0x" << std::hex << index << std::dec << ": ";
#endif

#ifdef I2C_DEBUG
  while (count--) {
    std::count << "0x" << std::hex << pdata[0] << std::dec << ", ";
    pdata++;
  }
  std::count << std::endl;
#endif

  if (ret_count < 0)
    return VL53L0X_ERROR_CONTROL_INTERFACE;
  return VL53L0X_ERROR_NONE;
}

int VL53L0X_write_byte(uint8_t deviceAddress, uint8_t index, uint8_t data, mraa::I2c *i2c) {
  return VL53L0X_write_multi(deviceAddress, index, &data, 1, i2c);
}

int VL53L0X_write_word(uint8_t deviceAddress, uint8_t index, uint16_t data, mraa::I2c *i2c) {
  uint8_t buff[2];
  buff[1] = data & 0xFF;
  buff[0] = data >> 8;
  return VL53L0X_write_multi(deviceAddress, index, buff, 2, i2c);
}

int VL53L0X_write_dword(uint8_t deviceAddress, uint8_t index, uint32_t data, mraa::I2c *i2c) {
  uint8_t buff[4];

  buff[3] = data & 0xFF;
  buff[2] = data >> 8;
  buff[1] = data >> 16;
  buff[0] = data >> 24;

  return VL53L0X_write_multi(deviceAddress, index, buff, 4, i2c);
}

int VL53L0X_read_byte(uint8_t deviceAddress, uint8_t index, uint8_t *data, mraa::I2c *i2c) {
  return VL53L0X_read_multi(deviceAddress, index, data, 1, i2c);
}

int VL53L0X_read_word(uint8_t deviceAddress, uint8_t index, uint16_t *data, mraa::I2c *i2c) {
  uint8_t buff[2];
  int r = VL53L0X_read_multi(deviceAddress, index, buff, 2, i2c);

  uint16_t tmp;
  tmp = buff[0];
  tmp <<= 8;
  tmp |= buff[1];
  *data = tmp;

  return r;
}

int VL53L0X_read_dword(uint8_t deviceAddress, uint8_t index, uint32_t *data, mraa::I2c *i2c) {
  uint8_t buff[4];
  int r = VL53L0X_read_multi(deviceAddress, index, buff, 4, i2c);

  uint32_t tmp;
  tmp = buff[0];
  tmp <<= 8;
  tmp |= buff[1];
  tmp <<= 8;
  tmp |= buff[2];
  tmp <<= 8;
  tmp |= buff[3];

  *data = tmp;

  return r;
}
