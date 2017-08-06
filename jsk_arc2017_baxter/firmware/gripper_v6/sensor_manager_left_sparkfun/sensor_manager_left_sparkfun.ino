#include <SPI.h>
#include <ros.h>
#include <Wire.h>
#include <jsk_arc2017_baxter/GripperSensorStates.h>

#define WIRE Wire

/***** GLOBAL CONSTANTS *****/

#define VCNL4040_ADDR 0x60 //7-bit unshifted I2C address of VCNL4040

//Command Registers have an upper byte and lower byte.
#define PS_CONF1 0x03
//#define PS_CONF2 //High byte of PS_CONF1
#define PS_CONF3 0x04
//#define PS_MS //High byte of PS_CONF3
#define PS_DATA_L 0x08
//#define PS_DATA_M //High byte of PS_DATA_L


#define NSENSORS 2
#define PCA9547D_RESET 32
#define LOOP_TIME 50  // loop duration in ms

/***** ROS *****/
ros::NodeHandle  nh;
jsk_arc2017_baxter::GripperSensorStates gripper_sensor_msg;
ros::Publisher gripper_sensor_pub("lgripper_sensors", &gripper_sensor_msg);
uint32_t proximities[NSENSORS];


uint16_t dig_T1;
int16_t dig_T2;
int16_t dig_T3;
uint16_t dig_P1;
int16_t dig_P2;
int16_t dig_P3;
int16_t dig_P4;
int16_t dig_P5;
int16_t dig_P6;
int16_t dig_P7;
int16_t dig_P8;
int16_t dig_P9;

int ChgI2CMultiplexer(unsigned char adrs,unsigned char ch)
{
  unsigned char c;
  int  ans;

  Wire.beginTransmission(adrs);
  c = ch & 0x07;
  c = c | 0x08;
  Wire.write(c);
  ans = Wire.endTransmission();

  return ans ;
}

void measure_proximity()
{
  int i;
  for(i=0;i<NSENSORS;i++)
  {
    ChgI2CMultiplexer(0x70, i);
    proximities[i] = readFromCommandRegister(PS_DATA_L);
    stopProxSensor();
    delay(1);
  }

  gripper_sensor_msg.proximities = proximities;
  gripper_sensor_msg.proximities_length = NSENSORS;
}

void measure_pressure_and_flex()
{
  float press_act = 0.0;
  unsigned long int pres_raw, temp_raw;
  readData(&pres_raw, &temp_raw);
  press_act = (float)calibration_P(pres_raw, calibration_T(temp_raw)) / 100.0;
  gripper_sensor_msg.pressure = press_act;

  delay(1);

  gripper_sensor_msg.r_finger_flex = analogRead(A0);
  gripper_sensor_msg.l_finger_flex = analogRead(A1);
}


void initVCNL4040()
{
  startProxSensor();

  delay(1);
  //Set the options for PS_CONF3 and PS_MS bytes
  byte conf3 = 0x00;
  byte ms = 0b00000001; //Set IR LED current to 75mA
  //byte ms = 0b00000010; //Set IR LED current to 100mA
  //byte ms = 0b00000110; //Set IR LED current to 180mA
  //byte ms = 0b00000111; //Set IR LED current to 200mA
  writeToCommandRegister(PS_CONF3, conf3, ms);
}

void startProxSensor()
{
  //Clear PS_SD to turn on proximity sensing
  //byte conf1 = 0b00000000; //Clear PS_SD bit to begin reading
  byte conf1 = 0b00001110; //Integrate 8T, Clear PS_SD bit to begin reading
  byte conf2 = 0b00001000; //Set PS to 16-bit
  //byte conf2 = 0b00000000; //Clear PS to 12-bit
  writeToCommandRegister(PS_CONF1, conf1, conf2); //Command register, low byte, high byte
}

void stopProxSensor()
{
  //Set PS_SD to turn off proximity sensing
  byte conf1 = 0b00000001; //Set PS_SD bit to stop reading
  byte conf2 = 0b00000000;
  writeToCommandRegister(PS_CONF1, conf1, conf2); //Command register, low byte, high byte
}

//Reads a two byte value from a command register
unsigned int readFromCommandRegister(byte commandCode)
{
  Wire.beginTransmission(VCNL4040_ADDR);
  Wire.write(commandCode);
  Wire.endTransmission(false); //Send a restart command. Do not release bus.

  Wire.requestFrom(VCNL4040_ADDR, 2); //Command codes have two bytes stored in them

  unsigned int data = Wire.read();
  data |= Wire.read() << 8;

  return (data);
}


//Write a two byte value to a Command Register
void writeToCommandRegister(byte commandCode, byte lowVal, byte highVal)
{
  Wire.beginTransmission(VCNL4040_ADDR);
  Wire.write(commandCode);
  Wire.write(lowVal); //Low byte of command
  Wire.write(highVal); //High byte of command
  Wire.endTransmission(); //Release bus
}


void setup()
{
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.advertise(gripper_sensor_pub);

  pinMode(PCA9547D_RESET, OUTPUT);
  digitalWrite(PCA9547D_RESET, HIGH);
  Wire.begin();

  int i;
  for(i=0;i<NSENSORS;i++)
  {
    ChgI2CMultiplexer(0x70,i);
    initVCNL4040(); //Configure sensor
  }

  pinMode(SS,OUTPUT);
  SPI.setClockDivider(SPI_CLOCK_DIV8);
  SPI.begin();
  initBME();
  readTrim();
}


void loop()
{
  unsigned long time;
  time = millis();

  delay(1);

  measure_proximity();
  measure_pressure_and_flex();

  gripper_sensor_pub.publish(&gripper_sensor_msg);

  int i;
  for(i=0;i<NSENSORS;i++)
  {
    ChgI2CMultiplexer(0x70,i);
    startProxSensor();
  }

  while (millis() < time + LOOP_TIME); // enforce constant loop time

  nh.spinOnce();
}

void initBME()
{
  digitalWrite(SS, LOW);
  SPI.transfer((0xF5 & 0x7F));
  SPI.transfer(0x20);
  SPI.transfer((0xF4 & 0x7F));
  SPI.transfer(0x27);
  digitalWrite(SS, HIGH);
}

void readTrim()
{
  uint8_t data[24];
  int i;
  digitalWrite(SS, LOW);
  SPI.transfer((0x88 | 0x80));
  for (i = 0; i < 24; i++)
  {
    data[i] = SPI.transfer(0);
  }
  digitalWrite(SS, HIGH);

  dig_T1 = (data[1] << 8) | data[0];
  dig_T2 = (data[3] << 8) | data[2];
  dig_T3 = (data[5] << 8) | data[4];
  dig_P1 = (data[7] << 8) | data[6];
  dig_P2 = (data[9] << 8) | data[8];
  dig_P3 = (data[11] << 8) | data[10];
  dig_P4 = (data[13] << 8) | data[12];
  dig_P5 = (data[15] << 8) | data[14];
  dig_P6 = (data[17] << 8) | data[16];
  dig_P7 = (data[19] << 8) | data[18];
  dig_P8 = (data[21] << 8) | data[20];
  dig_P9 = (data[23] << 8) | data[22];
}

void readData(unsigned long int * pres_raw, unsigned long int * temp_raw)
{
  uint8_t data[8];
  int i;
  digitalWrite(SS, LOW);
  SPI.transfer((0xF7 | 0x80));
  for (i = 0; i < 8; i++)
  {
    data[i] = SPI.transfer(0x00);
  }
  digitalWrite(SS, HIGH);
  *pres_raw = data[0];
  *pres_raw = ((*pres_raw) << 8) | data[1];
  *pres_raw = ((*pres_raw) << 4) | (data[2] >> 4);
  *temp_raw = data[3];
  *temp_raw = ((*temp_raw) << 8) | data[4];
  *temp_raw = ((*temp_raw) << 4) | (data[5] >> 4);
}

signed long int calibration_T(signed long int adc_T)
{

  signed long int var1, var2, T;
  var1 = ((((adc_T >> 3) - ((signed long int)dig_T1 << 1))) * ((signed long int)dig_T2)) >> 11;
  var2 = (((((adc_T >> 4) - ((signed long int)dig_T1)) * ((adc_T >> 4) - ((signed long int)dig_T1))) >> 12) * ((signed long int)dig_T3)) >> 14;

  return (var1 + var2);
}

unsigned long int calibration_P(signed long int adc_P, signed long int t_fine)
{
  signed long int var1, var2;
  unsigned long int P;
  var1 = (((signed long int)t_fine) >> 1) - (signed long int)64000;
  var2 = (((var1 >> 2) * (var1 >> 2)) >> 11) * ((signed long int)dig_P6);
  var2 = var2 + ((var1 * ((signed long int)dig_P5)) << 1);
  var2 = (var2 >> 2) + (((signed long int)dig_P4) << 16);
  var1 = (((dig_P3 * (((var1 >> 2) * (var1 >> 2)) >> 13)) >> 3) + ((((signed long int)dig_P2) * var1) >> 1)) >> 18;
  var1 = ((((32768 + var1)) * ((signed long int)dig_P1)) >> 15);
  if (var1 == 0)
  {
      return 0;
  }
  P = (((unsigned long int)(((signed long int)1048576) - adc_P) - (var2 >> 12))) * 3125;
  if (P < 0x80000000)
  {
      P = (P << 1) / ((unsigned long int) var1);
  }
  else
  {
      P = (P / (unsigned long int)var1) * 2;
  }
  var1 = (((signed long int)dig_P9) * ((signed long int)(((P >> 3) * (P >> 3)) >> 13))) >> 12;
  var2 = (((signed long int)(P >> 2)) * ((signed long int)dig_P8)) >> 13;
  P = (unsigned long int)((signed long int)P + ((var1 + var2 + dig_P7) >> 4));
  return P;
}
