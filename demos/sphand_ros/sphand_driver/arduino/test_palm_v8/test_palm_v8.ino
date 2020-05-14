#include <Wire.h>
#include <VL53L0X.h>

/***** GLOBAL CONSTANTS *****/

#define VCNL4040_ADDR 0x60 //7-bit unshifted I2C address of VCNL4040

//Command Registers have an upper byte and lower byte.
#define PS_CONF1 0x03
//#define PS_CONF2 //High byte of PS_CONF1
#define PS_CONF3 0x04
//#define PS_MS //High byte of PS_CONF3
#define PS_DATA_L 0x08
//#define PS_DATA_M //High byte of PS_DATA_L

#define NSENSORS 4

VL53L0X tof_sen[NSENSORS];

// PCA9546A
int ChgI2CMultiplexer(unsigned char adrs,unsigned char ch)
{
  unsigned char c;
  int  ans;

  Wire.beginTransmission(adrs);
  c = 1 << ch;
  Wire.write(c);
  ans = Wire.endTransmission();

  return ans;
}

void measure_proximity()
{
  int i;
  for(i=0;i<NSENSORS;i++)
  {
    ChgI2CMultiplexer(0x71, i);
    startProxSensor();
    delay(10);
    Serial.print("intensity");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(readFromCommandRegister(PS_DATA_L));
    Serial.println();
    stopProxSensor();
    delay(1);
  }
}

void initVCNL4040()
{
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
  Serial.begin(9600);
  Wire.begin();

  int i;
  for (i = 0; i < NSENSORS; i++)
  {
    ChgI2CMultiplexer(0x71, i);
    initVCNL4040(); //Configure sensor
    tof_sen[i].init();
    tof_sen[i].setTimeout(500);
  }
}

void loop()
{
  measure_proximity();
  int i;
  int res;
  for (i = 0; i < NSENSORS; i++)
  {
    res = ChgI2CMultiplexer(0x71, i);
    Serial.print("tof");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(tof_sen[i].readRangeSingleMillimeters());
    if (tof_sen[i].timeoutOccurred())
    {
      Serial.print("TIMEOUT");
    }
    Serial.println();
  }
  Serial.print("Multiplexer res: ");
  Serial.println(res);
}
