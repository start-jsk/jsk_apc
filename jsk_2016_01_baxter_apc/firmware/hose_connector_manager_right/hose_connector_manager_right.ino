#include <SPI.h>
#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>

unsigned long int temp_raw, pres_raw;
signed long int t_fine;

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

ros::NodeHandle nh; //write with IDE
// ros::NodeHandle_<ArduinoHardware, 1, 2, 512, 512> nh;

std_msgs::Float64 pressure_msg;
std_msgs::Bool bool_msg;

ros::Publisher pressure_pub("gripper_front/limb/right/pressure/state", &pressure_msg);
ros::Publisher state_pub("gripper_front/limb/right/pressure/grabbed/state", &bool_msg);

unsigned long  publisher_timer = 0;

void setup() {
    nh.getHardware()->setBaud(115200);
    nh.initNode();
    nh.advertise(pressure_pub);
    nh.advertise(state_pub);

    SPI.begin();
    SPI.setClockDivider(SPI_CLOCK_DIV8);
    initBME();
    readTrim();
}


void loop() {
    float temp_act = 0.0, press_act = 0.0;
    unsigned long int press_cal, temp_cal;

    if (millis() > publisher_timer) {
        readData();
        temp_cal = calibration_T(temp_raw);
        press_cal = calibration_P(pres_raw);
        temp_act = (float)temp_cal / 100.0;
        press_act = (float)press_cal / 100.0;

        bool_msg.data = (press_act < 840);
        pressure_msg.data = press_act;

        state_pub.publish(&bool_msg);
        pressure_pub.publish(&pressure_msg);

        publisher_timer = millis() + 100;
    }
    nh.spinOnce();
}

void initBME()
{
    digitalWrite(SS, LOW);
    SPI.transfer((0xF5 & 0x7F));
    SPI.transfer(0xA0);
    SPI.transfer((0xF4 & 0x7F));
    SPI.transfer(0x27);
    digitalWrite(SS, HIGH);
}

void readTrim()
{
    uint8_t data[32];
    int i;
    digitalWrite(SS, LOW);
    SPI.transfer((0x88 | 0x80));
    for (i = 0; i < 24; i++) {
        data[i] = SPI.transfer(0);
    }
    digitalWrite(SS, HIGH);
    delay(1);
    digitalWrite(SS, LOW);
    SPI.transfer((0xA1 | 0x80));
    data[24] = SPI.transfer(0);
    digitalWrite(SS, HIGH);
    delay(1);
    digitalWrite(SS, LOW);
    SPI.transfer((0xE1 | 0x80));
    for (i = 25; i < 32; i++) {
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

void readData()
{
    uint32_t data[8];
    int i;
    digitalWrite(SS, LOW);
    SPI.transfer((0xF7 | 0x80));
    for (i = 0; i < 8; i++) {
        data[i] = SPI.transfer(0x00);
    }
    digitalWrite(SS, HIGH);
    pres_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4);
    temp_raw = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4);
}

signed long int calibration_T(signed long int adc_T)
{

    signed long int var1, var2, T;
    var1 = ((((adc_T >> 3) - ((signed long int)dig_T1 << 1))) * ((signed long int)dig_T2)) >> 11;
    var2 = (((((adc_T >> 4) - ((signed long int)dig_T1)) * ((adc_T >> 4) - ((signed long int)dig_T1))) >> 12) * ((signed long int)dig_T3)) >> 14;

    t_fine = var1 + var2;
    T = (t_fine * 5 + 128) >> 8;
    return T;
}

unsigned long int calibration_P(signed long int adc_P)
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
