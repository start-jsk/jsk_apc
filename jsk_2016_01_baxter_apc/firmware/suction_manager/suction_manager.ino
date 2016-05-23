/*
 * Control Suction Modules
 */

#include <ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>

// callbacks
void messageCb(const std_msgs::Bool& toggle_msg, const int pin_no){
  if(toggle_msg.data){
    digitalWrite(pin_no, HIGH);
  } else {
    digitalWrite(pin_no, LOW);
  }
}
void messageCb_D4(const std_msgs::Bool& toggle_msg) { return messageCb(toggle_msg, 4);}
void messageCb_D5(const std_msgs::Bool& toggle_msg) { return messageCb(toggle_msg, 5);}
void messageCb_D6(const std_msgs::Bool& toggle_msg) { return messageCb(toggle_msg, 6);}
void messageCb_D7(const std_msgs::Bool& toggle_msg) { return messageCb(toggle_msg, 7);}

// vacuum
ros::Subscriber<std_msgs::Bool> sub1("/vacuum_gripper/limb/right", &messageCb_D4); //D4
ros::Subscriber<std_msgs::Bool> sub2("/vacuum_gripper/limb/left", &messageCb_D5);
// vent
ros::Subscriber<std_msgs::Bool> sub3("/vacuum_gripper/limb/right_vent", &messageCb_D6);
ros::Subscriber<std_msgs::Bool> sub4("/vacuum_gripper/limb/left_vent", &messageCb_D7);
// sensor
std_msgs::Float64 right_pressure_1, right_pressure_2, left_pressure_1, left_pressure_2;
ros::Publisher right_pressure_1_pub("/vacuum_gripper/limb/right/pressure1/state", &right_pressure_1);
ros::Publisher right_pressure_2_pub("/vacuum_gripper/limb/right/pressure2/state", &right_pressure_2);
ros::Publisher left_pressure_1_pub("/vacuum_gripper/limb/left/pressure1/state", &left_pressure_1);
ros::Publisher left_pressure_2_pub("/vacuum_gripper/limb/left/pressure2/state", &left_pressure_2);

ros::NodeHandle nh;

void setup()
{
  pinMode(13, OUTPUT); // LED
  pinMode(4, OUTPUT); // D4
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(4, INPUT); // A4
  pinMode(5, INPUT); //
  pinMode(6, INPUT); //
  pinMode(7, INPUT); //
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.subscribe(sub1);
  nh.subscribe(sub2);
  nh.subscribe(sub3);
  nh.subscribe(sub4);
  nh.advertise(right_pressure_1_pub);
  nh.advertise(right_pressure_2_pub);
  nh.advertise(left_pressure_1_pub);
  nh.advertise(left_pressure_2_pub);

  digitalWrite(4, LOW);
  digitalWrite(5, LOW);
  digitalWrite(6, HIGH);
  digitalWrite(7, HIGH);


}

void loop()
{
  right_pressure_1.data = analogRead(4);
  right_pressure_2.data = analogRead(5);
  left_pressure_1.data = analogRead(6);
  left_pressure_2.data = analogRead(7);

  right_pressure_1_pub.publish(&right_pressure_1);
  right_pressure_2_pub.publish(&right_pressure_2);
  left_pressure_1_pub.publish(&left_pressure_1);
  left_pressure_2_pub.publish(&left_pressure_2);

  digitalWrite(13, HIGH-digitalRead(13));   // blink the led

  nh.spinOnce();
  delay(100);
}
