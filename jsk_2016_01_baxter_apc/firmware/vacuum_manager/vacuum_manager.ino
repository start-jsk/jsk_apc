#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>

const int RIGHT = 5;
const int LEFT = 6;

ros::NodeHandle nh;

void message_rightCb(const std_msgs::Bool& toggle_msg) {
  if (toggle_msg.data && digitalRead(3)) {
    digitalWrite(RIGHT, HIGH);
  } else {
    digitalWrite(RIGHT, LOW);
  }
}

void message_leftCb(const std_msgs::Bool& toggle_msg) {
  if (toggle_msg.data && digitalRead(3)) {
    digitalWrite(LEFT, HIGH);
  } else {
    digitalWrite(LEFT, LOW);
  }
}

ros::Subscriber<std_msgs::Bool> sub_right("/vacuum_gripper/limb/right", &message_rightCb);
ros::Subscriber<std_msgs::Bool> sub_left("/vacuum_gripper/limb/left", &message_leftCb);

std_msgs::String str_msg_right;
std_msgs::String str_msg_left;

ros::Publisher pub_right("/vacuum_gripper/limb/right/state", &str_msg_right);
ros::Publisher pub_left("/vacuum_gripper/limb/left/state", &str_msg_left);

unsigned long publisher_timer = 0;

void setup()
{
  pinMode(RIGHT, OUTPUT);
  pinMode(LEFT, OUTPUT);
  pinMode(3, INPUT_PULLUP);
  nh.getHardware()->setBaud(115200);
  nh.initNode();
  nh.subscribe(sub_right);
  nh.subscribe(sub_left);
  nh.advertise(pub_right);
  nh.advertise(pub_left);
}

void loop()
{
  if (digitalRead(3) == LOW) {
    digitalWrite(RIGHT, LOW);
    digitalWrite(LEFT, LOW);
  }
  if (millis() > publisher_timer) {
    if (digitalRead(RIGHT)) {
      str_msg_right.data = "ON";
    } else {
      str_msg_right.data = "OFF";
    }
    if (digitalRead(LEFT)) {
      str_msg_left.data = "ON";
    } else {
      str_msg_left.data = "OFF";
    }

    pub_right.publish(&str_msg_right);
    pub_left.publish(&str_msg_left);

    publisher_timer = millis() + 100;
  }
  nh.spinOnce();
}

