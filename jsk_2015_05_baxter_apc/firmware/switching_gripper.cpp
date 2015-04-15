/*
 * Switch SSR on callback
 */

#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
const int PIN = 13;
const int PRESSURE_SENSOR_PIN = 6;

ros::NodeHandle nh;

// following code did not work well.
// $rostopic pub /on_off_gripper std_msgs/String {data: "ON"}

void messageCb( const std_msgs::Bool& toggle_msg){
    if(toggle_msg.data){
        digitalWrite(PIN, HIGH);
    } else {
        digitalWrite(PIN, LOW);
    }
}

// void messageCb(const std_msgs::Empty& toggle_msg){
//     digitalWrite(PIN, HIGH-digitalRead(PIN));   // Switch SSR
// }

ros::Subscriber<std_msgs::Bool> sub("on_off_gripper", &messageCb);
//ros::Subscriber<std_msgs::Empty> sub("on_off_gripper", &messageCb);

// publish state <on/off>
std_msgs::String str_msg;
ros::Publisher pub("on_off_gripper/state", &str_msg);

// publish whether gripper grabbed.
std_msgs::Bool grabbed_msg;
ros::Publisher grabbed_pub("gripper_grabbed/state", &grabbed_msg);

void setup()
{
    pinMode(PIN, OUTPUT);
    pinMode(PRESSURE_SENSOR_PIN, INPUT);
    nh.initNode();
    nh.subscribe(sub);
    nh.advertise(pub);
    nh.advertise(grabbed_pub);
}

void loop()
{
    // publish gripper on/off.
    if(digitalRead(PIN)){
        str_msg.data = "ON";
    } else {
        str_msg.data = "OFF";
    }
    pub.publish(&str_msg);

    // publish grabbed?
    grabbed_msg.data = digitalRead(PRESSURE_SENSOR_PIN) == HIGH;
    grabbed_pub.publish(&grabbed_msg);

    nh.spinOnce();
    delay(1);
}
