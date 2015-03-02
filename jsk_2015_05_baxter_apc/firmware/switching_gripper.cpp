/*
 * Switch SSR on callback
 */

#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
const int PIN = 13;

ros::NodeHandle  nh;

// following code did not work well.
// $rostopic pub /on_off_gripper std_msgs/String {data: "ON"}

// void messageCb( const std_msgs::String& toggle_msg){
//     if(toggle_msg.data == "ON" || toggle_msg.data == "on"){
//         digitalWrite(PIN, HIGH);
//     } else if(toggle_msg.data == "OFF" || toggle_msg.data == "off"){
//         digitalWrite(PIN, LOW);
//     }
//     //digitalWrite(PIN, HIGH-digitalRead(PIN));   // Switch SSR
// }

void messageCb(const std_msgs::Empty& toggle_msg){
    digitalWrite(PIN, HIGH-digitalRead(PIN));   // Switch SSR
}

//ros::Subscriber<std_msgs::String> sub("on_off_gripper", &messageCb);
ros::Subscriber<std_msgs::Empty> sub("on_off_gripper", &messageCb);

// publish state <on/off>
std_msgs::String str_msg;
ros::Publisher pub("on_off_gripper/state", &str_msg);

void setup()
{
    pinMode(PIN, OUTPUT);
    nh.initNode();
    nh.subscribe(sub);
    nh.advertise(pub);
}

void loop()
{
    if(digitalRead(PIN)){
        str_msg.data = "ON";
    } else {
        str_msg.data = "OFF";
    }
    pub.publish(&str_msg);
    nh.spinOnce();
    delay(1);
}
