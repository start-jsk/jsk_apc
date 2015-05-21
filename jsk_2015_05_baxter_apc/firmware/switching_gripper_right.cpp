/*
 * Switch SSR on callback
 */

#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
const int PIN = 13;
const int PRESSURE_SENSOR_PIN = 6;
const int DEBUG_BUTTON = 5;
bool prev_debug_state = false;

// ros::NodeHandle nh;
ros::NodeHandle_<ArduinoHardware, 1, 2, 512, 512> nh;

void messageCb( const std_msgs::Bool& toggle_msg){
    if(toggle_msg.data){
        digitalWrite(PIN, HIGH);
    } else {
        digitalWrite(PIN, LOW);
    }
}

ros::Subscriber<std_msgs::Bool> sub("/vacuum_gripper/limb/right", &messageCb);

// publish state <on/off>
std_msgs::String str_msg;
ros::Publisher pub("/vacuum_gripper/limb/right/state", &str_msg);

// publish whether gripper grabbed.
std_msgs::Bool grabbed_msg;
ros::Publisher grabbed_pub("/gripper_grabbed/limb/right/state", &grabbed_msg);

unsigned long publisher_timer;

void setup()
{
    pinMode(PIN, OUTPUT);
    pinMode(PRESSURE_SENSOR_PIN, INPUT);
    pinMode(DEBUG_BUTTON, INPUT);
    nh.getHardware()->setBaud(115200);
    nh.initNode();
    nh.subscribe(sub);
    nh.advertise(pub);
    nh.advertise(grabbed_pub);
}

void loop()
{
    if(millis() > publisher_timer){
        // for debug button
        if(digitalRead(DEBUG_BUTTON) == HIGH){
            digitalWrite(PIN, HIGH);
            prev_debug_state = true;
        } else if(prev_debug_state == true) {
            digitalWrite(PIN, LOW);
            prev_debug_state = false;
        }

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

        publisher_timer = millis() + 100;
    }
    nh.spinOnce();
}
