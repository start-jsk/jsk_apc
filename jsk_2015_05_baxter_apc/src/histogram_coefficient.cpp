#include <iostream>
#include <vector>
#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "jsk_recognition_msgs/ColorHistogram.h"
#include "jsk_perception/color_histogram_label_match.h"

class HistogramEfficient {
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_hist1_;
    ros::Subscriber sub_hist2_;
    cv::Mat histogram1_;
    cv::Mat histogram2_;
    // std::vector<float> histogram1_;
    // std::vector<float> histogram2_;
public:
    ros::Publisher result_pub_;
    HistogramEfficient() : nh_() {
        sub_hist1_ = nh_.subscribe("/histogram1", 1,
            &HistogramEfficient::hist1Cb, this);
        sub_hist2_ = nh_.subscribe("/histogram2", 1,
            &HistogramEfficient::hist2Cb, this);
        result_pub_ = nh_.advertise<std_msgs::Float32>(
            "/histogram_coefficient/output", 1);
    }
    void hist1Cb(const jsk_recognition_msgs::ColorHistogramConstPtr& msg)
    {
        ROS_INFO("histgram1");
        histogram1_ = msg->histogram;
        cv::normalize(
    }
    void hist2Cb(const jsk_recognition_msgs::ColorHistogramConstPtr& msg)
    {
        ROS_INFO("histgram2");
        histogram2_ = msg->histogram;
    }
    void coefficients()
    {
        jsk_perception::ColorHistogramLabelMatch::coefficients(

    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "histogram_coefficient");
    ros::Rate loop_rate(10);

    HistogramEfficient histcoef_node;

    while (ros::ok()) {
        std_msgs::Float32 msg;
        msg.data = 1.0;
        histcoef_node.result_pub_.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
