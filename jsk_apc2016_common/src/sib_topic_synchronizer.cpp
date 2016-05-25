#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <jsk_apc2016_common/SegmentationInBinSync.h>
#include <ros/ros.h>
#include <sstream>

using namespace sensor_msgs;
using namespace message_filters;

Image color_msg;
Image dist_msg;
Image height_msg;
Image mask_msg;

ros::Publisher pub_;

void callback(const ImageConstPtr& color_msg, const ImageConstPtr& dist_msg, const ImageConstPtr& height_msg, const ImageConstPtr& mask_msg)
{
    jsk_apc2016_common::SegmentationInBinSync sync_msg;
    sync_msg.header = color_msg->header;
    sync_msg.color_msg = *color_msg;
    sync_msg.dist_msg = *dist_msg;
    sync_msg.height_msg = *height_msg;
    sync_msg.mask_msg = *mask_msg;
    pub_.publish(sync_msg);
    ros::Duration(0.5).sleep();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rbo_segmentation_in_bin_sync");

    ros::NodeHandle nh("~");

    pub_ = nh.advertise<jsk_apc2016_common::SegmentationInBinSync>("output", 1);

    message_filters::Subscriber<Image> image_sub(nh, "input/image", 1);
    message_filters::Subscriber<Image> dist_sub(nh, "input/dist", 1);
    message_filters::Subscriber<Image> height_sub(nh, "input/height", 1);
    message_filters::Subscriber<Image> mask_sub(nh, "input/mask", 1);

    typedef sync_policies::ApproximateTime<Image, Image, Image, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), image_sub, dist_sub, height_sub, mask_sub);

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));
    ros::spin();
    return 0;
}
