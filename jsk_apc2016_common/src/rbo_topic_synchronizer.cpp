#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <jsk_apc2016_common/SegmentationInBinSync.h>
#include <std_msgs/String.h>
#include <ros/ros.h>
#include <sstream>

using namespace sensor_msgs;
using namespace message_filters;

Image stored_image;
CameraInfo stored_caminfo;
PointCloud2 stored_pc;

ros::Publisher pub_;

void callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info, const PointCloud2ConstPtr& points)
{
    Image image_sent;
    image_sent = (Image)*image;

    CameraInfo caminfo_sent;
    caminfo_sent = (CameraInfo) *cam_info;

    PointCloud2 points_sent;
    points_sent = (PointCloud2) *points;

    jsk_apc2016_common::SegmentationInBinSync sync_data;
    sync_data.image_color = image_sent;
    sync_data.cam_info = caminfo_sent;
    sync_data.points = points_sent;
    pub_.publish(sync_data);
    ros::Duration(3).sleep();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "segmentation_in_bin_sync");

  ros::NodeHandle nh("~");


  pub_ = nh.advertise<jsk_apc2016_common::SegmentationInBinSync>("output", 1);

  message_filters::Subscriber<Image> image_sub(nh, "input/image", 1);
  message_filters::Subscriber<CameraInfo> info_sub(nh, "input/info", 1);
  message_filters::Subscriber<PointCloud2> point_sub(nh, "input", 1);

  typedef sync_policies::ApproximateTime<Image, CameraInfo, PointCloud2> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), image_sub, info_sub, point_sub);

  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);
  ros::spin();

  return 0;
}
