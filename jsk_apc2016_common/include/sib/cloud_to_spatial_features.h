#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>

#include <jsk_apc2016_common/BinInfo.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

#include <pcl_ros/publisher.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>


using namespace message_filters;
using namespace sensor_msgs;

typedef boost::shared_ptr< jsk_apc2016_common::BinInfo const> BinInfoPtr;

typedef sync_policies::ApproximateTime<PointCloud2, jsk_apc2016_common::BinInfo> MySyncPolicy;

class SIBSpatial
{
    public:
        SIBSpatial() : listener_(tfBuffer), nh_("~"){};
        ~SIBSpatial();

        void callback(const PointCloud2ConstPtr& cloud_msg_ptr, const BinInfoPtr& target_bin_ptr);

        void onInit();

        void subsribe();

        
        ros::Publisher dist_pub;
        ros::Publisher height_pub;
        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener listener_;
        ros::NodeHandle nh_;

        Synchronizer<MySyncPolicy> * sync;


    private: 
        uint8_t dist(const float x, const float y, const float z, const jsk_recognition_msgs::BoundingBox & bbox);
        uint8_t height(const float z, const jsk_recognition_msgs::BoundingBox & bbox);

};
