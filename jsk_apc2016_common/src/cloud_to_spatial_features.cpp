#include <sib/cloud_to_spatial_features.h>


using namespace message_filters;
using namespace sensor_msgs;


void SIBSpatial::onInit()
{
    dist_pub = nh_.advertise<sensor_msgs::Image>("dist", 5);
    height_pub = nh_.advertise<sensor_msgs::Image>("height", 5);
}

void SIBSpatial::subsribe() {
    message_filters::Subscriber<PointCloud2> point_sub(nh_, "input", 1);
    message_filters::Subscriber<jsk_apc2016_common::BinInfo> target_bin_sub(nh_, "target_bin", 1);

    sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(100), point_sub, target_bin_sub);

    sync->registerCallback(boost::bind(&SIBSpatial::callback, this, _1, _2));

    ros::spin();
}

void SIBSpatial::callback(const PointCloud2ConstPtr& cloud_msg_ptr, const BinInfoPtr& target_bin_ptr)
{
    float pos_x;
    float pos_y;
    float pos_z;
    Eigen::Affine3d cloud2bb_aff;
    Eigen::Matrix4f cloud2bb_mat;

    sensor_msgs::PointCloud2 cloud_msg_transformed;

    jsk_apc2016_common::BinInfo target_bin = (jsk_apc2016_common::BinInfo) *target_bin_ptr;

    /*
     * do transformation on cloud
     */
    ros::Time beginning = ros::Time::now() + ros::Duration(0.5);

    // This takes 4 seconds to complete...
    ros::Time start_tf = ros::Time::now();
    geometry_msgs::TransformStamped cloud2bb;

    if (!tfBuffer.canTransform(/*target_frame*/target_bin.header.frame_id, /*source_frame*/cloud_msg_ptr->header.frame_id,ros::Time(0), ros::Duration(0.5)))
    {
        ROS_WARN("cloud_spatial_features could not find tf");  
        return;
    }

    cloud2bb = tfBuffer.lookupTransform(/*target_frame*/target_bin.header.frame_id, /*source_frame*/cloud_msg_ptr->header.frame_id,ros::Time(0));

    cloud2bb_aff = tf2::transformToEigen(cloud2bb);
    cloud2bb_mat = (cloud2bb_aff.matrix()).cast<float>();
    
    pcl_ros::transformPointCloud(cloud2bb_mat, *cloud_msg_ptr, cloud_msg_transformed);


//    tfBuffer.transform(*cloud_msg_ptr, cloud_msg_transformed, target_bin.header.frame_id, beginning, cloud_msg_ptr->header.frame_id, ros::Duration(10.0));
    ros::Time finish_tf = ros::Time::now();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        cloud_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(cloud_msg_transformed, *cloud_transformed);

    /*
     * get dist and height from the target bin
     * ref:  pcl::toROSMsg (CloudT, sensor_msgs::Image) 
     */
    sensor_msgs::Image dist_msg;
    dist_msg.height = cloud_transformed->height;
    dist_msg.width = cloud_transformed->width;
    dist_msg.encoding = "mono8";
    dist_msg.step = dist_msg.width * sizeof(uint8_t);
    dist_msg.data.resize(dist_msg.step * dist_msg.height);

    sensor_msgs::Image height_msg;
    height_msg.height = cloud_transformed->height;
    height_msg.width = cloud_transformed->width;
    height_msg.encoding = "mono8";
    height_msg.step = height_msg.width * sizeof(uint8_t);
    height_msg.data.resize(height_msg.step * height_msg.height);

    ros::Time start_loop = ros::Time::now();
    for (size_t v = 0; v < cloud_transformed->height; v++)
    {
        for (size_t u = 0; u < cloud_transformed->width; u++)
        {
            pos_x = cloud_transformed->at(u, v).x;
            pos_y = cloud_transformed->at(u, v).y;
            pos_z = cloud_transformed->at(u, v).z;
            dist_msg.data[v * dist_msg.step + u * sizeof(uint8_t)] = dist(pos_x, pos_y, pos_z, target_bin_ptr->bbox);
            height_msg.data[v * dist_msg.step + u * sizeof(uint8_t)] = height(pos_z, target_bin_ptr->bbox);
        }
    }
    ros::Time finish_loop = ros::Time::now();

    // frame id is that of camera
    dist_msg.header = cloud_msg_ptr->header;
    height_msg.header = cloud_msg_ptr->header;
    dist_pub.publish(dist_msg);
    height_pub.publish(height_msg);


    std::cout << finish_tf - start_tf << std::endl;
    std::cout << finish_loop - start_loop << std::endl;
}

inline
uint8_t SIBSpatial::dist(const float x, const float y, const float z, const jsk_recognition_msgs::BoundingBox & bbox) {
    using namespace std;
    float dist_x;
    float dist_y;
    float dist_z;
    float min_dist;

    if (abs(x) <= bbox.dimensions.x/2)
        dist_x = bbox.dimensions.x/2 - x;
    else
        dist_x = 0;

    if (y >= 0 && y < bbox.dimensions.y/2)
        dist_y = abs(bbox.dimensions.y/2 - y);
    else if(y < 0 && abs(y) < bbox.dimensions.y/2)
        dist_y = abs(y + bbox.dimensions.y/2);
    else
        dist_y = 0;

    if (z >= 0 && z < bbox.dimensions.z/2)
        dist_z = abs(bbox.dimensions.z/2 - z);
    else if(z < 0 && abs(z) < bbox.dimensions.z/2)
        dist_z = abs(z + bbox.dimensions.z/2);
    else
        dist_z = 0;

    // in mm
    min_dist = 1000 * min(min(dist_x, dist_y), dist_z);
    return static_cast<uint8_t>(min_dist);  
}

inline
uint8_t SIBSpatial::height(const float z, const jsk_recognition_msgs::BoundingBox & bbox) {
    using namespace std;
    float dist_z;

    if (abs(z) <= bbox.dimensions.z/2)
    {
        dist_z = bbox.dimensions.z/2 + z;
    }
    else
        dist_z = 0;
    // following RBO's metric
    return static_cast<uint8_t>(dist_z * 2 * 255);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "segmentation_in_bin_sync");
    SIBSpatial* sib = new SIBSpatial();
    sib->onInit();
    sib->subsribe();
    return 0;
}
