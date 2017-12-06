// Based on
// https://github.com/ros-controls/ros_control/blob/indigo-devel/transmission_interface/include/transmission_interface/four_bar_linkage_transmission_loader.h

#ifndef JSK_ARC2017_BAXTER_TENDON_TRANSMISSION_LOADER_H
#define JSK_ARC2017_BAXTER_TENDON_TRANSMISSION_LOADER_H

// TinyXML
#include <tinyxml.h>

// ros_control
#include <transmission_interface/transmission_loader.h>

namespace jsk_arc2017_baxter
{

class TendonTransmissionLoader : public transmission_interface::TransmissionLoader
{
public:
  TransmissionPtr load(const transmission_interface::TransmissionInfo& transmission_info);

private:
  static bool getJointConfig(const transmission_interface::TransmissionInfo& transmission_info,
                             std::vector<double>& joint_reduction, std::vector<double>& joint_limit);

  static ParseStatus getJointLimit(const TiXmlElement& parent_el, const std::string& joint_name,
                                   const std::string& transmission_name, bool required, double& limit);
};

} // namespace

#endif // header guard
