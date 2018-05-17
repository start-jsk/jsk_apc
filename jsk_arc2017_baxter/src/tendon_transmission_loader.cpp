// Based on
// https://github.com/ros-controls/ros_control/blob/indigo-devel/transmission_interface/src/four_bar_linkage_transmission_loader.cpp

// Boost
#include <boost/lexical_cast.hpp>

// ROS
#include <ros/console.h>

// Pluginlib
#include <pluginlib/class_list_macros.h>

// ros_control
#include <hardware_interface/internal/demangle_symbol.h>
#include <jsk_arc2017_baxter/tendon_transmission.h>
#include <jsk_arc2017_baxter/tendon_transmission_loader.h>

namespace jsk_arc2017_baxter
{
TendonTransmissionLoader::TransmissionPtr
TendonTransmissionLoader::load(const transmission_interface::TransmissionInfo& transmission_info)
{
  // Transmission should contain only one actuator
  if (!checkActuatorDimension(transmission_info, 1)) {return TransmissionPtr();}

  // Get joint configuration
  std::vector<double> jnt_reduction;
  std::vector<double> jnt_limit;
  const bool jnt_config_ok = getJointConfig(transmission_info,
                                            jnt_reduction,
                                            jnt_limit);

  if (!jnt_config_ok) {return TransmissionPtr();}

  // Transmission instance
  try
  {
    TransmissionPtr transmission(new TendonTransmission(jnt_reduction, jnt_limit));
    return transmission;
  }
  catch(const transmission_interface::TransmissionInterfaceException& ex)
  {
    using hardware_interface::internal::demangledTypeName;
    ROS_ERROR_STREAM_NAMED("parser", "Failed to construct transmission '" << transmission_info.name_ << "' of type '" <<
                           demangledTypeName<TendonTransmission>()<< "'. " << ex.what());
    return TransmissionPtr();
  }
}

bool TendonTransmissionLoader::getJointConfig(const transmission_interface::TransmissionInfo& transmission_info,
                                              std::vector<double>& joint_reduction, std::vector<double>& joint_limit)
{
  const std::vector<transmission_interface::JointInfo>& joints = transmission_info.joints_;
  joint_reduction.resize(joints.size());
  joint_limit.resize(joints.size());
  for (int i = 0; i < joints.size(); i++)
  {
    std::string jnt_name = joints[i].name_;
    TiXmlElement jnt_element = loadXmlElement(joints[i].xml_element_);

    // Parse required mechanical reduction
    const ParseStatus reduction_status = getJointReduction(jnt_element,
                                                           jnt_name,
                                                           transmission_info.name_,
                                                           true, // Required
                                                           joint_reduction[i]);
    if (reduction_status != SUCCESS) {return false;}

    // Parse required joint limit
    const ParseStatus limit_status = getJointLimit(jnt_element,
                                                   jnt_name,
                                                   transmission_info.name_,
                                                   true, // Required
                                                   joint_limit[i]);
    if (limit_status != SUCCESS) {return false;}
  }

  return true;
}

TendonTransmissionLoader::ParseStatus TendonTransmissionLoader::getJointLimit(const TiXmlElement& parent_el,
                                                                              const std::string& joint_name,
                                                                              const std::string& transmission_name,
                                                                              bool required, double& limit)
{
  // Get XML element
  const TiXmlElement* limit_el = parent_el.FirstChildElement("limit");
  if (!limit_el)
  {
    if (required)
    {
      ROS_ERROR_STREAM_NAMED("parser", "Joint '" << joint_name << "' of transmission '" << transmission_name
                                                 << "' does not specify the required <limit> element.");
    }
    else
    {
      ROS_DEBUG_STREAM_NAMED("parser", "Joint '" << joint_name << "' of transmission '" << transmission_name
                                                 << "' does not specify the optional <limit> element.");
    }
    return NO_DATA;
  }

  // Cast to number
  try
  {
    limit = boost::lexical_cast<double>(limit_el->GetText());
  }
  catch (const boost::bad_lexical_cast&)
  {
    ROS_ERROR_STREAM_NAMED("parser", "Joint '" << joint_name << "' of transmission '" << transmission_name
                                               << "' specifies the <limit> element, but is not a number.");
    return BAD_TYPE;
  }
  return SUCCESS;
}

} // namespace

PLUGINLIB_EXPORT_CLASS(jsk_arc2017_baxter::TendonTransmissionLoader,
                       transmission_interface::TransmissionLoader)
