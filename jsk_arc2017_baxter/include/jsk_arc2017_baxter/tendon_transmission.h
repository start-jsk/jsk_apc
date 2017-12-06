// Based on
// https://github.com/ros-controls/ros_control/blob/indigo-devel/transmission_interface/include/transmission_interface/four_bar_linkage_transmission.h

#ifndef JSK_ARC2017_BAXTER_TENDON_TRANSMISSION_H
#define JSK_ARC2017_BAXTER_TENDON_TRANSMISSION_H

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <string>
#include <vector>

#include <transmission_interface/transmission.h>
#include <transmission_interface/transmission_interface_exception.h>

namespace jsk_arc2017_baxter
{

class TendonTransmission : public transmission_interface::Transmission
{
public:
  TendonTransmission(const std::vector<double>& joint_reduction,
                     const std::vector<double>& joint_limit);

  void actuatorToJointEffort(const transmission_interface::ActuatorData& act_data,
                                   transmission_interface::JointData&    jnt_data);

  void actuatorToJointVelocity(const transmission_interface::ActuatorData& act_data,
                                     transmission_interface::JointData&    jnt_data);

  void actuatorToJointPosition(const transmission_interface::ActuatorData& act_data,
                                     transmission_interface::JointData&    jnt_data);

  void jointToActuatorEffort(const transmission_interface::JointData&    jnt_data,
                                   transmission_interface::ActuatorData& act_data);

  void jointToActuatorVelocity(const transmission_interface::JointData&    jnt_data,
                                     transmission_interface::ActuatorData& act_data);

  void jointToActuatorPosition(const transmission_interface::JointData&    jnt_data,
                                     transmission_interface::ActuatorData& act_data);

  std::size_t numActuators() const {return 1;}
  std::size_t numJoints()    const {return jnt_reduction_.size();}

  const std::vector<double>& getJointReduction()    const {return jnt_reduction_;}
  const std::vector<double>& getJointLimit()       const {return jnt_limit_;}

protected:
  std::vector<double>  jnt_reduction_;
  std::vector<double>  jnt_limit_;
};

inline TendonTransmission::TendonTransmission(const std::vector<double>& joint_reduction,
                                              const std::vector<double>& joint_limit)
  : transmission_interface::Transmission(),
    jnt_reduction_(joint_reduction),
    jnt_limit_(joint_limit)
{
  if (jnt_reduction_.size() != jnt_limit_.size())
  {
    throw transmission_interface::TransmissionInterfaceException("Reduction and limit vectors of a tendon transmission "
                                                                 "must have the same size.");
  }
  if (boost::algorithm::any_of_equal(jnt_reduction_, 0))
  {
    throw transmission_interface::TransmissionInterfaceException("Transmission reduction ratios cannot be zero.");
  }
}

inline void TendonTransmission::actuatorToJointEffort(const transmission_interface::ActuatorData& act_data,
                                                            transmission_interface::JointData&    jnt_data)
{
  assert(numActuators() == act_data.effort.size() && numJoints() == jnt_data.effort.size());
  assert(act_data.effort[0]);
  BOOST_FOREACH (double* d, jnt_data.effort)
  {
    assert(d);
  }

  std::vector<double>& jr = jnt_reduction_;

  // Distribute actuator effort to joints equally
  for (int i = 0; i < numJoints(); i++)
  {
    *jnt_data.effort[i] = (*act_data.effort[0] / numJoints()) * jr[i];
  }
}

inline void TendonTransmission::actuatorToJointVelocity(const transmission_interface::ActuatorData& act_data,
                                                              transmission_interface::JointData&    jnt_data)
{
  // Call actuatorToJointPosition first
  actuatorToJointPosition(act_data, jnt_data);

  assert(numActuators() == act_data.velocity.size() && numJoints() == jnt_data.velocity.size());
  assert(act_data.velocity[0]);
  BOOST_FOREACH (double* d, jnt_data.velocity)
  {
    assert(d);
  }

  std::vector<double>& jr = jnt_reduction_;
  std::vector<double>& jl = jnt_limit_;
  std::vector<double*> movable_jnt_vel;

  // Distribute actuator velocity to movable joints, which don't collide to limits
  for (int i = 0; i < numJoints(); i++)
  {
    if ((jl[i] > 0 && *jnt_data.position[i] < jl[i]) || (jl[i] < 0 && *jnt_data.position[i] > jl[i]))
    {
      movable_jnt_vel.push_back(jnt_data.velocity[i]);
    }
    else
    {
      *jnt_data.velocity[i] = 0;
    }
  }
  BOOST_FOREACH (double* jv, movable_jnt_vel)
  for (int i = 0; i < movable_jnt_vel.size(); i++)
  {
    *movable_jnt_vel[i] = *act_data.velocity[0] / movable_jnt_vel.size() / jr[i];
  }
}

inline void TendonTransmission::actuatorToJointPosition(const transmission_interface::ActuatorData& act_data,
                                                              transmission_interface::JointData&    jnt_data)
{
  assert(numActuators() == act_data.position.size() && numJoints() == jnt_data.position.size());
  assert(act_data.position[0]);
  BOOST_FOREACH (double* d, jnt_data.position)
  {
    assert(d);
    *d = 0;
  }

  std::vector<double>& jr = jnt_reduction_;
  std::vector<double>& jl = jnt_limit_;
  double rest_act_pos = *act_data.position[0];
  bool is_remaining;
  int limited_jnt_num = 0;

  // Distribute actuator position to joints
  do
  {
    double rest_act_pos_div = rest_act_pos / (numJoints() - limited_jnt_num);
    is_remaining = false;
    for (int i = 0; i < numJoints(); i++)
    {
      // Try to distribute position equally
      double rest_jnt_pos = rest_act_pos_div / jr[i];
      double curr_pos = *jnt_data.position[i];
      // Select movable joints only
      if ((jl[i] > 0 && curr_pos < jl[i]) || (jl[i] < 0 && curr_pos > jl[i]))
      {
        curr_pos += rest_jnt_pos;
        if ((jl[i] > 0 && curr_pos < jl[i]) || (jl[i] < 0 && curr_pos > jl[i]))
        {
          *jnt_data.position[i] = curr_pos;
        }
        else
        {
          // Detect joint collision to limit
          *jnt_data.position[i] = jl[i];
          limited_jnt_num++;
          is_remaining = true;
        }
        rest_act_pos -= *jnt_data.position[i] * jr[i];
      }
    }
    // Loop until whole actuator position is distributed or all joints are limited
  } while (is_remaining && limited_jnt_num < numJoints());
}

inline void TendonTransmission::jointToActuatorEffort(const transmission_interface::JointData&    jnt_data,
                                                            transmission_interface::ActuatorData& act_data)
{
  assert(numActuators() == act_data.effort.size() && numJoints() == jnt_data.effort.size());
  assert(act_data.effort[0]);
  BOOST_FOREACH (double* d, jnt_data.effort)
  {
    assert(d);
  }

  std::vector<double>& jr = jnt_reduction_;

  // Sum joint effort for actuator effort
  *act_data.effort[0] = 0;
  for (int i = 0; i < numJoints(); i++)
  {
    *act_data.effort[0] += *jnt_data.effort[i] / jr[i];
  }
}

inline void TendonTransmission::jointToActuatorVelocity(const transmission_interface::JointData&    jnt_data,
                                                              transmission_interface::ActuatorData& act_data)
{
  assert(numActuators() == act_data.velocity.size() && numJoints() == jnt_data.velocity.size());
  assert(act_data.velocity[0]);
  BOOST_FOREACH (double* d, jnt_data.velocity)
  {
    assert(d);
  }

  std::vector<double>& jr = jnt_reduction_;

  // Sum joint velocity for actuator velocity
  *act_data.velocity[0] = 0;
  for (int i = 0; i < numJoints(); i++)
  {
    *act_data.velocity[0] += *jnt_data.velocity[i] * jr[i];
  }
}

inline void TendonTransmission::jointToActuatorPosition(const transmission_interface::JointData&    jnt_data,
                                                              transmission_interface::ActuatorData& act_data)
{
  assert(numActuators() == act_data.position.size() && numJoints() == jnt_data.position.size());
  assert(act_data.position[0]);
  BOOST_FOREACH (double* d, jnt_data.position)
  {
    assert(d);
  }

  std::vector<double>& jr = jnt_reduction_;

  // Sum joint position for actuator position
  *act_data.position[0] = 0;
  for (int i = 0; i < numJoints(); i++)
  {
    *act_data.position[0] += *jnt_data.position[i] * jr[i];
  }
}

} // jsk_arc2017_baxter

#endif // JSK_ARC2017_BAXTER_TENDON_TRANSMISSION_H
