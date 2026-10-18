// NERO (session 4): NERO_DOS_CREEP="v_creep,t_react,a_stop,acc_th[,hold_s]" — a vehicle that creeps (<= v_creep for the
// last hold_s seconds, not accelerating faster than acc_th) is projected forward only by the distance it needs to stop
// (v*t_react + v^2 / (2*a_stop)) instead of v*time_horizon: Saudi-style drivers roll toward a turning ego at 1-2 m/s and
// stop just short; with the 8 s projection every one of them stopped ego inside the junction. Unset = upstream.
#ifndef NERO_CREEP_HPP_
#define NERO_CREEP_HPP_

#include <autoware_perception_msgs/msg/predicted_object.hpp>

namespace autoware::motion_velocity_planner::dynamic_obstacle_stop
{
struct NeroCreepParam
{
  double v_creep{-1.0};
  double t_react{0.5};
  double a_stop{2.0};
  double acc_th{0.3};
  double hold_s{2.0};
};
const NeroCreepParam & nero_creep_param();
// record the object's speed (call once per cycle for every vehicle, before any speed filter)
void nero_creep_observe(const autoware_perception_msgs::msg::PredictedObject & object);
// the forward reach to use for this object if it is a creeper (< 0: not a creeper, use upstream)
double nero_creep_reach(const autoware_perception_msgs::msg::PredictedObject & object);
// a slow vehicle that is pulling away (>= 0.5 m/s and acc_th faster than its slowest in the last second): it is kept in the
// check below minimum_object_velocity (a creeper that goes the moment ego goes)
bool nero_creep_launching(const autoware_perception_msgs::msg::PredictedObject & object);
}  // namespace autoware::motion_velocity_planner::dynamic_obstacle_stop
#endif  // NERO_CREEP_HPP_
