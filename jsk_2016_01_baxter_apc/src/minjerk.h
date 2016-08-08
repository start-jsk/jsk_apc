#include <vector>
#include <math.h>

class Interpolator {
protected:
  std::vector<double> position_list;    // list of control point
  std::vector<double> time_list;        // list of time[sec] from start for each control point

  double position;      //current data point
  double time;          // time [sec] from start
  int segment_num;      // number of total segment
  double segment_time;  // time[sec] with in each segment
  int segment;          // index of segment which is currently processing
  bool interpolatingp;

public:
  Interpolator() {
    time = 0.0;
    segment_time = 0.0;
    segment = 0;
    segment_num = 0;
    interpolatingp = false;
  }
  void Reset(std::vector<double> pl, std::vector<double> tl) {
    position_list = pl;
    time_list = tl;
    time = 0.0;
    segment_time = 0.0;
    segment = 0;
    segment_num = position_list.size() - 1;
    interpolatingp = false;
  }
  void StartInterpolation() { interpolatingp = true; }
  void StopInterpolation() { interpolatingp = false; }
  double PassTime(double dt) {
    if ( interpolatingp ) {
      position = Interpolation();
      time += dt;
      segment_time += dt;
      if ( time > time_list[segment] ) {
        segment_time = time - time_list[segment];
        segment++;
      }
      if ( segment >= segment_num) { Reset(position_list, time_list); }
    }
    return position;
  }
  virtual double Interpolation() {};
};

class MinJerk : public Interpolator {
private:
  std::vector<double> velocity_list;
  double velocity;
  std::vector<double> acceleration_list;
  double acceleration;
public:
  MinJerk() : Interpolator() {
  }
  void Reset(std::vector<double> pl, std::vector<double> vl, std::vector<double> al, std::vector<double> tl) {
    Interpolator::Reset(pl, tl);
    velocity_list = vl;
    acceleration_list = al;
  }
  virtual double Interpolation() {
    double xi = position_list[segment];
    double xf = position_list[segment+1];
    double vi = velocity_list[segment];
    double vf = velocity_list[segment+1];
    double ai = acceleration_list[segment];
    double af = acceleration_list[segment+1];

    double t = time_list[segment] - ((segment>0)?(time_list[segment - 1]):0); // total time of segment
    double A = (xf-(xi+vi*t+(ai/2.0)*t*t))/(t*t*t);
    double B = (vf-(vi+ai*t))/(t*t);
    double C = (af-ai)/t;
    double a0 = xi;
    double a1 = vi;
    double a2 = ai/2.0;
    double a3 = 10*A-4*B+0.5*C;
    double a4 = (-15*A+7*B-C)/t;
    double a5 = (6*A-3*B+0.5*C)/(t*t);

    position = a0+a1*pow(segment_time,1)+a2*pow(segment_time,2)+a3*pow(segment_time,3)+a4*pow(segment_time,4)+a5*pow(segment_time,5);
    velocity = a1+2*a2*pow(segment_time,1)+3*a3*pow(segment_time,2)+4*a4*pow(segment_time,3)+5*a5*pow(segment_time,4);
    acceleration = 2*a2+6*a3*pow(segment_time,1)+12*a4*pow(segment_time,2)+20*a5*pow(segment_time,3);

    return position;
  }
};
