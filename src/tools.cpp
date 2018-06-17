#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

#define ABS_MIN_VAL 0.0000001

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() == 0u) return(rmse);
  if (estimations.size() != ground_truth.size()) return(rmse);

  //accumulate squared residuals
  for (unsigned int i=0u; i < estimations.size(); ++i)
  {
    VectorXd c = estimations[i] - ground_truth[i];
    c = c.array() * c.array();
    rmse += c;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return(rmse);
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float p2 = px*px + py*py;

  //check division by zero
  if (fabs(p2) < ABS_MIN_VAL) p2 = ABS_MIN_VAL;

  //compute the Jacobian matrix
  float wp2 = sqrt(p2);
  float wp3 = p2*wp2;

  Hj << px/wp2, py/wp2, 0, 0,
        -py/p2, px/p2, 0, 0,
        py*(vx*py - vy*px)/wp3, px*(vy*px - vx*py)/wp3,
        px/wp2, py/wp2;
  
  return(Hj);
}
