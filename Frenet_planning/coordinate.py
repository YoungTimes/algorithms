import numpy as np
from math import *

def cartesian_to_frenet1D(rs, rx, ry, rtheta, x, y):
    s_condition = np.zeros(1)
    d_condition = np.zeros(1)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)    
    
    s_condition[0] = rs
    
    return s_condition, d_condition


def frenet_to_cartesian1D(rs, rx, ry, rtheta, s_condition, d_condition):
    if fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")
        
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]    

    return x, y

def cartesian_to_frenet2D(rs, rx, ry, rtheta, rkappa, x, y, v, theta):
    s_condition = np.zeros(2)
    d_condition = np.zeros(2)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    delta_theta = theta - rtheta
    tan_delta_theta = tan(delta_theta)
    cos_delta_theta = cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    
    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

    return s_condition, d_condition


def frenet_to_cartesian2D(rs, rx, ry, rtheta, rkappa, s_condition, d_condition):
    if fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")
        
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = cos(delta_theta)
    
    theta = NormalizeAngle(delta_theta + rtheta)    
    
    d_dot = d_condition[1] * s_condition[1]
    
    v = sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_condition[1] * s_condition[1] + d_dot * d_dot)   

    return x, y, v, theta 


def NormalizeAngle(angle):
    a = fmod(angle+np.pi, 2*np.pi)
    if a < 0.0:
        a += (2.0*np.pi)        
    return a - np.pi

def cartesian_to_frenet3D(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
    s_condition = np.zeros(3)
    d_condition = np.zeros(3)
    
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    delta_theta = theta - rtheta
    tan_delta_theta = tan(delta_theta)
    cos_delta_theta = cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    
    d_condition[2] = (-kappa_r_d_prime * tan_delta_theta + 
      one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta *
          (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
    
    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = ((a * cos_delta_theta -
                       s_condition[1] * s_condition[1] *
                       (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) /
                          one_minus_kappa_r_d)
    return s_condition, d_condition


def frenet_to_cartesian3D(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
    if fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")
        
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = cos(delta_theta)
    
    theta = NormalizeAngle(delta_theta + rtheta)
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        
    kappa = ((((d_condition[2] + kappa_r_d_prime * tan_delta_theta) *
                 cos_delta_theta * cos_delta_theta) /
                    (one_minus_kappa_r_d) +
                rkappa) *
               cos_delta_theta / (one_minus_kappa_r_d))
    
    
    d_dot = d_condition[1] * s_condition[1]
    
    v = sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_condition[1] * s_condition[1] + d_dot * d_dot)
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * (kappa) - rkappa     
    a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta +
           s_condition[1] * s_condition[1] / cos_delta_theta *
               (d_condition[1] * delta_theta_prime - kappa_r_d_prime))
    return x, y, v, a, theta, kappa 
