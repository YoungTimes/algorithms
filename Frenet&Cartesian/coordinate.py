import numpy as np
from math import *
import matplotlib.pyplot as plt
from cubic_spline import *

def frenet_to_cartesian1D(rs, rx, ry, rtheta, s_condition, d_condition):
    if fabs(rs - s_condition[0])>= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")
        
    cos_theta_r = cos(rtheta)
    sin_theta_r = sin(rtheta)
    
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]    
    return x, y

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


waypoints_np   = None
waypoints_file = "./racetrack_waypoints.txt"
with open(waypoints_file) as waypoints_file_handle:
    waypoints = list(csv.reader(waypoints_file_handle, 
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
    waypoints_np = np.array(waypoints)

x = waypoints_np[:, 0]
y = waypoints_np[:, 1]

ego = np.array([-126.87, -526.30])
    
ds = 0.1  # [m] distance of each interpolated points

sp = Spline2D(x, y)
s = np.arange(0, sp.s[-1], ds)

left_bound = []
right_bound = []
new_x = []

for i_s in s:
    rx, ry = sp.calc_position(i_s)
    rtheta = sp.calc_yaw(i_s)

    new_x.append(np.array([rx, ry]))

    l_s_condition = np.array([i_s])
    l_d_condition = np.array([1.5])
    l_x, l_y = frenet_to_cartesian1D(i_s, rx, ry, rtheta, l_s_condition, l_d_condition)
    left_bound.append(np.array([l_x, l_y]))
    
    r_s_condition = np.array([i_s])
    r_d_condition = np.array([-1.5])
    r_x, r_y = frenet_to_cartesian1D(i_s, rx, ry, rtheta, r_s_condition, r_d_condition)   
    right_bound.append(np.array([r_x, r_y]))

left_bound = np.array(left_bound)
right_bound = np.array(right_bound)
new_x = np.array(new_x)

def find_nearest_rs(sp, x, y, s):
    min_dist = float('inf')
    rs = 0.0
    rs_idx = 0
    for i in range(len(s)):
        i_s = s[i]
        dx = x - sp.calc_position(i_s)[0]
        dy = y - sp.calc_position(i_s)[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if min_dist > dist:
            min_dist = dist
            rs = i_s
            rs_idx = i

    return rs, rs_idx

def polyfit(coeffs, t):
    return coeffs[0] + coeffs[1] * t + coeffs[2] * t * t + coeffs[3] * t * t * t

def calc_cubic_poly_curve_coeffs(x_s, y_s, y_s_1d, x_e, y_e, y_e_1d):
    A = np.array([
        [1, x_s, math.pow(x_s, 2.0), math.pow(x_s, 3.0)],
        [0, 1,   2 * x_s,       3 * math.pow(x_s, 2.0)],
        [1, x_e, math.pow(x_e, 2.0), math.pow(x_e, 3.0)],
        [0, 1,   2 * x_e,       3 * math.pow(x_e, 2.0)]
    ])

    A = A.astype(np.float)

    b = np.array([[y_s], [y_s_1d], [y_e], [y_e_1d]])

    A_inv = np.linalg.inv(A)

    coffes = np.dot(A_inv, b)

    return coffes

rs, idx = find_nearest_rs(sp, ego[0], ego[1], s)

ego_s, ego_d = cartesian_to_frenet1D(rs, sp.calc_position(rs)[0], sp.calc_position(rs)[1], sp.calc_yaw(rs), ego[0], ego[1])

planning_horizon = 200 * ds

targets = []
for i in range(11):
    x0 = ego_s
    y0 = ego_d
    dx0 = 0

    x1 = ego_s + planning_horizon
    y1 = -1.5 + i * 0.3
    dx1 = 0

    coffes = calc_cubic_poly_curve_coeffs(x0, y0, dx0, x1, y1, dx1)

    targets.append(coffes)

plt.plot(x, y, 'r')
plt.plot(new_x[:, 0], new_x[:, 1], 'y')
plt.plot(left_bound[:,0],left_bound[:,1], 'b')
plt.plot(right_bound[:,0],right_bound[:,1], 'b')

plt.plot(ego[0], ego[1], 'o')

t = np.linspace(ego_s, ego_s + planning_horizon, 500)

for i in range(11):
    coeffs = targets[i]
    d = polyfit(coeffs, t)
    traj = []
    for j in range(500):
        i_s = s[idx + j]
        rx, ry = sp.calc_position(i_s)
        rtheta = sp.calc_yaw(i_s)
        
        s_condition = np.array([i_s])
        d_condition = np.array([d[j]])
        x, y = frenet_to_cartesian1D(i_s, rx, ry, rtheta, s_condition, d_condition)
        traj.append(np.array([x, y]))
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], 'g')
plt.show()
